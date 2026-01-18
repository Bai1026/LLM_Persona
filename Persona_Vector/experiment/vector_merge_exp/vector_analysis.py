#!/usr/bin/env python3
"""
Vector Analysis - API 對話分析器
這個腳本會讀取問題資料集，透過本地 API 進行對話，並將結果儲存為結構化的 JSON 檔案
"""

import json
import requests
import time
from datetime import datetime
from typing import Dict, List, Any
from rich import print
from rich.console import Console
from rich.progress import track, Progress, TaskID
from rich.table import Table
from pathlib import Path

class ConversationAnalyzer:
    """對話分析器類別"""
    
    def __init__(self, api_url: str = "http://localhost:5001/chat", dataset_path: str = "./persona_trait_data/neutral_task/dataset.json"):
        self.api_url = api_url
        self.dataset_path = dataset_path
        self.console = Console()
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "api_url": api_url,
                "dataset_path": dataset_path,
                "total_conversations": 0,
                "successful_conversations": 0,
                "failed_conversations": 0
            },
            "conversations": []
        }
    
    def load_dataset(self) -> Dict[str, str]:
        """載入問題資料集"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.console.print(f"✅ 成功載入 {len(data)} 個問題", style="green")
            return data
        except Exception as e:
            self.console.print(f"❌ 載入資料集失敗: {e}", style="red")
            return {}
    
    def chat_with_api(self, user_input: str) -> Dict[str, Any]:
        """透過 API 進行對話"""
        try:
            response = requests.post(
                self.api_url, 
                json={'user_input': user_input},
                timeout=300
            )
            response.raise_for_status()
            return {
                "success": True,
                "data": response.json(),
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds()
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            }
    
    def process_conversations(self, dataset: Dict[str, str]) -> None:
        """處理所有對話"""
        self.results["metadata"]["total_conversations"] = len(dataset)
        
        with Progress() as progress:
            task = progress.add_task("處理對話中...", total=len(dataset))
            
            for idx, (topic, question) in enumerate(dataset.items(), 1):
                # 準備輸入
                user_input = f"{topic}: {question}"
                
                # 呼叫 API
                api_result = self.chat_with_api(user_input)
                
                # 建立對話記錄
                conversation = {
                    "id": idx,
                    "topic": topic,
                    "question": question,
                    "user_input": user_input,
                    "timestamp": datetime.now().isoformat(),
                }
                
                if api_result["success"]:
                    conversation.update({
                        "status": "success",
                        "response": api_result["data"].get("response", ""),
                        "response_time": api_result["response_time"],
                        "api_status_code": api_result["status_code"]
                    })
                    self.results["metadata"]["successful_conversations"] += 1
                    
                    # 顯示成功資訊
                    self.console.print(f"✅ [{idx}/{len(dataset)}] {topic[:50]}...", style="green")
                    
                else:
                    conversation.update({
                        "status": "failed",
                        "error": api_result["error"],
                        "api_status_code": api_result.get("status_code")
                    })
                    self.results["metadata"]["failed_conversations"] += 1
                    
                    # 顯示錯誤資訊
                    self.console.print(f"❌ [{idx}/{len(dataset)}] {topic[:50]}... - {api_result['error']}", style="red")
                
                self.results["conversations"].append(conversation)
                
                # 更新進度條
                progress.update(task, advance=1)
                
                # 短暫延遲避免 API 過載
                time.sleep(0.1)
    
    def save_results(self, output_path: str = None) -> str:
        """儲存結果到檔案"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"./persona_trait_data/neutral_task/{timestamp}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            
            self.console.print(f"💾 結果已儲存到: {output_path}", style="cyan")
            return output_path
            
        except Exception as e:
            self.console.print(f"❌ 儲存失敗: {e}", style="red")
            return ""
    
    def display_summary(self) -> None:
        """顯示分析摘要"""
        meta = self.results["metadata"]
        
        # 建立摘要表格
        table = Table(title="對話分析摘要")
        table.add_column("項目", justify="left", style="cyan")
        table.add_column("數值", justify="right", style="green")
        
        table.add_row("總對話數", str(meta["total_conversations"]))
        table.add_row("成功對話數", str(meta["successful_conversations"]))
        table.add_row("失敗對話數", str(meta["failed_conversations"]))
        
        if meta["total_conversations"] > 0:
            success_rate = (meta["successful_conversations"] / meta["total_conversations"]) * 100
            table.add_row("成功率", f"{success_rate:.1f}%")
        
        table.add_row("分析時間", meta["timestamp"])
        
        self.console.print(table)
        
        # 顯示失敗的對話（如果有的話）
        failed_conversations = [c for c in self.results["conversations"] if c["status"] == "failed"]
        if failed_conversations:
            self.console.print("\n❌ 失敗的對話:", style="red bold")
            for conv in failed_conversations:
                self.console.print(f"  - {conv['topic']}: {conv.get('error', 'Unknown error')}")
    
    def run_analysis(self, output_path: str = None) -> str:
        """執行完整分析流程"""
        self.console.print("🚀 開始對話分析...", style="bold blue")
        
        # 載入資料集
        dataset = self.load_dataset()
        if not dataset:
            self.console.print("❌ 無法載入資料集，分析終止", style="red")
            return ""
        
        # 處理對話
        self.process_conversations(dataset)
        
        # 顯示摘要
        self.display_summary()
        
        # 儲存結果
        output_file = self.save_results(output_path)
        
        self.console.print("✨ 分析完成！", style="bold green")
        return output_file


def main():
    """主函式"""
    # 建立分析器
    analyzer = ConversationAnalyzer()
    
    # 執行分析
    result_file = analyzer.run_analysis()
    
    if result_file:
        print(f"\n📄 完整結果已儲存在: {result_file}")


if __name__ == "__main__":
    main()