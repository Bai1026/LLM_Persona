#!/usr/bin/env python3
"""
Token 計算器 - 計算 JSON 對話檔案中的 input 和 output tokens
支援兩種格式：
1. 單純對話格式 (user = input, assistant = output)
2. Discussion 格式 (複雜的多代理討論格式)
"""

import json
import argparse
import tiktoken
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class TokenStats:
    """Token 統計結果"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    def add(self, other: 'TokenStats'):
        """合併統計"""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens

class TokenCounter:
    """Token 計算器"""
    
    def __init__(self, model_name: str = "gpt-4"):
        """初始化 tokenizer"""
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # 如果模型不存在，使用 cl100k_base (GPT-4 的編碼)
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
    def count_tokens(self, text: str) -> int:
        """計算文字的 token 數量"""
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def count_message_tokens(self, messages: List[Dict[str, str]]) -> TokenStats:
        """計算一組訊息的 tokens"""
        stats = TokenStats()
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            token_count = self.count_tokens(content)
            
            if role == "user":
                stats.input_tokens += token_count
            elif role == "assistant":
                stats.output_tokens += token_count
            
        stats.total_tokens = stats.input_tokens + stats.output_tokens
        return stats
    
    def analyze_simple_format(self, data: List[Dict]) -> TokenStats:
        """分析簡單對話格式"""
        total_stats = TokenStats()
        
        print("📊 分析簡單對話格式...")
        
        for i, conversation in enumerate(data):
            if "PersonaAPI" in conversation:
                messages = conversation["PersonaAPI"]
                stats = self.count_message_tokens(messages)
                
                print(f"  對話 {i+1}: Input={stats.input_tokens}, Output={stats.output_tokens}")
                total_stats.add(stats)
        
        return total_stats
    
    def analyze_discussion_format(self, data: Dict) -> TokenStats:
        """分析 discussion 格式"""
        total_stats = TokenStats()
        
        print("📊 分析 Discussion 格式...")
        
        for question, agents in data.items():
            print(f"\n❓ 問題: {question[:100]}...")
            
            for agent_name, conversations in agents.items():
                print(f"  🤖 代理: {agent_name}")
                
                stats = self.count_message_tokens(conversations)
                print(f"     Input={stats.input_tokens}, Output={stats.output_tokens}")
                
                total_stats.add(stats)
        
        return total_stats
    
    def detect_format(self, data: Any) -> str:
        """檢測檔案格式"""
        if isinstance(data, list):
            return "simple"
        elif isinstance(data, dict):
            # 檢查是否為 discussion 格式
            for key, value in data.items():
                if isinstance(value, dict):
                    for agent_name, conversations in value.items():
                        if isinstance(conversations, list) and conversations:
                            if "role" in conversations[0]:
                                return "discussion"
            return "unknown"
        else:
            return "unknown"
    
    def analyze_file(self, file_path: str) -> TokenStats:
        """分析檔案並返回統計結果"""
        print(f"📁 讀取檔案: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ 讀取檔案失敗: {e}")
            return TokenStats()
        
        format_type = self.detect_format(data)
        print(f"🔍 檢測到格式: {format_type}")
        
        if format_type == "simple":
            return self.analyze_simple_format(data)
        elif format_type == "discussion":
            return self.analyze_discussion_format(data)
        else:
            print("❌ 未知的檔案格式")
            return TokenStats()

def print_results(stats: TokenStats, file_name: str):
    """印出結果"""
    print(f"\n" + "="*60)
    print(f"📈 檔案 {file_name} 的 Token 統計結果")
    print("="*60)
    print(f"📥 Input Tokens (user):       {stats.input_tokens:,}")
    print(f"📤 Output Tokens (assistant): {stats.output_tokens:,}")
    print(f"📊 Total Tokens:              {stats.total_tokens:,}")
    print("="*60)
    
    # 計算比例
    if stats.total_tokens > 0:
        input_ratio = (stats.input_tokens / stats.total_tokens) * 100
        output_ratio = (stats.output_tokens / stats.total_tokens) * 100
        print(f"📊 Input 比例:   {input_ratio:.1f}%")
        print(f"📊 Output 比例:  {output_ratio:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="計算對話檔案中的 input/output tokens")
    parser.add_argument("file_path", help="JSON 檔案路徑")
    parser.add_argument("--model", default="gpt-4", help="使用的模型名稱 (預設: gpt-4)")
    parser.add_argument("--output", "-o", help="輸出結果到檔案")
    
    args = parser.parse_args()
    
    # 檢查檔案是否存在
    if not Path(args.file_path).exists():
        print(f"❌ 檔案不存在: {args.file_path}")
        return
    
    # 建立 token 計算器
    counter = TokenCounter(args.model)
    
    # 分析檔案
    stats = counter.analyze_file(args.file_path)
    
    # 印出結果
    file_name = Path(args.file_path).name
    print_results(stats, file_name)
    
    # 輸出到檔案
    if args.output:
        try:
            result = {
                "file": file_name,
                "model": args.model,
                "input_tokens": stats.input_tokens,
                "output_tokens": stats.output_tokens,
                "total_tokens": stats.total_tokens,
                "input_ratio": (stats.input_tokens / stats.total_tokens * 100) if stats.total_tokens > 0 else 0,
                "output_ratio": (stats.output_tokens / stats.total_tokens * 100) if stats.total_tokens > 0 else 0
            }
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"💾 結果已儲存到: {args.output}")
        except Exception as e:
            print(f"❌ 儲存失敗: {e}")

if __name__ == "__main__":
    main()
