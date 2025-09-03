import subprocess
import sys
import os
import glob
from pathlib import Path
import argparse
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def auto_eval_persona():
    """自動執行 Persona 評估流程"""
    parser = argparse.ArgumentParser(description="自動執行 Persona API 評估")
    parser.add_argument("-d", "--dataset", required=True, help="資料集檔案路徑")
    parser.add_argument("-t", "--task", choices=["AUT", "Scientific", "Instances", "Similarities"], 
                       required=True, help="任務類型")
    parser.add_argument("-p", "--prompt", type=int, default=1, help="提示詞編號")
    parser.add_argument("-v", "--gpt_version", default="4", choices=["3", "4"], help="GPT 版本")
    parser.add_argument("--no_eval", action="store_true", help="只產生結果，不進行評估")
    parser.add_argument("--baseline", action="store_true", help="使用 Pure OpenAI API 模式作為 baseline")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI 模型名稱 (當使用 baseline 模式時)")
    
    args = parser.parse_args()
    
    if args.baseline:
        print(f"🚀 開始 Pure OpenAI API Baseline 評估 - 任務: {args.task}")
        print(f"📋 使用模型: {args.model}")
    else:
        print(f"🚀 開始自動評估流程 - 任務: {args.task}")
    
    # 檢查 OpenAI API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 請設定 OPENAI_API_KEY 環境變數")
        return
    
    # 1. 產生結果檔案
    print("📊 步驟 1: 產生結果檔案...")
    
    if args.baseline:
        # 使用 Pure OpenAI API 模式
        generate_cmd = [
            sys.executable, "openai_baseline.py",
            "-d", args.dataset,
            "-t", args.task,
            "-p", str(args.prompt),
            "--model", args.model
        ]
    else:
        # 使用 Persona API 模式
        generate_cmd = [
            sys.executable, "persona_conversation.py",
            "-d", args.dataset,
            "-t", args.task,
            "-p", str(args.prompt)
        ]
    
    # 如果不是 no_eval 模式，則啟用評估
    if not args.no_eval:
        generate_cmd.append("-e")  # 啟用評估模式
    
    print(f"🔧 執行命令: {' '.join(generate_cmd)}")
    
    # 執行命令並即時顯示輸出
    try:
        result = subprocess.run(generate_cmd, text=True, timeout=300)  # 5分鐘超時
        
        if result.returncode == 0:
            print("✅ 流程完成!")
        else:
            print(f"❌ 執行失敗，返回代碼: {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("❌ 執行超時（5分鐘），可能是 API 連線問題")
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")

if __name__ == "__main__":
    auto_eval_persona()