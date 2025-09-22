#!/usr/bin/env python3
"""
批次 Token 計算器 - 處理多個檔案
"""

import json
import argparse
import os
import pandas as pd
from pathlib import Path
from token_counter import TokenCounter
from typing import List, Dict

def find_json_files(directory: str, pattern: str = "*.json") -> List[str]:
    """尋找指定目錄下的 JSON 檔案"""
    path = Path(directory)
    if not path.exists():
        print(f"❌ 目錄不存在: {directory}")
        return []
    
    json_files = list(path.glob(pattern))
    return [str(f) for f in json_files]

def process_multiple_files(file_paths: List[str], model: str = "gpt-4") -> List[Dict]:
    """處理多個檔案"""
    counter = TokenCounter(model)
    results = []
    
    print(f"📁 找到 {len(file_paths)} 個檔案")
    print("="*60)
    
    for i, file_path in enumerate(file_paths, 1):
        print(f"📄 處理檔案 {i}/{len(file_paths)}: {Path(file_path).name}")
        
        try:
            stats = counter.analyze_file(file_path)
            
            result = {
                "檔案名稱": Path(file_path).name,
                "檔案路徑": file_path,
                "Input_Tokens": stats.input_tokens,
                "Output_Tokens": stats.output_tokens,
                "Total_Tokens": stats.total_tokens,
                "Input_比例(%)": round((stats.input_tokens / stats.total_tokens * 100) if stats.total_tokens > 0 else 0, 2),
                "Output_比例(%)": round((stats.output_tokens / stats.total_tokens * 100) if stats.total_tokens > 0 else 0, 2)
            }
            
            results.append(result)
            print(f"   ✅ Input: {stats.input_tokens:,}, Output: {stats.output_tokens:,}, Total: {stats.total_tokens:,}")
            
        except Exception as e:
            print(f"   ❌ 處理失敗: {e}")
            result = {
                "檔案名稱": Path(file_path).name,
                "檔案路徑": file_path,
                "Input_Tokens": 0,
                "Output_Tokens": 0,
                "Total_Tokens": 0,
                "Input_比例(%)": 0,
                "Output_比例(%)": 0,
                "錯誤": str(e)
            }
            results.append(result)
        
        print()
    
    return results

def print_summary(results: List[Dict]):
    """印出摘要統計"""
    if not results:
        print("❌ 沒有結果可以顯示")
        return
    
    total_input = sum(r["Input_Tokens"] for r in results)
    total_output = sum(r["Output_Tokens"] for r in results)
    total_all = sum(r["Total_Tokens"] for r in results)
    
    print("="*80)
    print("📈 批次處理摘要統計")
    print("="*80)
    print(f"📁 處理檔案數量:     {len(results)}")
    print(f"📥 總 Input Tokens:   {total_input:,}")
    print(f"📤 總 Output Tokens:  {total_output:,}")
    print(f"📊 總計 Tokens:       {total_all:,}")
    
    if total_all > 0:
        print(f"📊 平均 Input 比例:   {(total_input / total_all * 100):.1f}%")
        print(f"📊 平均 Output 比例:  {(total_output / total_all * 100):.1f}%")
    
    print("="*80)

def save_to_csv(results: List[Dict], output_file: str):
    """儲存結果到 CSV"""
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"💾 結果已儲存到 CSV: {output_file}")
    except Exception as e:
        print(f"❌ 儲存 CSV 失敗: {e}")

def save_to_json(results: List[Dict], output_file: str):
    """儲存結果到 JSON"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 結果已儲存到 JSON: {output_file}")
    except Exception as e:
        print(f"❌ 儲存 JSON 失敗: {e}")

def main():
    parser = argparse.ArgumentParser(description="批次計算多個對話檔案的 tokens")
    parser.add_argument("directory", help="包含 JSON 檔案的目錄")
    parser.add_argument("--pattern", default="*.json", help="檔案搜尋模式 (預設: *.json)")
    parser.add_argument("--model", default="gpt-4", help="使用的模型名稱 (預設: gpt-4)")
    parser.add_argument("--output-csv", help="輸出 CSV 檔案路徑")
    parser.add_argument("--output-json", help="輸出 JSON 檔案路徑")
    parser.add_argument("--recursive", "-r", action="store_true", help="遞迴搜尋子目錄")
    
    args = parser.parse_args()
    
    # 尋找檔案
    if args.recursive:
        pattern = f"**/{args.pattern}"
    else:
        pattern = args.pattern
    
    file_paths = find_json_files(args.directory, pattern)
    
    if not file_paths:
        print(f"❌ 在目錄 {args.directory} 中找不到符合 {pattern} 的檔案")
        return
    
    # 處理檔案
    results = process_multiple_files(file_paths, args.model)
    
    # 印出摘要
    print_summary(results)
    
    # 儲存結果
    if args.output_csv:
        save_to_csv(results, args.output_csv)
    
    if args.output_json:
        save_to_json(results, args.output_json)

if __name__ == "__main__":
    main()
