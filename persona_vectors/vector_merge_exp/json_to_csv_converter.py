#!/usr/bin/env python3
"""
將 JSON 檔案轉換成 CSV 格式
輸出欄位: topic, question, response, cre_response_proj, env_response_proj
"""

import json
import pandas as pd
import argparse
from pathlib import Path

def convert_json_to_csv(json_file_path, output_csv_path=None):
    """將 JSON 檔案轉換成 CSV"""
    
    # 讀取 JSON 檔案
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取 conversations 資料
    conversations = data.get('conversations', [])
    
    if not conversations:
        print("❌ 沒有找到 conversations 資料")
        return None
    
    # 準備 CSV 資料
    csv_data = []
    
    for conv in conversations:
        # 找出 creative 和 environmental 的投影值
        cre_proj = None
        env_proj = None
        
        # 搜尋包含 creative 和 environmentalist 的欄位
        for key, value in conv.items():
            if 'creative' in key.lower() and 'proj' in key.lower():
                cre_proj = value
            elif 'environmental' in key.lower() and 'proj' in key.lower():
                env_proj = value
        
        # 建立 CSV 行
        row = {
            'topic': conv.get('topic', ''),
            'question': conv.get('question', ''),
            'response': conv.get('response', ''),
            'cre_response_proj': cre_proj,
            'env_response_proj': env_proj
        }
        
        csv_data.append(row)
    
    # 轉換為 DataFrame
    df = pd.DataFrame(csv_data)
    
    # 設定輸出檔案路徑
    if output_csv_path is None:
        input_path = Path(json_file_path)
        output_csv_path = input_path.parent / f"{input_path.stem}.csv"
    
    # 儲存為 CSV
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"✅ CSV 檔案已儲存至: {output_csv_path}")
    print(f"📊 共轉換 {len(csv_data)} 筆對話記錄")
    
    # 顯示前幾行預覽
    print(f"\n📋 資料預覽:")
    print(df.head().to_string())
    
    return df

def main():
    parser = argparse.ArgumentParser(description="將 JSON 檔案轉換成 CSV 格式")
    parser.add_argument("json_file", 
                       help="要轉換的 JSON 檔案路徑")
    parser.add_argument("--output", "-o",
                       help="輸出 CSV 檔案路徑（可選，預設為同檔名的 .csv）")
    
    args = parser.parse_args()
    
    # 檢查輸入檔案是否存在
    if not Path(args.json_file).exists():
        print(f"❌ 檔案不存在: {args.json_file}")
        return
    
    print(f"🚀 開始轉換 JSON 檔案: {args.json_file}")
    
    # 執行轉換
    convert_json_to_csv(args.json_file, args.output)

if __name__ == "__main__":
    main()