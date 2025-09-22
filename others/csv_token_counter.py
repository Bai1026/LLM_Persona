#!/usr/bin/env python3
"""
CSV Token Counter
計算 CSV 檔案中 prompt 和 answer 欄位的 tokens
prompt = input tokens, answer = output tokens
"""

import pandas as pd
import tiktoken
import argparse
import sys
import os
from typing import Dict, Any

class CSVTokenCounter:
    def __init__(self, model: str = "gpt-4"):
        """初始化 token 計算器"""
        try:
            self.encoding = tiktoken.encoding_for_model(model)
            self.model = model
        except KeyError:
            print(f"⚠️  模型 {model} 不支援，使用 cl100k_base encoding")
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self.model = "cl100k_base"
    
    def count_tokens(self, text: str) -> int:
        """計算文字的 token 數量"""
        if not text or pd.isna(text):
            return 0
        return len(self.encoding.encode(str(text)))
    
    def process_csv(self, csv_path: str) -> Dict[str, Any]:
        """處理 CSV 檔案並計算 tokens"""
        try:
            # 讀取 CSV
            df = pd.read_csv(csv_path)
            
            # 檢查必要欄位
            if 'prompt' not in df.columns or 'answer' not in df.columns:
                raise ValueError(f"CSV 檔案必須包含 'prompt' 和 'answer' 欄位。可用欄位: {df.columns.tolist()}")
            
            # 計算每行的 tokens
            df['input_tokens'] = df['prompt'].apply(self.count_tokens)
            df['output_tokens'] = df['answer'].apply(self.count_tokens)
            df['total_tokens'] = df['input_tokens'] + df['output_tokens']
            
            # 統計結果
            stats = {
                'file': os.path.basename(csv_path),
                'model': self.model,
                'total_rows': len(df),
                'valid_rows': len(df.dropna(subset=['prompt', 'answer'])),
                'total_input_tokens': df['input_tokens'].sum(),
                'total_output_tokens': df['output_tokens'].sum(),
                'total_tokens': df['total_tokens'].sum(),
                'avg_input_tokens': df['input_tokens'].mean(),
                'avg_output_tokens': df['output_tokens'].mean(),
                'avg_total_tokens': df['total_tokens'].mean(),
                'input_ratio': (df['input_tokens'].sum() / df['total_tokens'].sum() * 100) if df['total_tokens'].sum() > 0 else 0,
                'output_ratio': (df['output_tokens'].sum() / df['total_tokens'].sum() * 100) if df['total_tokens'].sum() > 0 else 0,
                'detailed_data': df[['question', 'prompt', 'answer', 'input_tokens', 'output_tokens', 'total_tokens']].to_dict('records') if 'question' in df.columns else df[['prompt', 'answer', 'input_tokens', 'output_tokens', 'total_tokens']].to_dict('records')
            }
            
            return stats
            
        except Exception as e:
            raise Exception(f"處理 CSV 檔案時出錯: {e}")
    
    def print_stats(self, stats: Dict[str, Any], show_details: bool = False):
        """印出統計結果"""
        print(f"\n📊 檔案 {stats['file']} 的 Token 統計結果")
        print("=" * 60)
        print(f"🔧 使用模型: {stats['model']}")
        print(f"📄 總行數: {stats['total_rows']:,}")
        print(f"✅ 有效行數: {stats['valid_rows']:,}")
        print("-" * 60)
        print(f"📥 Input Tokens (prompt):  {stats['total_input_tokens']:,}")
        print(f"📤 Output Tokens (answer): {stats['total_output_tokens']:,}")
        print(f"📊 Total Tokens:           {stats['total_tokens']:,}")
        print("-" * 60)
        print(f"📊 Input 比例:   {stats['input_ratio']:.1f}%")
        print(f"📊 Output 比例:  {stats['output_ratio']:.1f}%")
        print("-" * 60)
        print(f"📊 平均 Input Tokens:  {stats['avg_input_tokens']:.1f}")
        print(f"📊 平均 Output Tokens: {stats['avg_output_tokens']:.1f}")
        print(f"📊 平均 Total Tokens:  {stats['avg_total_tokens']:.1f}")
        print("=" * 60)
        
        if show_details:
            print("\n📋 詳細資料:")
            for i, row in enumerate(stats['detailed_data'][:10], 1):  # 只顯示前10筆
                print(f"\n--- 第 {i} 筆 ---")
                if 'question' in row:
                    print(f"Question: {row['question'][:100]}...")
                print(f"Input tokens: {row['input_tokens']}")
                print(f"Output tokens: {row['output_tokens']}")
                print(f"Total tokens: {row['total_tokens']}")
            
            if len(stats['detailed_data']) > 10:
                print(f"\n... 還有 {len(stats['detailed_data']) - 10} 筆資料")
    
    def save_results(self, stats: Dict[str, Any], output_path: str, format: str = "csv"):
        """儲存結果到檔案"""
        try:
            if format.lower() == "csv":
                # 建立結果 DataFrame
                df_results = pd.DataFrame(stats['detailed_data'])
                df_results.to_csv(output_path, index=False, encoding='utf-8')
                print(f"✅ 詳細結果已儲存到: {output_path}")
                
            elif format.lower() == "json":
                import json
                # 移除詳細資料以減少檔案大小
                summary_stats = {k: v for k, v in stats.items() if k != 'detailed_data'}
                summary_stats['sample_data'] = stats['detailed_data'][:5]  # 只保留前5筆範例
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(summary_stats, f, ensure_ascii=False, indent=2)
                print(f"✅ 統計結果已儲存到: {output_path}")
                
        except Exception as e:
            print(f"❌ 儲存結果時出錯: {e}")

def main():
    parser = argparse.ArgumentParser(description="計算 CSV 檔案中 prompt 和 answer 的 tokens")
    parser.add_argument("csv_file", help="要處理的 CSV 檔案路徑")
    parser.add_argument("--model", default="gpt-4o", help="使用的模型 (預設: gpt-4)")
    parser.add_argument("--output", help="輸出檔案路徑")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="輸出格式 (預設: csv)")
    parser.add_argument("--details", action="store_true", help="顯示詳細資料")
    
    args = parser.parse_args()
    
    # 檢查檔案是否存在
    if not os.path.exists(args.csv_file):
        print(f"❌ 檔案不存在: {args.csv_file}")
        sys.exit(1)
    
    # 建立計算器
    counter = CSVTokenCounter(args.model)
    
    try:
        # 處理 CSV 檔案
        print(f"🔄 正在處理檔案: {args.csv_file}")
        stats = counter.process_csv(args.csv_file)
        
        # 顯示結果
        counter.print_stats(stats, show_details=args.details)
        
        # 儲存結果
        if args.output:
            counter.save_results(stats, args.output, args.format)
        
    except Exception as e:
        print(f"❌ 處理過程中出錯: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
