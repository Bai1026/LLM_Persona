#!/usr/bin/env python3
"""
整理 Result 資料夾中所有 comprehensive 評估結果的 trait_averages
"""

import json
import os
import pandas as pd
from pathlib import Path
import argparse

def extract_info_from_filename(filename):
    """從檔名中提取資訊"""
    # 移除副檔名
    name = filename.replace('.json', '')
    
    # 只處理 comprehensive 結果，移除相應後綴
    name = name.replace('_evaluation_results_comprehensive', '')
    
    # 提取模型名稱（如果有）
    model = "local"  # 預設
    if name.endswith('_gpt'):
        model = "gpt"
        name = name.replace('_gpt', '')
    elif name.endswith('_gemini'):
        model = "gemini"
        name = name.replace('_gemini', '')
    
    return name, model

def load_all_results(result_dir):
    """載入所有結果檔案"""
    results = []
    result_path = Path(result_dir)
    
    if not result_path.exists():
        print(f"❌ 結果資料夾不存在: {result_dir}")
        return results
    
    json_files = list(result_path.glob("*.json"))
    print(f"📁 找到 {len(json_files)} 個 JSON 檔案")
    
    for file_path in json_files:
        # 只處理 comprehensive 類型的檔案
        if '_evaluation_results_comprehensive' not in file_path.name:
            print(f"⏭️  跳過非 comprehensive 檔案: {file_path.name}")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 檢查是否有 trait_averages
            if 'trait_averages' not in data:
                print(f"⚠️  {file_path.name} 沒有 trait_averages 欄位")
                continue
            
            # 提取檔名資訊
            persona_combo, model = extract_info_from_filename(file_path.name)
            
            # 建立結果記錄
            result = {
                'filename': file_path.name,
                'persona_combination': persona_combo,
                'evaluation_model': model,
                **data['trait_averages']
            }
            
            results.append(result)
            print(f"✅ 載入: {file_path.name} -> {persona_combo} ({model})")
            
        except Exception as e:
            print(f"❌ 載入 {file_path.name} 時發生錯誤: {e}")
    
    return results

def create_summary_table(results):
    """建立摘要表格"""
    if not results:
        print("❌ 沒有可用的結果資料")
        return None, None
    
    # 轉換為 DataFrame
    df = pd.DataFrame(results)
    
    # 重新排列欄位順序
    trait_columns = ['empathetic', 'analytical', 'creative', 'environmental', 'futurist']
    other_columns = ['filename', 'persona_combination', 'evaluation_model']
    
    # 確保所有 trait 欄位都存在
    for trait in trait_columns:
        if trait not in df.columns:
            df[trait] = 0.0
    
    # 重新排序欄位
    df = df[other_columns + trait_columns]
    
    # 建立統計表格
    stats_df = create_prefix_statistics(df, trait_columns)
    
    return df, stats_df

def create_prefix_statistics(df, trait_columns):
    """建立相同前綴的統計表格"""
    # 取得所有唯一的 persona 前綴
    prefixes = set()
    for combo in df['persona_combination'].unique():
        # 前綴是整個 persona_combination，因為它已經是從檔名解析出來的前綴部分
        # （在 extract_info_from_filename 中已經移除了 _evaluation_results_comprehensive）
        prefixes.add(combo)
    
    stats_results = []
    
    for prefix in sorted(prefixes):
        # 找出所有完全匹配此前綴的組合
        prefix_data = df[df['persona_combination'] == prefix]
        
        if len(prefix_data) == 0:
            continue
            
        # 按評估模型分組計算平均
        for model in prefix_data['evaluation_model'].unique():
            model_data = prefix_data[prefix_data['evaluation_model'] == model]
            
            if len(model_data) > 0:
                result = {
                    'prefix': prefix,
                    'evaluation_model': model,
                    'count': len(model_data)
                }
                
                # 計算每個 trait 的平均
                for trait in trait_columns:
                    if trait in model_data.columns:
                        result[f'{trait}_avg'] = model_data[trait].mean()
                    else:
                        result[f'{trait}_avg'] = 0.0
                
                stats_results.append(result)
        
        # 計算所有模型的總平均
        if len(prefix_data) > 0:
            result = {
                'prefix': prefix,
                'evaluation_model': 'ALL_AVG',
                'count': len(prefix_data)
            }
            
            for trait in trait_columns:
                if trait in prefix_data.columns:
                    result[f'{trait}_avg'] = prefix_data[trait].mean()
                else:
                    result[f'{trait}_avg'] = 0.0
            
            stats_results.append(result)
    
    # 轉換為 DataFrame
    if stats_results:
        stats_df = pd.DataFrame(stats_results)
        # 重新排序欄位
        base_columns = ['prefix', 'evaluation_model', 'count']
        avg_columns = [f'{trait}_avg' for trait in trait_columns]
        stats_df = stats_df[base_columns + avg_columns]
        return stats_df
    else:
        return pd.DataFrame()

def save_results(df, stats_df, output_dir):
    """儲存結果"""
    if df is None:
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 儲存原始結果
    csv_file = output_path / "comprehensive_trait_averages.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"💾 CSV 檔案已儲存至: {csv_file}")
    
    # 儲存統計結果
    if stats_df is not None and not stats_df.empty:
        stats_csv_file = output_path / "prefix_statistics.csv"
        stats_df.to_csv(stats_csv_file, index=False, encoding='utf-8')
        print(f"💾 統計 CSV 檔案已儲存至: {stats_csv_file}")
    
    # 儲存為 Excel（如果有 pandas 和 openpyxl）
    try:
        excel_file = output_path / "comprehensive_trait_averages.xlsx"
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='原始資料', index=False)
            if stats_df is not None and not stats_df.empty:
                stats_df.to_excel(writer, sheet_name='前綴統計', index=False)
        print(f"💾 Excel 檔案已儲存至: {excel_file}")
    except ImportError:
        print("⚠️  未安裝 openpyxl，跳過 Excel 輸出")
    
    # 儲存為 JSON
    json_file = output_path / "comprehensive_trait_averages.json"
    output_data = {
        'raw_data': df.to_dict('records'),
        'statistics': stats_df.to_dict('records') if stats_df is not None and not stats_df.empty else []
    }
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"💾 JSON 檔案已儲存至: {json_file}")

def print_summary_stats(df, stats_df):
    """列印摘要統計"""
    if df is None:
        return
    
    print("\n" + "="*80)
    print("📊 摘要統計")
    print("="*80)
    
    # 按評估模型分組
    print("\n🤖 按評估模型分組:")
    model_counts = df['evaluation_model'].value_counts()
    for model, count in model_counts.items():
        print(f"  {model}: {count} 個結果")
    
    # Trait 平均分數
    trait_columns = ['empathetic', 'analytical', 'creative', 'environmental', 'futurist']
    print(f"\n🎯 各 Trait 的整體平均分數:")
    for trait in trait_columns:
        if trait in df.columns:
            avg_score = df[trait].mean()
            print(f"  {trait}: {avg_score:.2f}")
    
    # 最高分數的組合
    print(f"\n🏆 各 Trait 最高分數的組合:")
    for trait in trait_columns:
        if trait in df.columns:
            max_idx = df[trait].idxmax()
            max_row = df.iloc[max_idx]
            print(f"  {trait}: {max_row[trait]:.2f} ({max_row['persona_combination']} - {max_row['evaluation_model']})")
    
    # 顯示前綴統計
    if stats_df is not None and not stats_df.empty:
        print(f"\n📋 前綴統計 (相同前綴的平均分數):")
        print("-" * 80)
        
        # 按前綴分組顯示
        for prefix in stats_df['prefix'].unique():
            prefix_data = stats_df[stats_df['prefix'] == prefix]
            print(f"\n🎭 {prefix}:")
            
            for _, row in prefix_data.iterrows():
                model = row['evaluation_model']
                count = row['count']
                trait_scores = []
                
                for trait in trait_columns:
                    avg_col = f'{trait}_avg'
                    if avg_col in row:
                        trait_scores.append(f"{trait}:{row[avg_col]:.2f}")
                
                print(f"  {model:10s} (n={count:2d}): " + " | ".join(trait_scores))

def create_comparison_table(df):
    """建立不同評估模型的比較表格"""
    if df is None:
        return None
    
    # 按 persona_combination 和 evaluation_model 進行透視
    trait_columns = ['empathetic', 'analytical', 'creative', 'environmental', 'futurist']
    
    print(f"\n📋 不同評估模型的比較:")
    print("-" * 120)
    
    # 取得所有唯一的 persona 組合
    unique_combos = df['persona_combination'].unique()
    
    for combo in sorted(unique_combos):
        combo_data = df[df['persona_combination'] == combo]
        
        if len(combo_data) > 1:  # 只顯示有多個模型結果的組合
            print(f"\n🎭 {combo}:")
            for _, row in combo_data.iterrows():
                model = row['evaluation_model']
                scores = [f"{row[trait]:.1f}" for trait in trait_columns if trait in row]
                print(f"  {model:8s}: " + " | ".join(f"{trait}:{score:>6s}" for trait, score in zip(trait_columns, scores)))

def main():
    parser = argparse.ArgumentParser(description="整理所有 comprehensive 評估結果的 trait_averages")
    parser.add_argument("--result_dir", 
                       default="persona_trait_data/neutral_task/Result",
                       help="結果資料夾路徑")
    parser.add_argument("--output_dir",
                       default="analysis_output",
                       help="輸出資料夾路徑")
    parser.add_argument("--show_comparison", 
                       action="store_true",
                       help="顯示不同評估模型的詳細比較")
    
    args = parser.parse_args()
    
    print("🚀 開始分析所有 comprehensive 評估結果...")
    
    # 載入所有結果
    results = load_all_results(args.result_dir)
    
    if not results:
        print("❌ 沒有找到任何有效的結果檔案")
        return
    
    # 建立摘要表格
    df, stats_df = create_summary_table(results)
    
    # 儲存結果
    save_results(df, stats_df, args.output_dir)
    
    # 列印摘要統計
    print_summary_stats(df, stats_df)
    
    # 顯示比較表格
    if args.show_comparison:
        create_comparison_table(df)
    
    print(f"\n✅ 分析完成！共處理 {len(results)} 個結果檔案")
    print(f"📁 輸出檔案儲存在: {args.output_dir}")

if __name__ == "__main__":
    main()