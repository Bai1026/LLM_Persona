import json
import os
import re
import pandas as pd
from pathlib import Path
import argparse

def parse_filename(filename):
    """
    從檔名中提取 layer 和 coefficient 參數
    格式範例: AUT_persona_api_1122-0441_10_l12_c1_simple_eval_results.json
    回傳: (layer, coef) 例如 (12, 1.0)
    """
    # 提取 layer (l後面的數字)
    layer_match = re.search(r'_l(\d+)', filename)
    # 提取 coefficient (c後面的數字，可能是小數)
    coef_match = re.search(r'_c([\d.]+)', filename)
    
    if layer_match and coef_match:
        layer = int(layer_match.group(1))
        coef = float(coef_match.group(1))
        return layer, coef
    return None, None

def parse_eval_results(folder_path):
    """
    解析資料夾中所有包含 simple_eval 的 JSON 檔案
    """
    results = []
    
    # 取得資料夾中所有符合條件的檔案
    if not os.path.exists(folder_path):
        print(f"資料夾不存在: {folder_path}")
        return None
    
    json_files = []
    for file in os.listdir(folder_path):
        if 'simple_eval' in file and file.endswith('.json'):
            json_files.append(file)
    
    if not json_files:
        print(f"在 {folder_path} 中找不到包含 'simple_eval' 的 JSON 檔案")
        return None
    
    print(f"找到 {len(json_files)} 個評估結果檔案")
    
    # 解析每個檔案
    for filename in json_files:
        filepath = os.path.join(folder_path, filename)
        
        # 從檔名提取參數
        layer, coef = parse_filename(filename)
        
        if layer is None or coef is None:
            print(f"警告: 無法從檔名提取參數: {filename}")
            continue
        
        # 讀取 JSON 資料
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            summary = data.get('summary', {})
            
            # 提取分數
            ori_mean = float(summary.get('average_originality', 0))
            ela_mean = float(summary.get('average_elaboration', 0))
            
            # 提取標準差（只要數字部分）
            ori_std_text = summary.get('originality_std', '0')
            ela_std_text = summary.get('elaboration_std', '0')
            
            # 從字串中提取數字
            ori_std_match = re.search(r'[\d.]+', ori_std_text)
            ela_std_match = re.search(r'[\d.]+', ela_std_text)
            
            ori_std = float(ori_std_match.group()) if ori_std_match else 0.0
            ela_std = float(ela_std_match.group()) if ela_std_match else 0.0
            
            results.append({
                'filename': filename,
                'layer': layer,
                'coef': coef,
                'ori_mean': ori_mean,
                'ori_std': ori_std,
                'ela_mean': ela_mean,
                'ela_std': ela_std
            })
            
            print(f"  ✓ {filename}: layer={layer}, coef={coef}, ori={ori_mean:.3f}±{ori_std:.3f}, ela={ela_mean:.3f}±{ela_std:.3f}")
            
        except Exception as e:
            print(f"錯誤: 無法解析檔案 {filename}: {e}")
            continue
    
    return results

def create_csv_tables(results, output_folder):
    """
    建立 4 個 CSV 表格，x 軸是 layer，y 軸是 coefficient
    """
    if not results:
        print("沒有資料可以建立表格")
        return
    
    # 建立 DataFrame
    df = pd.DataFrame(results)
    
    # 取得所有唯一的 layer 和 coef 值並排序
    layers = sorted(df['layer'].unique())
    coefs = sorted(df['coef'].unique())
    
    print(f"\nLayers: {layers}")
    print(f"Coefficients: {coefs}")
    
    # 建立 4 個指標的表格
    metrics = ['ori_mean', 'ori_std', 'ela_mean', 'ela_std']
    metric_names = {
        'ori_mean': 'Originality Mean',
        'ori_std': 'Originality Std',
        'ela_mean': 'Elaboration Mean',
        'ela_std': 'Elaboration Std'
    }
    
    os.makedirs(output_folder, exist_ok=True)
    
    for metric in metrics:
        # 建立 pivot table: rows=coef, columns=layer
        pivot_table = df.pivot_table(
            values=metric,
            index='coef',
            columns='layer',
            aggfunc='first'  # 如果有重複，取第一個
        )
        
        # 排序
        pivot_table = pivot_table.sort_index(ascending=True)
        pivot_table = pivot_table[sorted(pivot_table.columns)]
        
        # 儲存 CSV
        output_file = os.path.join(output_folder, f'{metric}.csv')
        pivot_table.to_csv(output_file, float_format='%.3f')
        
        print(f"\n✓ 已建立 {metric_names[metric]} 表格: {output_file}")
        print(pivot_table.to_string(float_format=lambda x: f'{x:.3f}'))

def main():
    parser = argparse.ArgumentParser(description='解析評估結果並建立 CSV 表格')
    # parser.add_argument('-i', '--input', type=str, required=True,
    #                     help='輸入資料夾路徑 (包含 simple_eval JSON 檔案)')
    # parser.add_argument('-o', '--output', type=str, default=None,
    #                     help='輸出資料夾路徑 (預設為輸入資料夾)')
    
    # args = parser.parse_args()
    
    # input_folder = args.input
    # output_folder = args.output if args.output else input_folder
    
    input_folder = './Main_Results/Rebuttal/Gemma_Rebuttal/more/'
    output_folder = './Main_Results/Rebuttal/Gemma_Rebuttal_Output_more'

    print(f"解析資料夾: {input_folder}")
    print(f"輸出資料夾: {output_folder}")
    print("="*60)
    
    # 解析結果
    results = parse_eval_results(input_folder)
    
    if results:
        # 建立 CSV 表格
        create_csv_tables(results, output_folder)
        print(f"\n完成！共處理 {len(results)} 個檔案")
    else:
        print("沒有找到可用的評估結果")

if __name__ == "__main__":
    main()
