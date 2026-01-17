#!/usr/bin/env python3
"""
直接將評估結果 JSON 檔案加入到 leaderboard CSV 中
使用方式: python add_to_leaderboard.py <json_file_path> <csv_output_path>
"""

import json
import sys
from pathlib import Path
from automation_csv import calculate_mean_std, write_results_to_csv

def process_json_to_leaderboard(json_file_path, csv_output_path):
    """
    直接處理 JSON 檔案並加入到 leaderboard
    """
    
    # 讀取 JSON 檔案
    with open(json_file_path, 'r', encoding='utf-8') as f:
        total_results = json.load(f)
    
    # 計算平均值和標準差
    mean_std_results = calculate_mean_std(total_results)
    
    # 從檔案路徑提取檔案名稱
    json_file_name = Path(json_file_path).name
    
    # 寫入 CSV
    write_results_to_csv(json_file_name, mean_std_results, csv_output_path, version=4)
    
    print(f"✅ 成功將 {json_file_name} 的結果加入到 {csv_output_path}")
    print(f"📊 結果摘要:")
    print(f"   - 流暢度 (Fluency): {mean_std_results['mean_fluency']:.3f} ± {mean_std_results['std_fluency']:.3f}")
    print(f"   - 彈性 (Flexibility): {mean_std_results['mean_flexibility']:.3f} ± {mean_std_results['std_flexibility']:.3f}")
    print(f"   - 原創性 (Originality): {mean_std_results['mean_originality']:.3f} ± {mean_std_results['std_originality']:.3f}")
    print(f"   - 精細度 (Elaboration): {mean_std_results['mean_elaboration']:.3f} ± {mean_std_results['std_elaboration']:.3f}")

def main():
    # TODO: change the json file path here.
    json_file_path = "/workspace/LLM_Persona/LLM-Discussion/Results/AUT/Eval_Result/persona_agent/evaluation_AUT_persona_api_0908-0548_10_sampling_4.json"
    csv_output_path = '../Results/LeaderBoard/LeaderBoard-AUT.csv'
    
    # 檢查檔案是否存在
    if not Path(json_file_path).exists():
        print(f"❌ 錯誤: 找不到檔案 {json_file_path}")
        sys.exit(1)
    
    try:
        process_json_to_leaderboard(json_file_path, csv_output_path)
    except Exception as e:
        print(f"❌ 處理過程中發生錯誤: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
