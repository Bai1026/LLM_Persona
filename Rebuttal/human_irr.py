import numpy as np
import pandas as pd
from rich import print
import krippendorff

# # --- 步驟 1: 填入您的原始數據 ---
# # ⚠️ 請將每行替換為您每個 Rater 的 72 個原始分數
# RATINGS = np.array([
#     # Rater 1: 72 scores (Ori_SA, Ela_SA, Ori_LLMD, Ela_LLMD, Ori_BILLY, Ela_BILLY)
#     [0.0, 0.0, 0.0, ..., 0.0], # <-- 請替換為 Rater 1 的 72 個分數
#     # Rater 2: 72 scores
#     [0.0, 0.0, 0.0, ..., 0.0], # <-- 請替換為 Rater 2 的 72 個分數
#     # Rater 3: 72 scores
#     [0.0, 0.0, 0.0, ..., 0.0], # <-- 請替換為 Rater 3 的 72 個分數
#     # Rater 4: 72 scores
#     [0.0, 0.0, 0.0, ..., 0.0], # <-- 請替換為 Rater 4 的 72 個分數
#     # Rater 5: 72 scores
#     [0.0, 0.0, 0.0, ..., 0.0]  # <-- 請替換為 Rater 5 的 72 個分數
# ]) 

# read data from csv
data_df = pd.read_csv('human_scores.csv', header=None)

# 只取前 72 個欄位的評分資料（從第 2 欄開始，跳過第一欄的 "People" 標籤）
# 並且跳過第一列的標題和最後幾列的平均值
RATINGS = data_df.iloc[1:, 1:73].to_numpy()

# 轉換為 float，將無法轉換的值設為 NaN
RATINGS = pd.DataFrame(RATINGS).apply(pd.to_numeric, errors='coerce').to_numpy()

print(f"原始資料維度: {RATINGS.shape}")
print(f"前幾列資料:\n{RATINGS[:3, :10]}")  # 顯示前 3 個評分者的前 10 個分數

# 確認數據維度
print(f"數據維度: {RATINGS.shape}")
print(f"期望維度: (評分者數量, 72)")

# --- 步驟 2: 定義方法和切片邏輯 ---
METHODS = ['SA', 'LLMD', 'BILLY']
METRICS = ['Originality', 'Elaboration']
ITEM_COUNT = 12
RESULTS = {}

print("\n" + "="*60)
print("開始計算 Krippendorff's Alpha")
print("="*60)

# --- 步驟 3: 循環計算 IRR (Krippendorff's Alpha) ---

for i, method in enumerate(METHODS):
    # 計算起始索引 (Start Index)
    # i=0 (SA): Ori starts at 0, Ela starts at 12
    # i=1 (LLMD): Ori starts at 24, Ela starts at 36
    # i=2 (BILLY): Ori starts at 48, Ela starts at 60
    
    ori_start = i * (2 * ITEM_COUNT)
    ela_start = ori_start + ITEM_COUNT
    
    print(f"\n【{method}】")
    print(f"  Originality: columns {ori_start} to {ori_start + ITEM_COUNT - 1}")
    print(f"  Elaboration: columns {ela_start} to {ela_start + ITEM_COUNT - 1}")
    
    # --- Originality ---
    # 矩陣切片: 取所有 Rater 的 Ori 分數 (12 個 items)
    ori_data_slice = RATINGS[:, ori_start : ori_start + ITEM_COUNT]
    
    # 檢查是否有 NaN 值
    ori_valid_mask = ~np.isnan(ori_data_slice)
    print(f"  Originality 有效評分: {np.sum(ori_valid_mask)} / {ori_data_slice.size}")
    
    # 轉置: 將數據從 (Rater x Item) 轉為 (Item x Rater)，以符合 IRR 庫的要求
    ori_data_T = ori_data_slice.T
    
    # 執行 IRR 計算 (使用 'krippendorff' 庫)
    # level_of_measurement 可以是 'nominal', 'ordinal', 'interval', 'ratio'
    # 對於李克特量表 (1-5)，通常使用 'ordinal' 或 'interval'
    try:
        alpha_ori = krippendorff.alpha(reliability_data=ori_data_T, level_of_measurement='ordinal')
        RESULTS[f'{method}_Originality'] = alpha_ori
        print(f"  ✓ Originality Alpha = {alpha_ori:.4f}")
    except Exception as e:
        print(f"  ✗ Originality 計算失敗: {e}")
        RESULTS[f'{method}_Originality'] = np.nan
    
    # --- Elaboration ---
    ela_data_slice = RATINGS[:, ela_start : ela_start + ITEM_COUNT]
    
    # 檢查是否有 NaN 值
    ela_valid_mask = ~np.isnan(ela_data_slice)
    print(f"  Elaboration 有效評分: {np.sum(ela_valid_mask)} / {ela_data_slice.size}")
    
    ela_data_T = ela_data_slice.T

    # 執行 IRR 計算
    try:
        alpha_ela = krippendorff.alpha(reliability_data=ela_data_T, level_of_measurement='ordinal')
        RESULTS[f'{method}_Elaboration'] = alpha_ela
        print(f"  ✓ Elaboration Alpha = {alpha_ela:.4f}")
    except Exception as e:
        print(f"  ✗ Elaboration 計算失敗: {e}")
        RESULTS[f'{method}_Elaboration'] = np.nan

# 輸出最終結果
print("\n" + "="*60)
print("Krippendorff's Alpha 計算結果")
print("="*60)

results_df = pd.DataFrame({
    'Method': [k.replace('_', ' ') for k in RESULTS.keys()],
    'Alpha': [f"{v:.4f}" if not np.isnan(v) else "N/A" for v in RESULTS.values()]
})

print(results_df.to_string(index=False))

# 保存結果到 CSV
results_df.to_csv('krippendorff_alpha_results.csv', index=False)
print("\n✓ 結果已儲存至: krippendorff_alpha_results.csv")

# 解讀指南
print("\n" + "="*60)
print("Krippendorff's Alpha 解讀指南")
print("="*60)
print("α ≥ 0.800: 優秀的信度 (Excellent)")
print("0.667 ≤ α < 0.800: 可接受的信度 (Acceptable)")
print("α < 0.667: 信度不足 (Insufficient)")
print("="*60)