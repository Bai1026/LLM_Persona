import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from rich import print
from itertools import combinations

# read data from csv
data_df = pd.read_csv('human_scores.csv', header=None)

# 只取前 72 個欄位的評分資料（從第 2 欄開始，跳過第一欄的 "People" 標籤）
# 並且跳過第一列的標題和最後幾列的平均值
RATINGS = data_df.iloc[1:, 1:73].to_numpy()

# 轉換為 float，將無法轉換的值設為 NaN
RATINGS = pd.DataFrame(RATINGS).apply(pd.to_numeric, errors='coerce').to_numpy()

print(f"原始資料維度: {RATINGS.shape}")
print(f"前幾列資料:\n{RATINGS[:3, :10]}")  # 顯示前 3 個評分者的前 10 個分數

# --- 定義方法和切片邏輯 ---
METHODS = ['SA', 'LLMD', 'BILLY']
METRICS = ['Originality', 'Elaboration']
ITEM_COUNT = 12

print("\n" + "="*80)
print("開始計算 Kendall's τ (Tau) Correlation Coefficient")
print("="*80)

# 儲存所有結果
all_results = []

# --- 循環計算每個方法和指標的 Kendall's τ ---
for i, method in enumerate(METHODS):
    # 計算起始索引
    # i=0 (SA): Ori starts at 0, Ela starts at 12
    # i=1 (LLMD): Ori starts at 24, Ela starts at 36
    # i=2 (BILLY): Ori starts at 48, Ela starts at 60
    
    ori_start = i * (2 * ITEM_COUNT)
    ela_start = ori_start + ITEM_COUNT
    
    print(f"\n{'='*80}")
    print(f"【{method}】")
    print(f"{'='*80}")
    
    for metric_idx, (metric_name, start_idx) in enumerate([
        ('Originality', ori_start),
        ('Elaboration', ela_start)
    ]):
        print(f"\n  【{metric_name}】 (columns {start_idx} to {start_idx + ITEM_COUNT - 1})")
        
        # 取得該指標的資料
        data_slice = RATINGS[:, start_idx : start_idx + ITEM_COUNT]
        
        # 獲取評分者數量
        n_raters = data_slice.shape[0]
        
        # 檢查有效資料
        valid_mask = ~np.isnan(data_slice)
        print(f"    有效評分: {np.sum(valid_mask)} / {data_slice.size}")
        print(f"    評分者數量: {n_raters}")
        
        # 計算所有評分者之間的 pairwise Kendall's τ
        tau_values = []
        p_values = []
        pairs = []
        
        # 對每一對評分者計算 Kendall's τ
        for r1, r2 in combinations(range(n_raters), 2):
            rater1_scores = data_slice[r1, :]
            rater2_scores = data_slice[r2, :]
            
            # 只使用兩個評分者都有評分的項目
            valid_items = ~(np.isnan(rater1_scores) | np.isnan(rater2_scores))
            
            if np.sum(valid_items) >= 2:  # 至少需要 2 個共同評分的項目
                try:
                    tau, p_value = kendalltau(
                        rater1_scores[valid_items], 
                        rater2_scores[valid_items]
                    )
                    tau_values.append(tau)
                    p_values.append(p_value)
                    pairs.append((r1, r2))
                    print(f"    Rater {r1+1} vs Rater {r2+1}: τ = {tau:.4f}, p = {p_value:.4f} (n={np.sum(valid_items)})")
                except Exception as e:
                    print(f"    Rater {r1+1} vs Rater {r2+1}: 計算失敗 - {e}")
        
        # 計算平均 Kendall's τ
        if tau_values:
            # 移除 NaN 值後再計算統計量
            tau_values_clean = [t for t in tau_values if not np.isnan(t)]
            
            if tau_values_clean:
                mean_tau = np.mean(tau_values_clean)
                std_tau = np.std(tau_values_clean)
                median_tau = np.median(tau_values_clean)
                min_tau = np.min(tau_values_clean)
                max_tau = np.max(tau_values_clean)
                
                print(f"\n    統計摘要:")
                print(f"      平均 τ: {mean_tau:.4f} ± {std_tau:.4f}")
                print(f"      中位數 τ: {median_tau:.4f}")
                print(f"      範圍: [{min_tau:.4f}, {max_tau:.4f}]")
                print(f"      有效配對數量: {len(tau_values_clean)} / {len(tau_values)}")
                if len(tau_values_clean) < len(tau_values):
                    print(f"      ⚠️ 有 {len(tau_values) - len(tau_values_clean)} 個配對產生 NaN (可能是所有值相同)")
            else:
                mean_tau = np.nan
                std_tau = np.nan
                median_tau = np.nan
                min_tau = np.nan
                max_tau = np.nan
                print(f"\n    ⚠️ 所有配對都產生 NaN，無法計算統計量")
                print(f"      配對數量: {len(tau_values)}")
            
            # 儲存結果
            all_results.append({
                'Method': method,
                'Metric': metric_name,
                'Mean_Tau': mean_tau,
                'Std_Tau': std_tau,
                'Median_Tau': median_tau,
                'Min_Tau': min_tau,
                'Max_Tau': max_tau,
                'N_Pairs': len(tau_values)
            })
        else:
            print(f"\n    ⚠️ 無法計算 Kendall's τ")
            all_results.append({
                'Method': method,
                'Metric': metric_name,
                'Mean_Tau': np.nan,
                'Std_Tau': np.nan,
                'Median_Tau': np.nan,
                'Min_Tau': np.nan,
                'Max_Tau': np.nan,
                'N_Pairs': 0
            })

# 輸出最終結果表格
print("\n" + "="*80)
print("Kendall's τ 計算結果總表")
print("="*80)

results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))

# 保存詳細結果
results_df.to_csv('kendall_tau_results.csv', index=False)
print(f"\n✓ 詳細結果已儲存至: kendall_tau_results.csv")

# 建立簡化版結果（只顯示平均值）
summary_df = results_df[['Method', 'Metric', 'Mean_Tau', 'Std_Tau']].copy()
summary_df['Result'] = summary_df.apply(
    lambda row: f"{row['Mean_Tau']:.4f} ± {row['Std_Tau']:.4f}" 
    if not np.isnan(row['Mean_Tau']) else "N/A", 
    axis=1
)

print("\n" + "="*80)
print("簡化結果 (平均 Kendall's τ ± 標準差)")
print("="*80)
print(summary_df[['Method', 'Metric', 'Result']].to_string(index=False))

# 計算 Total Average (跨所有方法)
print("\n" + "="*80)
print("Total Average Correlation (跨所有方法)")
print("="*80)

ori_results = results_df[results_df['Metric'] == 'Originality']['Mean_Tau'].dropna()
ela_results = results_df[results_df['Metric'] == 'Elaboration']['Mean_Tau'].dropna()

if len(ori_results) > 0:
    ori_mean = ori_results.mean()
    ori_std = ori_results.std()
    print(f"Originality (Overall): {ori_mean:.4f} ± {ori_std:.4f}")
    print(f"  - 基於 {len(ori_results)} 個方法的平均")
else:
    print(f"Originality (Overall): N/A")

if len(ela_results) > 0:
    ela_mean = ela_results.mean()
    ela_std = ela_results.std()
    print(f"Elaboration (Overall): {ela_mean:.4f} ± {ela_std:.4f}")
    print(f"  - 基於 {len(ela_results)} 個方法的平均")
else:
    print(f"Elaboration (Overall): N/A")

# 解讀指南
print("\n" + "="*80)
print("Kendall's τ 解讀指南")
print("="*80)
print("τ = 1.0: 完全正相關 (Perfect positive correlation)")
print("0.7 < τ < 1.0: 強正相關 (Strong positive correlation)")
print("0.4 < τ < 0.7: 中等正相關 (Moderate positive correlation)")
print("0.0 < τ < 0.4: 弱正相關 (Weak positive correlation)")
print("τ = 0.0: 無相關 (No correlation)")
print("τ < 0.0: 負相關 (Negative correlation)")
print("="*80)
print("\n註: p < 0.05 表示相關性在統計上顯著")
print("="*80)
