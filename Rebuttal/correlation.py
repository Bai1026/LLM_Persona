import numpy as np
from scipy.stats import spearmanr, pearsonr
from rich import print

# --- 您的數據 ---

# # 人類平均分數 (Human Average Scores)
# human_str = "2.666666667, 3.555555556, 4.111111111, 3.444444444, 3.111111111, 3.111111111, 2.111111111, 3.555555556, 3.666666667, 2.666666667, 2.777777778"

# # LLM 評分 (LLM Scores)
# llm_str = "3, 4.665, 3.66, 3, 3, 4.33, 2.66, 3.5825, 4.33, 2.66, 3.08"

# read data from csv file
with open('merged_human_evaluation.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    human_str = lines[1].strip()[1:]
    llm_str = lines[2].strip()[4:]

print("--- 原始數據 ---")
print(f"Human Scores: {human_str}")
print(f"LLM Scores: {llm_str}")

# # --- 步驟 1: 解析並轉換成數字陣列 ---

def parse_scores(score_str):
    """將逗號分隔的字符串轉換為浮點數列表"""
    cleaned_str = score_str.replace("...", "").replace(" ", "")
    # 使用列表推導式進行轉換和過濾空值
    return [float(x) for x in cleaned_str.split(',') if x]

try:
    human_scores_all = parse_scores(human_str)
    llm_scores_all = parse_scores(llm_str)

    if len(human_scores_all) != len(llm_scores_all):
        print("【錯誤】人類評分與 LLM 評分的數量不匹配，無法計算相關性。")
    else:
        # --- 步驟 2: 執行 Spearman's Rank Correlation ---
        spearman_corr, spearman_p = spearmanr(human_scores_all, llm_scores_all)
        
        # --- 步驟 3: 執行 Pearson Correlation ---
        pearson_corr, pearson_p = pearsonr(human_scores_all, llm_scores_all)

        print("--- Spearman's Rank Correlation Result (Combined Data) ---")
        print(f"Total Number of Paired Scores: {len(human_scores_all)}")
        print(f"Spearman's Rho (相關係數): {spearman_corr:.4f}")
        print(f"P-value: {spearman_p:.4f}")
        
        print("\n--- Pearson Correlation Result (Combined Data) ---")
        print(f"Total Number of Paired Scores: {len(human_scores_all)}")
        print(f"Pearson's r (相關係數): {pearson_corr:.4f}")
        print(f"P-value: {pearson_p:.4f}")
        
        print("\n--- 數據分割提醒 ---")
        print("請注意：此結果是將 Originality 和 Elaboration 的分數混在一起計算的。")
        print("若要分別計算，您需要知道並在程式碼中定義 'N_SPLIT' (例如，前 N 個是 Originality)。")

        # --- 建議的分割計算程式碼 (請自行調整 N_SPLIT) ---
        N_SPLIT = 12 # 假設分割點
        
        # Spearman
        ori_spearman, ori_spearman_p = spearmanr(human_scores_all[:N_SPLIT], llm_scores_all[:N_SPLIT])
        ela_spearman, ela_spearman_p = spearmanr(human_scores_all[N_SPLIT:], llm_scores_all[N_SPLIT:])
        
        # Pearson
        ori_pearson, ori_pearson_p = pearsonr(human_scores_all[:N_SPLIT], llm_scores_all[:N_SPLIT])
        ela_pearson, ela_pearson_p = pearsonr(human_scores_all[N_SPLIT:], llm_scores_all[N_SPLIT:])
        
        print(f"\n若 N_SPLIT={N_SPLIT}:")
        print(f"  Originality  - Spearman Rho: {ori_spearman:.4f} (p={ori_spearman_p:.4f})")
        print(f"  Originality  - Pearson r: {ori_pearson:.4f} (p={ori_pearson_p:.4f})")
        print(f"  Elaboration  - Spearman Rho: {ela_spearman:.4f} (p={ela_spearman_p:.4f})")
        print(f"  Elaboration  - Pearson r: {ela_pearson:.4f} (p={ela_pearson_p:.4f})")

except Exception as e:
    print(f"計算錯誤: {e}")