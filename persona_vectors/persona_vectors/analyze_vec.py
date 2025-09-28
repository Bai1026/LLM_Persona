import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

PERSONA_FILES = {
    'Analytical Thinker': './Llama-3.1-8B-Instruct/multi_role/analytical_thinker_response_avg_diff.pt',
    'Creative Professional': './Llama-3.1-8B-Instruct/multi_role/creative_professional_response_avg_diff.pt',
    'Environmentalist': './Llama-3.1-8B-Instruct/multi_role/environmentalist_response_avg_diff.pt',
    'Futurist': './Llama-3.1-8B-Instruct/multi_role/futurist_response_avg_diff.pt',

    # 'Social Entrepreneur': './Llama-3.1-8B-Instruct/multi_role/social_entrepreneur.pt',
    # 'Industry Insider': './Llama-3.1-8B-Instruct/multi_role/industry_insider_response_avg_diff.pt',
}


MODEL_NAME = "Llama-3.1-8B-Instruct"
OUTPUT_DIR = "vector_analysis_charts"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_calculate_norms(filepath):
    """
    載入 .pt 檔案 (shape: [layers, hidden_dim])
    並計算每一層向量的 L2 範數 (Euclidean norm)。
    """
    try:
        layer_vectors = torch.load(filepath, map_location='cpu')
        # 對每一層 (dim=1) 計算 L2 範數
        norms = torch.linalg.norm(layer_vectors, ord=2, dim=1)
        return norms.numpy()
    except FileNotFoundError:
        print(f"警告：找不到檔案 {filepath}，將跳過。")
        return None

print("--- 正在載入向量並計算每一層的範數 ---")
all_norms_data = []
for name, path in PERSONA_FILES.items():
    norms = load_and_calculate_norms(path)
    if norms is not None:
        for layer_idx, norm_value in enumerate(norms):
            all_norms_data.append({
                "Persona": name,
                "Layer": layer_idx,
                "L2 Norm": norm_value
            })

# 將數據轉換為 Pandas DataFrame，方便後續繪圖
df_norms = pd.DataFrame(all_norms_data)
print("範數計算完成。")


# --- 步驟 3: 視覺化分析 ---

# 圖表一：每一層的範數變化（折線圖） - 更深入的分析
print("\n--- 正在生成圖表一：各層範數變化折線圖 ---")
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(15, 8))
sns.lineplot(data=df_norms, x='Layer', y='L2 Norm', hue='Persona', marker='o', markersize=5)
plt.title(f'Persona Vector L2 Norm Across Layers ({MODEL_NAME})', fontsize=18, pad=20)
plt.xlabel('Model Layer Index', fontsize=12)
plt.ylabel('L2 Norm (Vector Length)', fontsize=12)
plt.legend(title='Persona', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'norm_across_layers.png'))
print(f"圖表已儲存至 {OUTPUT_DIR}/norm_across_layers.png")
plt.close()


# 圖表二：平均範數比較（柱狀圖） - 您最初想要的「強度圖」
print("\n--- 正在生成圖表二：平均範數強度柱狀圖 ---")
# 計算每個 Persona 的平均範數
df_avg_norms = df_norms.groupby('Persona')['L2 Norm'].mean().reset_index().sort_values(by='L2 Norm', ascending=False)

plt.figure(figsize=(12, 7))
barplot = sns.barplot(data=df_avg_norms, x='L2 Norm', y='Persona', palette='viridis')
plt.title(f'Average Persona Vector Strength (L2 Norm) ({MODEL_NAME})', fontsize=18, pad=20)
plt.xlabel('Average L2 Norm (Higher = Stronger Influence)', fontsize=12)
plt.ylabel('Persona', fontsize=12)
# 在每個柱子上顯示數值
for i in barplot.containers:
    barplot.bar_label(i, fmt='%.2f', fontsize=10, padding=3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'average_norm_strength.png'))
print(f"圖表已儲存至 {OUTPUT_DIR}/average_norm_strength.png")
plt.close()

print("\n分析完成！")
