# tsne
from sklearn.manifold import TSNE
import umap

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 步驟 1: 設定您的檔案路徑 ---
PERSONA_FILES = {
    'Analytical Thinker': './Llama-3.1-8B-Instruct/multi_role/analytical_thinker_response_avg_diff.pt',
    'Creative Professional': './Llama-3.1-8B-Instruct/multi_role/creative_professional_response_avg_diff.pt',
    'Environmentalist': './Llama-3.1-8B-Instruct/multi_role/environmentalist_response_avg_diff.pt',
    'Futurist': './Llama-3.1-8B-Instruct/multi_role/futurist_response_avg_diff.pt',

    # 'Social Entrepreneur': './Llama-3.1-8B-Instruct/multi_role/social_entrepreneur.pt',
    # 'Industry Insider': './Llama-3.1-8B-Instruct/multi_role/industry_insider_response_avg_diff.pt',
}

# --- 新增：定義您想要融合的 Persona ---
PERSONAS_TO_MERGE = ['Environmentalist', 'Creative Professional']


MODEL_NAME = "Llama-3.1-8B-Instruct" # 已根據您的圖表更新
OUTPUT_DIR = "vector_analysis_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 步驟 2: 載入向量並計算範數 ---
def load_and_calculate_norms(filepath):
    try:
        layer_vectors = torch.load(filepath, map_location='cpu')
        norms = torch.linalg.norm(layer_vectors, ord=2, dim=1)
        return norms.numpy(), layer_vectors
    except FileNotFoundError:
        print(f"警告：找不到檔案 {filepath}，將跳過。")
        return None, None

print("--- 正在載入向量並計算每一層的範數 ---")
all_norms_data = []
all_layer_vectors = {} # 儲存完整的 Tensors
for name, path in PERSONA_FILES.items():
    norms, layer_vectors = load_and_calculate_norms(path)
    if norms is not None:
        all_layer_vectors[name] = layer_vectors
        for layer_idx, norm_value in enumerate(norms):
            all_norms_data.append({
                "Persona": name,
                "Layer": layer_idx,
                "L2 Norm": norm_value
            })

df_norms = pd.DataFrame(all_norms_data)
print("範數計算完成。")


# --- 步驟 3: 視覺化分析 ---

# # 圖表一：每一層的範數變化（折線圖）
# print("\n--- 正在生成圖表一：各層範數變化折線圖 ---")
# plt.style.use('seaborn-v0_8-whitegrid')
# plt.figure(figsize=(15, 8))
# # ... (此部分程式碼與您現有版本相同，為簡潔省略)
# # sns.lineplot(...)
# # plt.savefig(...)
# print(f"圖表已儲存至 {OUTPUT_DIR}/norm_across_layers.png")
# plt.close()

# # 圖表二：平均範數比較（柱狀圖）
# print("\n--- 正在生成圖表二：平均範數強度柱狀圖 ---")
# df_avg_norms = df_norms.groupby('Persona')['L2 Norm'].mean().reset_index().sort_values(by='L2 Norm', ascending=False)
# plt.figure(figsize=(12, 7))
# # ... (此部分程式碼與您現有版本相同，為簡潔省略)
# # sns.barplot(...)
# # plt.savefig(...)
# print(f"圖表已儲存至 {OUTPUT_DIR}/average_norm_strength.png")
# plt.close()


# --- 新增分析三：追蹤合併後 L2 Norm 的變化 ---
print("\n--- 正在執行分析三：合併後 L2 Norm 變化追蹤 ---")

# 篩選出要合併的向量
parent_vectors_tensors = [all_layer_vectors[name] for name in PERSONAS_TO_MERGE if name in all_layer_vectors]

if len(parent_vectors_tensors) == len(PERSONAS_TO_MERGE):
    # 逐層計算
    num_layers = parent_vectors_tensors[0].shape[0]
    merge_analysis_results = []

    for layer_idx in range(num_layers):
        parent_norms_at_layer = [torch.linalg.norm(vec[layer_idx]).item() for vec in parent_vectors_tensors]
        
        # 取得該層的父向量
        parent_vecs_at_layer = torch.stack([vec[layer_idx] for vec in parent_vectors_tensors])
        
        # 計算合併後的向量
        merged_vector_at_layer = torch.mean(parent_vecs_at_layer, dim=0)
        
        # 計算合併後向量的範數
        merged_norm_at_layer = torch.linalg.norm(merged_vector_at_layer).item()
        
        # 計算父向量範數的平均值
        avg_parent_norm_at_layer = np.mean(parent_norms_at_layer)
        
        # 計算稀釋/放大比例
        dilution_ratio = merged_norm_at_layer / avg_parent_norm_at_layer
        
        merge_analysis_results.append({
            "Layer": layer_idx,
            "Avg Parent Norm": avg_parent_norm_at_layer,
            "Merged Norm": merged_norm_at_layer,
            "Dilution Ratio": dilution_ratio
        })

    df_merge_analysis = pd.DataFrame(merge_analysis_results)
    
    # 打印總結
    avg_dilution = df_merge_analysis['Dilution Ratio'].mean()
    print("\n合併分析總結:")
    print(f"合併的 Personas: {', '.join(PERSONAS_TO_MERGE)}")
    print(f"平均父向量範數: {df_merge_analysis['Avg Parent Norm'].mean():.4f}")
    print(f"平均合併後範數: {df_merge_analysis['Merged Norm'].mean():.4f}")
    print(f"平均稀釋比例: {avg_dilution:.4f} (若 < 1 表示稀釋, > 1 表示放大)")

    # 繪製稀釋比例圖
    plt.figure(figsize=(15, 7))
    sns.lineplot(data=df_merge_analysis, x='Layer', y='Dilution Ratio', marker='o')
    plt.axhline(y=1, color='r', linestyle='--', label='No Change (Ratio=1)')
    plt.title(f'Merged Vector Norm Dilution Ratio Across Layers', fontsize=18)
    plt.xlabel('Model Layer Index', fontsize=12)
    plt.ylabel('Merged Norm / Average Parent Norm', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'merge_dilution_ratio.png'))
    print(f"稀釋比例圖已儲存至 {OUTPUT_DIR}/merge_dilution_ratio.png")
    plt.close()

else:
    print("錯誤：無法找到所有指定的 Persona 向量進行合併分析。")

# # --- 新增分析四：t-SNE / UMAP 可視化向量分布（含 Layer 深淺） ---
from sklearn.manifold import TSNE
import umap

print("\n--- 正在執行分析四：t-SNE / UMAP 可視化向量分布 ---")

# 將所有 persona vectors 展平為 [num_layers, hidden_size]
vector_list = []
label_list = []

for persona_name, tensor in all_layer_vectors.items():
    for layer_idx in range(tensor.shape[0]):
        vector_list.append(tensor[layer_idx].numpy())
        label_list.append(persona_name)

# 新增合併向量
if len(parent_vectors_tensors) == len(PERSONAS_TO_MERGE):
    merged_vectors = []
    for layer_idx in range(num_layers):
        merged = torch.mean(torch.stack([vec[layer_idx] for vec in parent_vectors_tensors]), dim=0)
        merged_vectors.append(merged.numpy())
        label_list.append("Merged Persona")
    vector_list.extend(merged_vectors)

# 降維處理
print("正在使用 UMAP 將向量降維至 2D ...")
reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='cosine', random_state=42)
embedding = reducer.fit_transform(vector_list)  # shape: [total_vectors, 2]

# 視覺化
plt.figure(figsize=(10, 8))
sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=label_list, palette='Set2', s=60)
plt.title("UMAP Projection of Persona Vectors (All Layers + Merged)", fontsize=16)
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.tight_layout()
tsne_output_path = os.path.join(OUTPUT_DIR, 'umap_persona_projection.png')
plt.savefig(tsne_output_path)
plt.close()

print(f"✅ UMAP 圖已儲存至：{tsne_output_path}")

# from sklearn.manifold import TSNE
# import umap

# print("\n--- 正在執行分析四：t-SNE / UMAP 可視化向量分布（含 Layer 深淺） ---")

# # 準備資料
# vector_list = []
# persona_labels = []
# layer_indices = []

# for persona_name, tensor in all_layer_vectors.items():
#     for layer_idx in range(tensor.shape[0]):
#         vector_list.append(tensor[layer_idx].numpy())
#         persona_labels.append(persona_name)
#         layer_indices.append(layer_idx)

# # 加上 Merged Persona
# if len(parent_vectors_tensors) == len(PERSONAS_TO_MERGE):
#     for layer_idx in range(num_layers):
#         merged = torch.mean(torch.stack([vec[layer_idx] for vec in parent_vectors_tensors]), dim=0)
#         vector_list.append(merged.numpy())
#         persona_labels.append("Merged Persona")
#         layer_indices.append(layer_idx)

# # 降維
# print("正在使用 UMAP 將向量降維至 2D ...")
# reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='cosine', random_state=42)
# embedding = reducer.fit_transform(vector_list)

# # 合併成 DataFrame 方便畫圖
# df_plot = pd.DataFrame({
#     "x": embedding[:, 0],
#     "y": embedding[:, 1],
#     "Persona": persona_labels,
#     "Layer": layer_indices
# })

# # 畫圖
# plt.figure(figsize=(11, 9))
# sns.scatterplot(
#     data=df_plot,
#     x="x", y="y",
#     hue="Layer",  # Layer 控制顏色深淺
#     style="Persona",  # Persona 控制 marker 形狀
#     palette="viridis",  # 可換成 "coolwarm", "cividis", "Spectral" 等
#     s=70
# )

# plt.title("UMAP Projection of Persona Vectors (Colored by Layer)", fontsize=16)
# plt.xlabel("Component 1")
# plt.ylabel("Component 2")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()

# output_path = os.path.join(OUTPUT_DIR, "umap_with_layer_shade.png")
# plt.savefig(output_path)
# plt.close()
# print(f"✅ 含 Layer 深淺的圖已儲存至：{output_path}")

# --- 新增分析五：計算 Merged Vector 與各 Parent Vector 的 Cosine Similarity ---
from sklearn.metrics.pairwise import cosine_similarity

print("\n--- 正在執行分析五：Merged Vector 與 Parent 向量的 Cosine Similarity 分析 ---")

cosine_results = []

if len(parent_vectors_tensors) == len(PERSONAS_TO_MERGE):
    for layer_idx in range(num_layers):
        # 取得合併後向量
        merged_vector = torch.mean(
            torch.stack([vec[layer_idx] for vec in parent_vectors_tensors]), dim=0
        ).numpy().reshape(1, -1)  # shape: [1, hidden]

        # 計算每個 parent 的 cosine similarity
        for i, persona_name in enumerate(PERSONAS_TO_MERGE):
            parent_vector = parent_vectors_tensors[i][layer_idx].numpy().reshape(1, -1)
            similarity = cosine_similarity(merged_vector, parent_vector)[0][0]

            cosine_results.append({
                "Layer": layer_idx,
                "Persona": persona_name,
                "Cosine Similarity": similarity
            })

    df_cosine = pd.DataFrame(cosine_results)

    # 🔥 視覺化
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df_cosine, x="Layer", y="Cosine Similarity", hue="Persona", marker="o")
    plt.title("Cosine Similarity: Merged Vector vs Each Parent Persona (per Layer)")
    plt.ylim(0, 1.0)
    plt.xlabel("Model Layer Index")
    plt.ylabel("Cosine Similarity")
    plt.legend(title="Parent Persona")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "cosine_similarity_per_layer.png")
    plt.savefig(output_path)
    plt.close()

    print(f"✅ Cosine Similarity 圖已儲存至：{output_path}")
else:
    print("⚠️ 找不到所有指定的 parent vectors，無法進行 cosine similarity 分析。")
