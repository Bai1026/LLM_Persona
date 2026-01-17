import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 您的原始數據 (Wide Format)

all_data = {
    "gemma_ela": {
        "alpha": [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0],
        12: [4.967, 5.000, 5.000, 4.967, 5.000, 4.933, 4.967],
        16: [4.967, 5.000, 4.833, 4.967, 5.000, 4.950, 3.778],
        20: [5.000, 4.867, 4.933, 4.967, 5.000, 4.967, 3.133],
        24: [5.000, 4.910, 4.967, 5.000, 4.900, 5.000, 3.567]
    },
    "gemma_ori": {
        "alpha": [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0],
        12: [4.900, 4.867, 4.933, 4.800, 4.967, 4.900, 4.933],
        16: [4.933, 4.967, 4.933, 4.933, 4.967, 4.967, 4.944],
        20: [5.000, 4.867, 5.000, 4.967, 5.000, 5.000, 4.933],
        24: [4.933, 5.000, 4.867, 4.920, 5.000, 5.000, 4.900]
    },
    "qwen_ela": {
        "alpha": [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0],
        12: [4.900, 4.500, 4.433, 4.467, 4.767, 4.367, 4.933],
        16: [4.600, 4.500, 4.300, 4.633, 4.633, 4.633, 3.600],
        20: [4.800, 4.567, 4.667, 4.600, 4.833, 4.900, 1.800],
        24: [4.733, 4.500, 4.633, 4.533, 4.733, 4.700, 4.367]
    },
    "qwen_ori": {
        "alpha": [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0],
        12: [4.300, 4.000, 4.100, 4.167, 4.167, 4.200, 4.567],
        16: [4.233, 4.233, 4.133, 4.533, 4.467, 4.633, 3.833],
        20: [4.533, 4.233, 4.400, 4.300, 4.633, 4.600, 3.533],
        24: [4.200, 4.133, 4.100, 4.433, 4.367, 4.500, 4.400]
    },
    "llama_ela": {
        "alpha": [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0],
        12: [4.900, 4.867, 4.833, 4.700, 4.867, 4.933, 4.800],
        16: [4.933, 4.633, 4.767, 4.900, 4.867, 4.833, 3.700],
        20: [4.900, 4.767, 4.900, 4.767, 4.833, 4.900, 4.467],
        24: [4.900, 4.567, 4.800, 4.700, 4.833, 4.700, 4.333]
    },
    "llama_ori": {
        "alpha": [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0],
        12: [4.133, 4.467, 4.433, 4.600, 4.667, 4.767, 4.667],
        16: [4.333, 4.267, 4.600, 4.633, 4.800, 4.833, 4.267],
        20: [4.467, 4.433, 4.500, 4.567, 4.533, 4.533, 4.800],
        24: [4.400, 4.333, 4.467, 4.433, 4.533, 4.433, 4.367]
    }
}

# data = {
#     'alpha': [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5],
#     12: [4.9, 4.867, 4.833, 4.700, 4.867, 4.933, 4.8],
#     16: [4.933, 4.633, 4.767, 4.900, 4.867, 4.833, 3.7],
#     20: [4.9, 4.767, 4.900, 4.767, 4.833, 4.900, 4.467],
#     24: [4.9, 4.567, 4.800, 4.700, 4.833, 4.700, 4.333]
# }

# 設定繪圖風格
sns.set_theme(style="whitegrid")

# 定義標題對應
titles = {
    'llama_ela': 'Llama-3.1-8B (Elaboration)',
    'llama_ori': 'Llama-3.1-8B (Originality)',
    'qwen_ela': 'Qwen2.5-7B (Elaboration)',
    'qwen_ori': 'Qwen2.5-7B (Originality)',
    'gemma_ela': 'Gemma-3-4B (Elaboration)',
    'gemma_ori': 'Gemma-3-4B (Originality)'
}

# 先繪製個別圖
for data_type in ['llama_ela', 'llama_ori', 'qwen_ela', 'qwen_ori', 'gemma_ela', 'gemma_ori']:
        
    data = all_data[data_type]

    df_wide = pd.DataFrame(data).set_index('alpha')

    # 執行 Long Format 轉換
    df_long = df_wide.reset_index().melt(
        id_vars='alpha',
        var_name='Layer',
        value_name='Ela_Score'
    )

    # 確保 Layer 是數值型態以便繪圖
    df_long['Layer'] = df_long['Layer'].astype(int)

    # *** 關鍵修正：將 alpha 值轉換為字串，確保圖例顯示正確標籤 ***
    df_long['alpha'] = df_long['alpha'].astype(str)

    # 繪製線圖
    plt.figure(figsize=(8, 6)) # 調整圖表大小
    line_plot = sns.lineplot(
        data=df_long,
        x='Layer',          # X 軸：層數
        y='Ela_Score',      # Y 軸：Elaboration 分數
        hue='alpha',        # 顏色/線條：導向強度 (alpha) - 現在是字串
        marker='o',         # 在數據點上標記圓圈
        palette='viridis'   # 選擇顏色主題
    )

    # 標記 default setting (Layer 20, Alpha 2.0) 用紅色圈圈
    default_point = df_long[(df_long['Layer'] == 20) & (df_long['alpha'] == '2.0')]
    if not default_point.empty:
        plt.scatter(
            default_point['Layer'], 
            default_point['Ela_Score'], 
            color='red', 
            s=200, 
            facecolors='none', 
            edgecolors='red', 
            linewidths=3,
            zorder=5,
            label='Default Setting'
        )

    # 設定標籤和標題
    plt.title(titles[data_type], fontsize=14)
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Score', fontsize=12)

    # 調整 X 軸刻度，確保只顯示您的實驗層數 (12, 16, 20, 24)
    plt.xticks(df_long['Layer'].unique())

    # 調整 Y 軸範圍以放大差異
    if data_type == 'gemma_ela':
        plt.ylim(2.0, 5.2) # 調整範圍以涵蓋 alpha=5 的低點
    elif data_type == 'qwen_ela':
        plt.ylim(1.0, 5.2) # 調整範圍以涵蓋 alpha=5 的低點
    else:
        plt.ylim(3.5, 5.2) # 調整範圍以涵蓋 alpha=5 的低點

    # 添加圖例標題
    line_plot.legend_.set_title('Steering Factor α')

    # 繪圖順序建議：先儲存，再顯示
    plt.savefig(f'{data_type}.png', dpi=300, bbox_inches='tight')
    plt.show()

# 建立組合圖 (2x3 子圖)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Ablation Study: Layer and Steering Factor Effects', fontsize=16, fontweight='bold')

# 重新排列：第一列是 Originality，第二列是 Elaboration
data_types = ['llama_ori', 'qwen_ori', 'gemma_ori', 'llama_ela', 'qwen_ela', 'gemma_ela']

for idx, data_type in enumerate(data_types):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    data = all_data[data_type]
    df_wide = pd.DataFrame(data).set_index('alpha')
    df_long = df_wide.reset_index().melt(
        id_vars='alpha',
        var_name='Layer',
        value_name='Ela_Score'
    )
    df_long['Layer'] = df_long['Layer'].astype(int)
    df_long['alpha'] = df_long['alpha'].astype(str)
    
    # 在子圖上繪製
    sns.lineplot(
        data=df_long,
        x='Layer',
        y='Ela_Score',
        hue='alpha',
        marker='o',
        palette='viridis',
        ax=ax,
        legend=(idx == 0)  # 只在第一個子圖顯示圖例
    )
    
    # 標記 default setting
    default_point = df_long[(df_long['Layer'] == 20) & (df_long['alpha'] == '2.0')]
    if not default_point.empty:
        ax.scatter(
            default_point['Layer'], 
            default_point['Ela_Score'], 
            color='red', 
            s=150, 
            facecolors='none', 
            edgecolors='red', 
            linewidths=2.5,
            zorder=5
        )
    
    # 設定子圖標題和標籤
    ax.set_title(titles[data_type], fontsize=12, fontweight='bold')
    ax.set_xlabel('Layer', fontsize=10)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_xticks(df_long['Layer'].unique())
    
    # 調整 Y 軸範圍
    if data_type == 'gemma_ela':
        ax.set_ylim(2.0, 5.2)
    elif data_type == 'qwen_ela':
        ax.set_ylim(1.0, 5.2)
    else:
        ax.set_ylim(3.5, 5.2)
    
    # 只在第一個子圖設定圖例
    if idx == 0:
        legend = ax.legend(title='Steering Factor α', loc='lower left', fontsize=8)
        legend.get_title().set_fontsize(9)

# 調整子圖間距
plt.tight_layout()

# 儲存組合圖
plt.savefig('combined_ablation_study.png', dpi=300, bbox_inches='tight')
plt.show()