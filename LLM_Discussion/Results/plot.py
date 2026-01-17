import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

# 設置Seaborn風格
sns.set_theme(style="white")

# --- 新圖表的數據 ---
labels = ['SA', 'SA-MRP', 'LLM Discussion', 'BILLY']
values = [22.3, 221.2, 88853.0, 62.2]
colors = ['royalblue', 'indianred', 'goldenrod', '#34A853']

# 設置圖表大小
fig, ax = plt.subplots(figsize=(8.5, 6.5))

# --- 修改處：加入 width 參數來調整寬度 ---
# 將寬度設為 0.5，您可以嘗試不同的值來找到最適合的效果
bars = ax.bar(labels, values, color=colors, width=0.45)

# --- 關鍵：設置Y軸為對數尺度 ---
ax.set_yscale('log')
ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

# --- 添加標籤和標題 ---
ax.set_ylabel('Average token per query', fontsize=14)
ax.set_xlabel('10000 queries', fontsize=14, labelpad=15)

# --- 在每個長條上添加數值標籤 ---
def autolabel(rects):
    """在每個長條上方附加一個文本標籤，顯示其高度。"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)
autolabel(bars)

# --- 加上我們之前設定的風格 ---
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

ax.set_ylim(top=290000)

# --- 圖例設定 ---
legend_elements = [Patch(facecolor=colors[i], label=labels[i]) for i in range(len(labels))]
ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=11)

# 隱藏X軸的刻度標籤
ax.set_xticklabels(['', '', '', ''])

# 調整整體佈局
fig.tight_layout()

# 儲存圖檔
plt.savefig('token_comparision_1007.png', dpi=300, bbox_inches='tight')

# 顯示圖表
plt.show()