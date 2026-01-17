import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 步驟 1: 將您的實驗數據填入這裡 ---
# 這是您提供的數據，我已為您整理好
# GPT Eval Score (distribute)
data = {
    # 單一 Persona 作為控制組 (Control)
    "Control": {
        "creative": 50.0,
        "environmental": 41.67,
        "analytical": 42.92,
        "futurist": 22.08,
    },
    # 融合後的實驗組 (Experiments)
    "Experiments": [
        # 基礎是 Env
        {"base": "environmental", "added": "analytical", "scores": {"creative": 8.75, "environmental": 66.25, "analytical": 9.17}},
        {"base": "environmental", "added": "creative", "scores": {"creative": 52.5, "environmental": 8.92, "analytical": 7.75}},
        # 基礎是 Cre
        {"base": "creative", "added": "analytical", "scores": {"creative": 10.83, "environmental": 11.25, "analytical": 37.08}},
        {"base": "creative", "added": "futurist", "scores": {"creative": 49.58, "environmental": 9.17, "analytical": 8.33}},
        # 基礎是 Ana
        {"base": "analytical", "added": "futurist", "scores": {"creative": 10.42, "environmental": 10.42, "analytical": 37.08}},
    ]
}

# --- 步驟 2: 計算相對變化 ---
results = []
for exp in data["Experiments"]:
    base_persona = exp["base"]
    added_persona = exp["added"]
    
    # 控制組分數 (單獨表現時的分數)
    control_score = data["Control"].get(base_persona, 0)
    
    # 實驗組分數 (融合後的分數)
    experiment_score = exp["scores"].get(base_persona, 0)
    
    # 計算百分比變化: (實驗組 - 控制組) / 控制組
    if control_score > 0:
        percent_change = ((experiment_score - control_score) / control_score) * 100
    else:
        percent_change = 0
        
    results.append({
        "Base Persona (Y-axis)": base_persona.capitalize(),
        "Added Persona (X-axis)": f"+ {added_persona.capitalize()}",
        "Score Change (%)": percent_change
    })

df = pd.DataFrame(results)

# --- 步驟 3: 創建熱力圖 ---
# 使用 pivot 將數據轉換為矩陣格式
pivot_df = df.pivot(index="Base Persona (Y-axis)", columns="Added Persona (X-axis)", values="Score Change (%)")

# 繪製熱力圖
plt.figure(figsize=(10, 7))
sns.heatmap(
    pivot_df, 
    annot=True,          # 顯示數字
    fmt=".1f",           # 格式化數字為一位小數
    cmap="vlag",         # vlag 是一個很好的「紅-白-藍」色盤，非常適合表示正負變化
    center=0,            # 將 0 設為顏色的中心點
    linewidths=.5,
    cbar_kws={'label': 'Score Change (%) vs. Control'}
)

plt.title("Persona Interaction Matrix: How Merging Amplifies or Suppresses Traits", fontsize=16)
plt.ylabel("Main Persona's Expression Score", fontsize=12)
plt.xlabel("Effect of Adding a Second Persona", fontsize=12)
plt.xticks(rotation=30, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("persona_interaction_heatmap.png")

print("人格互動熱力圖已儲存為 persona_interaction_heatmap.png")
