import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_projection_by_layer(csv_file_path):
    """
    分析每層的 proj_delta_on_a 和 proj_delta_on_b 平均值並繪製折線圖
    """
    
    # 讀取 CSV 檔案
    df = pd.read_csv(csv_file_path)
    
    # 按照 analysis_layer 分組並計算平均值
    layer_averages = df.groupby('analysis_layer').agg({
        'proj_delta_on_a': 'mean',
        'proj_delta_on_b': 'mean'
    }).reset_index()
    
    # 確保層數是從 20 到 32
    layers = range(20, 33)  # 20 到 32
    
    # 建立結果字典
    results = {}
    for layer in layers:
        layer_data = layer_averages[layer_averages['analysis_layer'] == layer]
        if not layer_data.empty:
            results[layer] = {
                'proj_delta_on_a_avg': layer_data['proj_delta_on_a'].iloc[0],
                'proj_delta_on_b_avg': layer_data['proj_delta_on_b'].iloc[0]
            }
    
    # 印出統計資訊
    print("每層平均值統計:")
    print("-" * 60)
    print(f"{'Layer':<8} {'proj_delta_on_a':<20} {'proj_delta_on_b':<20}")
    print("-" * 60)
    
    for layer in sorted(results.keys()):
        avg_a = results[layer]['proj_delta_on_a_avg']
        avg_b = results[layer]['proj_delta_on_b_avg']
        print(f"{layer:<8} {avg_a:<20.4f} {avg_b:<20.4f}")
    
    # 準備繪圖資料
    layers_list = sorted(results.keys())
    avg_a_values = [results[layer]['proj_delta_on_a_avg'] for layer in layers_list]
    avg_b_values = [results[layer]['proj_delta_on_b_avg'] for layer in layers_list]
    
    # 繪製折線圖
    plt.figure(figsize=(12, 8))
    
    plt.plot(layers_list, avg_a_values, marker='o', linewidth=2, 
             label='proj_delta_on_creative mean', color='blue')
    plt.plot(layers_list, avg_b_values, marker='s', linewidth=2, 
             label='proj_delta_on_environmentalist mean', color='red')
    
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('mean value of projection', fontsize=12)
    # plt.title('proj_delta mean value of neutral questions', fontsize=14, fontweight='bold')
    plt.title('proj_delta mean value of AUT questions', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 設置 x 軸刻度
    plt.xticks(layers_list, rotation=45)
    
    # 調整佈局
    plt.tight_layout()
    
    # 顯示圖表
    plt.show()
    
    # 儲存圖表
    plt.savefig('/workspace/LLM_Persona/persona_vectors/persona_vectors/projection_data/projection_layers_analysis.png', 
                dpi=300, bbox_inches='tight')
    
    return results

if __name__ == "__main__":
    # csv_path = "/workspace/LLM_Persona/persona_vectors/persona_vectors/projection_data/neutral_projection.csv"
    csv_path = "/workspace/LLM_Persona/persona_vectors/persona_vectors/projection_data/AUT_projection.csv"
    results = analyze_projection_by_layer(csv_path)
    print(f"\n圖表已儲存至: /workspace/LLM_Persona/persona_vectors/persona_vectors/projection_data/projection_layers_analysis.png")