import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import os

# 設定 matplotlib 使用英文
plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

def ensure_directory_exists(directory_path):
    """確保目錄存在，如果不存在則建立"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"📁 建立目錄: {directory_path}")

class VectorAnalyzer:
    """向量分析器"""
    
    def __init__(self, vector_path: str):
        self.vector_path = vector_path
        self.vector = torch.load(vector_path, weights_only=False)
        print(f"📊 載入向量: {vector_path}")
        print(f"📏 向量形狀: {self.vector.shape}")
    
    def basic_statistics(self):
        """基本統計資訊"""
        print(f"\n📈 基本統計:")
        print(f"   總元素數: {self.vector.numel():,}")
        print(f"   平均值: {self.vector.mean().item():.6f}")
        print(f"   標準差: {self.vector.std().item():.6f}")
        print(f"   最小值: {self.vector.min().item():.6f}")
        print(f"   最大值: {self.vector.max().item():.6f}")
        print(f"   非零元素: {(self.vector != 0).sum().item():,}")
        print(f"   非零比例: {(self.vector != 0).float().mean().item():.2%}")
        
        return {
            'mean': self.vector.mean().item(),
            'std': self.vector.std().item(),
            'min': self.vector.min().item(),
            'max': self.vector.max().item(),
            'non_zero_ratio': (self.vector != 0).float().mean().item()
        }
    
    def plot_layer_statistics(self, save_path: str = None):
        """繪製每層的統計資訊"""
        num_layers = self.vector.shape[0]
        
        # 計算每層的統計資訊
        layer_stats = []
        for i in range(num_layers):
            layer_vector = self.vector[i]
            stats = {
                'layer': i,
                'mean': layer_vector.mean().item(),
                'std': layer_vector.std().item(),
                'l2_norm': torch.norm(layer_vector, p=2).item(),
                'max_abs': layer_vector.abs().max().item()
            }
            layer_stats.append(stats)
        
        df = pd.DataFrame(layer_stats)
        
        # 建立視覺化
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Layer-wise Vector Statistics', fontsize=16)
        
        # 平均值
        axes[0, 0].plot(df['layer'], df['mean'], 'b-o', markersize=4)
        axes[0, 0].set_title('Mean Value per Layer')
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_ylabel('Mean Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 標準差
        axes[0, 1].plot(df['layer'], df['std'], 'r-o', markersize=4)
        axes[0, 1].set_title('Standard Deviation per Layer')
        axes[0, 1].set_xlabel('Layer Index')
        axes[0, 1].set_ylabel('Standard Deviation')
        axes[0, 1].grid(True, alpha=0.3)
        
        # L2 範數
        axes[1, 0].plot(df['layer'], df['l2_norm'], 'g-o', markersize=4)
        axes[1, 0].set_title('L2 Norm per Layer')
        axes[1, 0].set_xlabel('Layer Index')
        axes[1, 0].set_ylabel('L2 Norm')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 最大絕對值
        axes[1, 1].plot(df['layer'], df['max_abs'], 'purple', marker='o', markersize=4)
        axes[1, 1].set_title('Max Absolute Value per Layer')
        axes[1, 1].set_xlabel('Layer Index')
        axes[1, 1].set_ylabel('Max Absolute Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            # 確保 fig 目錄存在
            fig_dir = os.path.join(os.path.dirname(save_path), 'fig')
            ensure_directory_exists(fig_dir)
            
            # 修改儲存路徑到 fig 資料夾
            filename = os.path.basename(save_path)
            fig_save_path = os.path.join(fig_dir, filename)
            plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
            print(f"💾 圖表已儲存: {fig_save_path}")
        
        plt.show()
        
        return df
    
    def plot_value_distribution(self, layer_idx: int = 20, save_path: str = None):
        """繪製特定層的數值分佈"""
        if layer_idx >= self.vector.shape[0]:
            layer_idx = self.vector.shape[0] - 1
        
        layer_vector = self.vector[layer_idx].flatten().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f'Layer {layer_idx} Value Distribution', fontsize=14)
        
        # 直方圖
        axes[0].hist(layer_vector, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_title('Value Distribution Histogram')
        axes[0].set_xlabel('Vector Values')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # 箱線圖
        axes[1].boxplot(layer_vector, vert=True)
        axes[1].set_title('Value Distribution Boxplot')
        axes[1].set_ylabel('Vector Values')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            # 確保 fig 目錄存在
            fig_dir = os.path.join(os.path.dirname(save_path), 'fig')
            ensure_directory_exists(fig_dir)
            
            # 修改儲存路徑到 fig 資料夾
            filename = os.path.basename(save_path)
            fig_save_path = os.path.join(fig_dir, filename)
            plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
            print(f"💾 圖表已儲存: {fig_save_path}")
        
        plt.show()
    
    def plot_heatmap(self, layer_range: tuple = (15, 25), save_path: str = None):
        """繪製向量熱力圖"""
        start_layer, end_layer = layer_range
        end_layer = min(end_layer, self.vector.shape[0])
        
        # 選擇部分層和維度來展示
        selected_layers = self.vector[start_layer:end_layer]
        
        # 如果維度太大，採樣部分維度
        if selected_layers.shape[1] > 100:
            sample_dims = torch.randperm(selected_layers.shape[1])[:100]
            selected_layers = selected_layers[:, sample_dims]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            selected_layers.numpy(), 
            cmap='RdBu_r', 
            center=0,
            xticklabels=False,
            yticklabels=[f'Layer {i}' for i in range(start_layer, end_layer)]
        )
        plt.title(f'Vector Heatmap (Layers {start_layer}-{end_layer-1})')
        plt.xlabel('Vector Dimensions')
        plt.ylabel('Model Layers')
        
        if save_path:
            # 確保 fig 目錄存在
            fig_dir = os.path.join(os.path.dirname(save_path), 'fig')
            ensure_directory_exists(fig_dir)
            
            # 修改儲存路徑到 fig 資料夾
            filename = os.path.basename(save_path)
            fig_save_path = os.path.join(fig_dir, filename)
            plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
            print(f"💾 圖表已儲存: {fig_save_path}")
        
        plt.show()

class MultiVectorComparator:
    """多向量比較器"""
    
    def __init__(self, vector_dir: str):
        self.vector_dir = vector_dir
        self.vectors = {}
        self.load_all_vectors()
    
    def load_all_vectors(self):
        """載入所有向量"""
        if not os.path.exists(self.vector_dir):
            print(f"❌ 向量目錄不存在: {self.vector_dir}")
            return
        
        for filename in os.listdir(self.vector_dir):
            if filename.endswith("_response_avg_diff.pt"):
                role_name = filename.replace("_response_avg_diff.pt", "")
                vector_path = os.path.join(self.vector_dir, filename)
                
                try:
                    self.vectors[role_name] = torch.load(vector_path, weights_only=False)
                    print(f"📊 載入向量: {role_name} - {self.vectors[role_name].shape}")
                except Exception as e:
                    print(f"❌ 載入 {role_name} 向量失敗: {e}")
    
    def compare_statistics(self):
        """比較所有向量的統計資訊"""
        if not self.vectors:
            print("⚠️  沒有向量可以比較")
            return None
        
        comparison_data = []
        
        for role_name, vector in self.vectors.items():
            stats = {
                'role': role_name,
                'mean': vector.mean().item(),
                'std': vector.std().item(),
                'l2_norm': torch.norm(vector, p=2).item(),
                'max_abs': vector.abs().max().item(),
                'non_zero_ratio': (vector != 0).float().mean().item()
            }
            comparison_data.append(stats)
        
        df = pd.DataFrame(comparison_data)
        
        # 視覺化比較
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Multi-Role Vector Statistics Comparison', fontsize=16)
        
        metrics = ['mean', 'std', 'l2_norm', 'max_abs', 'non_zero_ratio']
        metric_names = ['Mean Value', 'Standard Deviation', 'L2 Norm', 'Max Absolute', 'Non-zero Ratio']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            row = i // 3
            col = i % 3
            
            bars = axes[row, col].bar(df['role'], df[metric])
            axes[row, col].set_title(name)
            axes[row, col].tick_params(axis='x', rotation=45)
            
            # 為每個條形添加數值標籤
            for bar in bars:
                height = bar.get_height()
                axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.4f}', ha='center', va='bottom')
        
        # 隱藏空白子圖
        axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        
        # 儲存圖表到 fig 資料夾
        fig_dir = os.path.join(self.vector_dir, 'fig')
        ensure_directory_exists(fig_dir)
        fig_save_path = os.path.join(fig_dir, 'multi_role_statistics_comparison.png')
        plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
        print(f"� 圖表已儲存: {fig_save_path}")
        
        print("�📊 顯示多角色統計比較圖表...")
        plt.show()
        
        return df
    
    def compute_similarity_matrix(self, layer_idx: int = 20):
        """計算向量相似度矩陣"""
        if not self.vectors:
            print("⚠️  沒有向量可以比較")
            return None
        
        role_names = list(self.vectors.keys())
        
        # 如果只有一個向量，建立自相似度矩陣
        if len(role_names) == 1:
            similarity_matrix = np.array([[1.0]])
        else:
            similarity_matrix = np.zeros((len(role_names), len(role_names)))
            
            for i, role1 in enumerate(role_names):
                for j, role2 in enumerate(role_names):
                    vec1 = self.vectors[role1][layer_idx].flatten()
                    vec2 = self.vectors[role2][layer_idx].flatten()
                    
                    # 計算餘弦相似度
                    similarity = torch.cosine_similarity(
                        vec1.unsqueeze(0), vec2.unsqueeze(0)
                    ).item()
                    similarity_matrix[i, j] = similarity
        
        # 繪製相似度熱力圖
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt='.3f',
            xticklabels=role_names,
            yticklabels=role_names,
            cmap='coolwarm',
            center=0
        )
        plt.title(f'Layer {layer_idx} Vector Similarity Matrix')
        plt.tight_layout()
        
        # 儲存圖表到 fig 資料夾
        fig_dir = os.path.join(self.vector_dir, 'fig')
        ensure_directory_exists(fig_dir)
        fig_save_path = os.path.join(fig_dir, f'similarity_matrix_layer_{layer_idx}.png')
        plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
        print(f"� 圖表已儲存: {fig_save_path}")
        
        print("�🔍 顯示向量相似度矩陣...")
        plt.show()
        
        return similarity_matrix, role_names
    
    def pca_visualization(self, layer_idx: int = 20):
        """使用 PCA 視覺化向量"""
        if len(self.vectors) < 2:
            print("⚠️  需要至少 2 個向量來進行 PCA 分析")
            return None
        
        # 準備資料
        vectors_list = []
        labels = []
        
        for role_name, vector in self.vectors.items():
            vectors_list.append(vector[layer_idx].flatten().numpy())
            labels.append(role_name)
        
        vectors_array = np.stack(vectors_list)
        
        # 執行 PCA
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors_array)
        
        # 繪製 PCA 結果
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        for i, (label, color) in enumerate(zip(labels, colors)):
            plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1], 
                       s=200, alpha=0.8, color=color, label=label)
            plt.annotate(label, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=12)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title(f'Layer {layer_idx} Vector PCA Visualization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 儲存圖表到 fig 資料夾
        fig_dir = os.path.join(self.vector_dir, 'fig')
        ensure_directory_exists(fig_dir)
        fig_save_path = os.path.join(fig_dir, f'pca_visualization_layer_{layer_idx}.png')
        plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
        print(f"💾 圖表已儲存: {fig_save_path}")
        
        print("🎯 顯示 PCA 視覺化圖表...")
        plt.show()
        
        return vectors_2d, pca

def analyze_vector(vector_path: str):
    """完整分析單一向量"""
    print(f"🔍 分析向量: {vector_path}")
    
    analyzer = VectorAnalyzer(vector_path)
    
    # 基本統計
    stats = analyzer.basic_statistics()

    print("\n📊 產生圖表 1: 每層統計資訊...")
    # 繪製層統計
    layer_df = analyzer.plot_layer_statistics(
        save_path=vector_path.replace('.pt', '_layer_stats.png')
    )
    
    print("\n📊 產生圖表 2: 數值分佈...")
    # 繪製數值分佈
    analyzer.plot_value_distribution(
        layer_idx=20,
        save_path=vector_path.replace('.pt', '_distribution.png')
    )
    
    print("\n📊 產生圖表 3: 熱力圖...")
    # 繪製熱力圖
    analyzer.plot_heatmap(
        save_path=vector_path.replace('.pt', '_heatmap.png')
    )
    
    return analyzer, stats, layer_df

def quick_vector_check(vector_path: str):
    """快速檢查向量品質"""
    try:
        # 載入向量
        vector = torch.load(vector_path, weights_only=False)
        print(f"✅ 成功載入向量: {vector_path}")
        print(f"📏 形狀: {vector.shape}")
        
        # 基本統計
        print(f"\n📊 基本統計:")
        print(f"   平均值: {vector.mean().item():.6f}")
        print(f"   標準差: {vector.std().item():.6f}")
        print(f"   範圍: [{vector.min().item():.6f}, {vector.max().item():.6f}]")
        print(f"   非零比例: {(vector != 0).float().mean().item():.2%}")
        
        # 檢查是否為零向量
        if torch.allclose(vector, torch.zeros_like(vector), atol=1e-6):
            print("⚠️  警告: 這似乎是零向量！")
            return False
        else:
            print("✅ 向量包含有意義的數值")
            return True
            
    except Exception as e:
        print(f"❌ 載入向量失敗: {e}")
        return False

def compare_all_vectors():
    """比較所有向量"""
    vector_dir = "persona_vectors/Qwen2.5-7B-Instruct/multi_role/"
    
    comparator = MultiVectorComparator(vector_dir)
    
    print(f"\n📊 找到 {len(comparator.vectors)} 個向量")
    
    if len(comparator.vectors) == 0:
        print("⚠️  沒有找到任何向量檔案")
        return None, None
    
    print("\n📊 產生圖表 4: 統計比較...")
    # 統計比較
    stats_df = comparator.compare_statistics()
    print(f"📋 統計比較結果:\n{stats_df}")
    
    # 儲存 CSV 到 csv 資料夾
    csv_dir = os.path.join(vector_dir, 'csv')
    ensure_directory_exists(csv_dir)
    csv_save_path = os.path.join(csv_dir, "comparison_stats.csv")
    stats_df.to_csv(csv_save_path, index=False)
    print(f"💾 統計比較結果已儲存: {csv_save_path}")

    print("\n📊 產生圖表 5: 相似度矩陣...")
    # 相似度矩陣
    sim_matrix, role_names = comparator.compute_similarity_matrix(layer_idx=20)
    if sim_matrix is not None:
        similarity_df = pd.DataFrame(sim_matrix, index=role_names, columns=role_names)
        print(f"📋 相似度矩陣:\n{similarity_df}")
        
        # 儲存 CSV 到 csv 資料夾
        csv_save_path = os.path.join(csv_dir, "similarity_matrix.csv")
        similarity_df.to_csv(csv_save_path)
        print(f"💾 相似度矩陣已儲存: {csv_save_path}")
    
    # PCA 視覺化（如果有多個向量）
    if len(comparator.vectors) >= 2:
        print("\n📊 產生圖表 6: PCA 視覺化...")
        pca_result = comparator.pca_visualization(layer_idx=20)
        if pca_result is not None:
            vectors_2d, pca = pca_result
            pca_df = pd.DataFrame(vectors_2d, index=role_names, columns=['PC1', 'PC2'])
            print(f"📋 PCA 結果 (前兩個主成分):\n{pca_df}")
            
            # 儲存 CSV 到 csv 資料夾
            csv_save_path = os.path.join(csv_dir, "pca_results.csv")
            pca_df.to_csv(csv_save_path)
            print(f"💾 PCA 結果已儲存: {csv_save_path}")
    
    return comparator, stats_df

if __name__ == "__main__":
    print("🎯 開始向量分析流程...")
    
    # 分析單一向量
    vector_path = "persona_vectors/Qwen2.5-7B-Instruct/multi_role/empathetic_counselor_response_avg_diff.pt"
    
    if os.path.exists(vector_path):
        print("🔍 步驟 1: 分析單一向量...")
        
        # 快速檢查
        is_valid = quick_vector_check(vector_path)
        
        if is_valid:
            # 詳細分析 (3個圖表)
            analyzer, stats, layer_df = analyze_vector(vector_path)
        else:
            print("⚠️  向量品質有問題，跳過詳細分析")
    else:
        print(f"❌ 找不到向量檔案: {vector_path}")
    
    # 比較多個向量
    print("\n🔍 步驟 2: 比較多個向量...")
    comparator, stats_df = compare_all_vectors()
    
    print("\n✅ 分析完成！您應該看到總共 5-6 個圖表")
    print("📊 圖表說明:")
    print("   1. Layer-wise Vector Statistics (2x2 網格)")
    print("   2. Layer 20 Value Distribution (1x2 網格)")  
    print("   3. Vector Heatmap")
    print("   4. Multi-Role Vector Statistics Comparison (2x3 網格)")
    print("   5. Layer 20 Vector Similarity Matrix")
    if comparator and len(comparator.vectors) >= 2:
        print("   6. Layer 20 Vector PCA Visualization")
    
    print("\n📁 檔案儲存位置:")
    print("   🖼️  圖表: persona_vectors/Qwen2.5-7B-Instruct/multi_role/fig/")
    print("   📊 CSV: persona_vectors/Qwen2.5-7B-Instruct/multi_role/csv/")