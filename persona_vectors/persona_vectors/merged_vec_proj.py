import torch
import os
import argparse
from pathlib import Path

def a_proj_b(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    計算向量 a 在向量 b 上的純量投影 (scalar projection)。
    結果是一個純量，代表 a 在 b 方向上的有向長度。
    """
    b_norm = b.norm()
    # 防止除以零的錯誤
    if b_norm == 0:
        return torch.tensor(0.0)
    
    # (a·b) / ||b||
    return torch.dot(a, b) / b_norm

def load_vector_from_pt(vector_path: str, layer: int, device: str = 'cuda') -> torch.Tensor:
    """
    從 .pt 檔案載入指定層數的向量
    
    Args:
        vector_path: .pt 檔案路徑
        layer: 要載入的層數
        device: 運算設備 ('cuda' or 'cpu')
    
    Returns:
        指定層的向量 tensor
    """
    print(f"📂 載入向量檔案: {vector_path}")
    data = torch.load(vector_path, weights_only=False)
    
    layer = layer + 1
    
    # 檢查檔案格式
    if isinstance(data, dict):
        # 字典格式：{layer_num: tensor, ...}
        if layer not in data:
            available_layers = list(data.keys())
            raise ValueError(f"❌ 層數 {layer} 不存在於檔案中。可用層數: {available_layers}")
        vector = data[layer].to(device)
    elif isinstance(data, torch.Tensor):
        # Tensor 格式：[num_layers, hidden_dim]
        if data.dim() != 2:
            raise ValueError(f"❌ 預期 2D tensor [num_layers, hidden_dim]，但得到 {data.shape}")
        
        num_layers, hidden_dim = data.shape
        print(f"🔍 檔案包含 {num_layers} 層，每層維度 {hidden_dim}")
        
        if layer >= num_layers or layer < 0:
            raise ValueError(f"❌ 層數 {layer} 超出範圍。可用層數: 0-{num_layers-1}")
        
        vector = data[layer].to(device)
    else:
        raise ValueError(f"❌ 不支援的檔案格式: {type(data)}")
    
    print(f"✅ 成功載入第 {layer} 層，向量維度: {vector.shape}")
    return vector

def main():
    # --- 參數解析 ---
    parser = argparse.ArgumentParser(description="載入 .pt 檔案並進行任意層數的向量投影運算")
    parser.add_argument("--vector_a_path", type=str, required=True, help="向量 A 的 .pt 檔案路徑")
    parser.add_argument("--vector_b_path", type=str, required=True, help="向量 B 的 .pt 檔案路徑")
    parser.add_argument("--layer_list", type=int, nargs="+", default=[15], help="要計算的層數清單")
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'], help="運算設備")
    parser.add_argument("--simulate_activation", action="store_true", help="是否模擬激活向量進行測試")
    
    args = parser.parse_args()
    
    # 檢查檔案存在性
    if not os.path.exists(args.vector_a_path):
        raise FileNotFoundError(f"❌ 向量 A 檔案不存在: {args.vector_a_path}")
    if not os.path.exists(args.vector_b_path):
        raise FileNotFoundError(f"❌ 向量 B 檔案不存在: {args.vector_b_path}")
    
    # 設定設備
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"🚀 腳本將在 [ {device.upper()} ] 設備上運行")
    print(f"🎯 目標層數: {args.layer_list}\n")
    
    # 對每個層數進行計算
    for layer in args.layer_list:
        print("=" * 80)
        print(f"🔍 計算第 {layer} 層")
        print("=" * 80)
        
        # --- 載入真實向量 ---
        try:
            vector_a = load_vector_from_pt(args.vector_a_path, layer, device)
            vector_b = load_vector_from_pt(args.vector_b_path, layer, device)
        except Exception as e:
            print(f"❌ 載入向量時發生錯誤: {e}")
            continue
        
        # 標準化向量（可選）
        vector_a_normalized = vector_a / vector_a.norm()
        vector_b_normalized = vector_b / vector_b.norm()
        
        # 模擬激活向量（如果需要測試）
        if args.simulate_activation:
            activation_vec = torch.randn_like(vector_a, device=device)
        else:
            # 如果不模擬，可以在這裡加入從其他來源載入激活向量的邏輯
            activation_vec = None

        print("--- 向量基本資訊 ---")
        print(f"向量 A 的長度 (Norm): {vector_a.norm().item():.4f}")
        print(f"向量 B 的長度 (Norm): {vector_b.norm().item():.4f}")
        print(f"向量 A (標準化) 的長度: {vector_a_normalized.norm().item():.4f}")
        print(f"向量 B (標準化) 的長度: {vector_b_normalized.norm().item():.4f}")
        print(f"A 和 B 的餘弦相似度: {torch.dot(vector_a_normalized, vector_b_normalized).item():.4f}\n")

        # --- 核心計算 ---
        print("--- 核心計算結果 ---")

        # (A + B) 的線性組合 (使用標準化向量)
        vector_sum = vector_a_normalized + vector_b_normalized

        # 計算 (A + B) 投影到 A 和 B
        proj_sum_on_a = a_proj_b(vector_sum, vector_a_normalized)
        proj_sum_on_b = a_proj_b(vector_sum, vector_b_normalized)

        print(f"1. (A+B) 投影到 A 上的純量結果: {proj_sum_on_a.item():.4f}")
        print(f"   數學驗證: (A proj A) + (B proj A) = {a_proj_b(vector_a_normalized, vector_a_normalized).item():.2f} + {a_proj_b(vector_b_normalized, vector_a_normalized).item():.4f} = {a_proj_b(vector_a_normalized, vector_a_normalized).item() + a_proj_b(vector_b_normalized, vector_a_normalized).item():.4f}\n")

        print(f"2. (A+B) 投影到 B 上的純量結果: {proj_sum_on_b.item():.4f}")
        print(f"   數學驗證: (A proj B) + (B proj B) = {a_proj_b(vector_a_normalized, vector_b_normalized).item():.4f} + {a_proj_b(vector_b_normalized, vector_b_normalized).item():.2f} = {a_proj_b(vector_a_normalized, vector_b_normalized).item() + a_proj_b(vector_b_normalized, vector_b_normalized).item():.4f}\n")

        # 如果有激活向量，進行投影計算
        if activation_vec is not None:
            # 計算 Activation 投影到各個向量
            proj_activation_on_a = a_proj_b(activation_vec, vector_a_normalized)
            proj_activation_on_b = a_proj_b(activation_vec, vector_b_normalized)
            proj_activation_on_sum = a_proj_b(activation_vec, vector_sum)

            print(f"3. 模型的 Activation 投影到 A: {proj_activation_on_a.item():.4f}")
            print(f"   (這表示模型回應在多大程度上符合『概念A』)\n")

            print(f"4. 模型的 Activation 投影到 B: {proj_activation_on_b.item():.4f}")
            print(f"   (這表示模型回應在多大程度上符合『概念B』)\n")

            print(f"5. 模型的 Activation 投影到 (A+B): {proj_activation_on_sum.item():.4f}")
            print(f"   (這表示模型回應在多大程度上同時符合『概念A』和『概念B』)\n")
        
        # 輸出向量合併的結果
        print("--- 向量合併分析 ---")
        merged_vector = (vector_a_normalized + vector_b_normalized) / 2  # 平均合併
        print(f"平均合併向量 (A+B)/2 的長度: {merged_vector.norm().item():.4f}")
        print(f"合併向量與 A 的餘弦相似度: {torch.dot(merged_vector, vector_a_normalized).item():.4f}")
        print(f"合併向量與 B 的餘弦相似度: {torch.dot(merged_vector, vector_b_normalized).item():.4f}\n")
        
        print(f"✅ 第 {layer} 層計算完成\n")

if __name__ == "__main__":
    main()