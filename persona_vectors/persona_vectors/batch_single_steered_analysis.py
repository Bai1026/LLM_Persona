import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich import print
import csv
import os
from datetime import datetime

# --- 輔助函式 ---
def a_proj_b(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """計算向量 a 在向量 b 上的純量投影。"""
    if a.dtype != b.dtype:
        a = a.to(b.dtype)
    
    b_norm = b.norm()
    return torch.tensor(0.0, dtype=b.dtype, device=b.device) if b_norm == 0 else torch.dot(a, b) / b_norm

# --- 設定參數 ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
VECTOR_A_PATH = "Llama-3.1-8B-Instruct/multi_role/creative_professional_response_avg_diff.pt"

STEERING_LAYER = 20  # 引導層
STEERING_COEFFICIENT = 2.0
ANALYSIS_LAYERS = list(range(20, 33))  # 20-32 觀察層

# --- 測試題目 ---
PROMPT_LIST = [
    "What are some creative use for Fork? The goal is to come up with creative ideas, which are ideas that strike people as clever, unusual, interesting, uncommon, humorous, innovative, or different. Present a list of 5 creative and diverse uses for Fork.",
    "Urban Planning (2050 City Block Masterplan): Design a masterplan for a new city block to be built in 2050. Describe core principles, layout, mobility, public space, services, and governance constraints.",
    "Product Launch (Micro-Teleportation for Small Objects): Outline a public launch plan for a micro-teleportation technology for small items. Include positioning, safety/regulation, go-to-market, operations, and risk."
]

def process_single_vector_steered_analysis(prompt, prompt_idx, model, tokenizer, steering_layer, 
                                         analysis_layers, steering_vector, 
                                         projection_vectors_all, steering_coefficient, device):
    """處理單個題目的單一向量引導分析"""
    
    print(f"\n處理題目 {prompt_idx+1}/{len(PROMPT_LIST)}")
    print(f"題目: {prompt[:80]}...")
    
    # 準備輸入
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]
    
    print("  🔄 執行基準測試（不加引導）...")
    # --- 1. 基準測試 ---
    with torch.no_grad():
        outputs_baseline = model.generate(
            **inputs, 
            max_new_tokens=512, 
            output_hidden_states=True, 
            return_dict_in_generate=True
        )

    # 提取所有分析層的基準 activation
    baseline_activations = {}
    for analysis_layer in analysis_layers:
        generated_hidden_states = []
        for token_idx, token_hidden_states in enumerate(outputs_baseline.hidden_states):
            if token_idx >= prompt_len:
                layer_activation = token_hidden_states[analysis_layer]
                generated_hidden_states.append(layer_activation[:, -1, :])
        
        if generated_hidden_states:
            baseline_activations[analysis_layer] = torch.stack(generated_hidden_states).mean(dim=0).squeeze()
        else:
            baseline_activations[analysis_layer] = outputs_baseline.hidden_states[0][analysis_layer][:, -1, :].squeeze()

    print(f"  🎯 執行單一向量引導測試（第{steering_layer}層加引導）...")
    # --- 2. 單一向量引導測試 ---
    steering_vec = steering_vector * steering_coefficient

    def steering_hook(module, args, kwargs, output):
        modified_output = output[0] + steering_vec
        return (modified_output,) + output[1:]

    hook_handle = model.model.layers[steering_layer].register_forward_hook(steering_hook, with_kwargs=True)
    
    try:
        with torch.no_grad():
            outputs_steered = model.generate(
                **inputs, 
                max_new_tokens=512, 
                output_hidden_states=True, 
                return_dict_in_generate=True
            )
    finally:
        hook_handle.remove()

    # 提取所有分析層的引導後 activation
    steered_activations = {}
    for analysis_layer in analysis_layers:
        generated_hidden_states = []
        for token_idx, token_hidden_states in enumerate(outputs_steered.hidden_states):
            if token_idx >= prompt_len:
                layer_activation = token_hidden_states[analysis_layer]
                generated_hidden_states.append(layer_activation[:, -1, :])
        
        if generated_hidden_states:
            steered_activations[analysis_layer] = torch.stack(generated_hidden_states).mean(dim=0).squeeze()
        else:
            steered_activations[analysis_layer] = outputs_steered.hidden_states[0][analysis_layer][:, -1, :].squeeze()

    print("  📊 計算各層對引導向量的投影分析...")
    # --- 3. 對每個分析層進行投影分析 ---
    results = []
    baseline_text = tokenizer.decode(outputs_baseline.sequences[0][prompt_len:])
    steered_text = tokenizer.decode(outputs_steered.sequences[0][prompt_len:])
    
    for analysis_layer in analysis_layers:
        # 取得該層的投影向量（用同一個引導向量做投影）
        projection_vector = projection_vectors_all[analysis_layer]
        
        # 取得該層的 activation
        activation_baseline = baseline_activations[analysis_layer]
        activation_steered = steered_activations[analysis_layer]
        
        # 計算投影
        proj_baseline = a_proj_b(activation_baseline, projection_vector)
        proj_steered = a_proj_b(activation_steered, projection_vector)
        
        # 差值分析
        activation_delta = activation_steered - activation_baseline
        proj_delta = a_proj_b(activation_delta, projection_vector)
        
        result = {
            'prompt_idx': prompt_idx,
            'prompt': prompt,
            'steering_layer': steering_layer,
            'analysis_layer': analysis_layer,
            'steering_coefficient': steering_coefficient,
            'proj_baseline': proj_baseline.item(),
            'proj_steered': proj_steered.item(),
            'proj_delta': proj_delta.item(),
            'baseline_text': baseline_text,
            'steered_text': steered_text
        }
        results.append(result)
        
        print(f"    層 {analysis_layer}: 基準={proj_baseline:.3f}, 引導後={proj_steered:.3f}, 變化={proj_delta:.3f}")
    
    return results

def main():
    print("="*50)
    print("單一向量引導分析實驗")
    print("="*50)
    print(f"模型: {MODEL_NAME}")
    print(f"引導層: {STEERING_LAYER}")
    print(f"分析層範圍: {ANALYSIS_LAYERS[0]}-{ANALYSIS_LAYERS[-1]}")
    print(f"引導係數: {STEERING_COEFFICIENT}")
    print(f"題目總數: {len(PROMPT_LIST)}")
    print("="*50)

    # 初始化
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    model_dtype = torch.bfloat16
    
    # 載入向量
    vectors_all = torch.load(VECTOR_A_PATH, map_location=device)
    
    # 準備引導向量（第20層）
    steering_vector = vectors_all[STEERING_LAYER].to(dtype=model_dtype)
    steering_vector = steering_vector / steering_vector.norm()  # 正規化
    
    # 準備所有分析層的投影向量並正規化（用同一個向量）
    projection_vectors_all = {}
    for layer in ANALYSIS_LAYERS:
        # projection_vectors_all[layer] = vectors_all[layer].to(dtype=model_dtype)
        projection_vectors_all[layer] = steering_vector 
        projection_vectors_all[layer] = projection_vectors_all[layer] / projection_vectors_all[layer].norm()

    # 準備結果儲存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"single_vector_steering_results_{timestamp}.csv"
    
    all_results = []
    
    # 執行批次實驗
    for prompt_idx, prompt in enumerate(PROMPT_LIST):
        print(f"\n{'='*60}")
        print(f"處理題目 {prompt_idx+1}/{len(PROMPT_LIST)}")
        print(f"{'='*60}")
        
        prompt_results = process_single_vector_steered_analysis(
            prompt, prompt_idx, model, tokenizer, 
            STEERING_LAYER, ANALYSIS_LAYERS,
            steering_vector, projection_vectors_all,
            STEERING_COEFFICIENT, device
        )
        
        all_results.extend(prompt_results)
        print(f"\n  ✅ 完成題目 {prompt_idx+1}")

    # 儲存結果到 CSV
    print(f"\n儲存結果到 {results_file}")
    with open(results_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'prompt_idx', 'analysis_layer', 'steering_layer', 'steering_coefficient',
            'proj_baseline', 'proj_steered', 'proj_delta'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in all_results:
            row = {k: v for k, v in result.items() if k in fieldnames}
            writer.writerow(row)

    # 顯示結果摘要
    print(f"\n✅ 實驗完成！")
    print(f"📊 結果檔案: {results_file}")
    print(f"🔢 總分析數量: {len(all_results)}")
    
    # 顯示第20層的投影值（應該接近1）
    layer_20_results = [r for r in all_results if r['analysis_layer'] == 20]
    if layer_20_results:
        avg_proj_steered_20 = sum(r['proj_steered'] for r in layer_20_results) / len(layer_20_results)
        print(f"🎯 第20層平均引導後投影: {avg_proj_steered_20:.4f} (預期接近1.0)")

if __name__ == "__main__":
    main()