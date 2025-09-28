import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich import print
import csv
import os
from datetime import datetime

# --- 輔助函數 ---
def a_proj_b(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """計算向量 a 在向量 b 上的純量投影。"""
    if a.dtype != b.dtype:
        a = a.to(b.dtype)
    
    b_norm = b.norm()
    return torch.tensor(0.0, dtype=b.dtype, device=b.device) if b_norm == 0 else torch.dot(a, b) / b_norm

# --- 設定參數 ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
VECTOR_A_PATH = "Llama-3.1-8B-Instruct/multi_role/creative_professional_response_avg_diff.pt"
VECTOR_B_PATH = "Llama-3.1-8B-Instruct/multi_role/environmentalist_response_avg_diff.pt"

STEERING_LAYER = 20
STEERING_COEFFICIENT = 2.0
ANALYSIS_LAYERS = list(range(20, 33))  # 20-32

# --- 10 個測試題目 ---
AUT_LIST = [
    "What are some creative use for Fork? The goal is to come up with creative ideas, which are ideas that strike people as clever, unusual, interesting, uncommon, humorous, innovative, or different. Present a list of 5 creative and diverse uses for Fork.",
    "What are some creative use for Jar? The goal is to come up with creative ideas, which are ideas that strike people as clever, unusual, interesting, uncommon, humorous, innovative, or different. Present a list of 5 creative and diverse uses for Jar."
]

INS_LIST = [
    "Name all the round things you can think of.",
    "Name all the things you can think of that will make a noise.",
    "Name all the things you can think of that have a screen."
]

SIMI_LIST = [
    "Tell me all the ways in which a kite and a balloon are alike.",
    "Tell me all the ways in which a pencil and a pen are alike.",
    "Tell me all the ways in which a chair and a couch are alike."
]

SCI_LIST = [
    "If you can take a spaceship to travel in outer space and go to a planet, what scientific questions do you want to research? For example, are there any living things on the planet?",
    "Please think up as many possible improvements as you can to a regular bicycle, making it more interesting, more useful and more beautiful. For example, make the tires reflective, so they can be seen in the dark."
]

NEUTRAL_LIST = [
    "Urban Planning (2050 City Block Masterplan): Design a masterplan for a new city block to be built in 2050. Describe core principles, layout, mobility, public space, services, and governance constraints.",
    "Product Launch (Micro-Teleportation for Small Objects): Outline a public launch plan for a micro-teleportation technology for small items. Include positioning, safety/regulation, go-to-market, operations, and risk.",
    "Social Issue (Countering Misinformation): Propose a multi-pronged plan to reduce misinformation on social platforms: policy, product, incentives, literacy, measurement.",
    "Corporate Strategy (Legacy Manufacturer vs. AI Disruption): Design a transformation strategy for a legacy manufacturer facing AI disruption: portfolio, org, tech stack, talent, risk, timeline.",
    "Healthcare Innovation (Reimagine the Hospital): Redesign the future hospital experience for patients, families, and staff. Address flows, safety, data, wellbeing, equity, and feasibility.",
    "Education Reform (Ideal High-School Curriculum): Propose a 4-year curriculum: core subjects, skills, experiential learning, assessment, inclusion, and teacher enablement.",
    "Disaster Response (Early Recovery Plan for a Metro Area): Draft an initial 30–60 day recovery plan after a major natural disaster: assessment, triage, logistics, comms, governance, equity.",
    "Space Exploration (Next 50 Years Priority): State and justify the top priority for human space exploration in the next 50 years. Define milestones, risks, ethics, and spillovers.",
    "Sustainable Fashion (Net-Zero Brand Model): Propose a business model for a fully sustainable fashion brand: materials, supply chain, circularity, economics, verification, storytelling.",
    "Global Challenge (Food Waste Reduction): Design a multi-layer plan to reduce global food waste across production, retail, and households: incentives, infra, tech, policy, culture."
]

# PROMPT_LIST = AUT_LIST + INS_LIST + SIMI_LIST + SCI_LIST
PROMPT_LIST = NEUTRAL_LIST

def process_single_prompt_complete_analysis(prompt, prompt_idx, model, tokenizer, steering_layer, 
                                         analysis_layers, vector_a_steer, vector_b_steer, 
                                         vectors_a_all, vectors_b_all, steering_coefficient, device):
    """處理單個題目的完整分析：一次基準 + 一次引導，分析所有層"""
    
    print(f"\n處理題目 {prompt_idx+1}/{len(PROMPT_LIST)}")
    print(f"題目: {prompt[:80]}...")
    
    # 準備輸入
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]
    
    print("  🔄 執行基準測試（不加引導）...")
    # --- 1. 基準測試：蒐集所有分析層的基準 activation ---
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

    print("  🎯 執行引導測試（第20層加引導）...")
    # --- 2. 引導測試：在第20層加引導，蒐集所有分析層的引導後 activation ---
    steering_vec = (vector_a_steer + vector_b_steer) * steering_coefficient

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

    print("  📊 計算各層投影分析...")
    # --- 3. 對每個分析層進行投影分析 ---
    results = []
    baseline_text = tokenizer.decode(outputs_baseline.sequences[0][prompt_len:])
    steered_text = tokenizer.decode(outputs_steered.sequences[0][prompt_len:])
    
    for analysis_layer in analysis_layers:
        # 取得該層的投影向量
        vector_a_proj = vectors_a_all[analysis_layer]
        vector_b_proj = vectors_b_all[analysis_layer]
        
        # 取得該層的 activation
        activation_baseline = baseline_activations[analysis_layer]
        activation_steered = steered_activations[analysis_layer]
        
        # 5a. 絕對投影分析
        proj_baseline_on_a = a_proj_b(activation_baseline, vector_a_proj)
        proj_baseline_on_b = a_proj_b(activation_baseline, vector_b_proj)
        proj_steered_on_a = a_proj_b(activation_steered, vector_a_proj)
        proj_steered_on_b = a_proj_b(activation_steered, vector_b_proj)

        # 5b. 差值投影分析
        activation_delta = activation_steered - activation_baseline
        proj_delta_on_a = a_proj_b(activation_delta, vector_a_proj)
        proj_delta_on_b = a_proj_b(activation_delta, vector_b_proj)
        
        result = {
            'prompt_idx': prompt_idx,
            'prompt': prompt,
            'steering_layer': steering_layer,
            'analysis_layer': analysis_layer,
            'steering_coefficient': steering_coefficient,
            'proj_baseline_on_a': proj_baseline_on_a.item(),
            'proj_baseline_on_b': proj_baseline_on_b.item(),
            'proj_steered_on_a': proj_steered_on_a.item(),
            'proj_steered_on_b': proj_steered_on_b.item(),
            'proj_delta_on_a': proj_delta_on_a.item(),
            'proj_delta_on_b': proj_delta_on_b.item(),
            'baseline_text': baseline_text,
            'steered_text': steered_text
        }
        results.append(result)
        
        print(f"    層 {analysis_layer}: A基準={proj_baseline_on_a:.3f}, B基準={proj_baseline_on_b:.3f}")
    
    return results

def main():
    print("="*50)
    print("批次引導向量分析實驗")
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
    
    # 載入引導向量（固定在第20層）
    vectors_a_all = torch.load(VECTOR_A_PATH, map_location=device)
    vectors_b_all = torch.load(VECTOR_B_PATH, map_location=device)
    
    vector_a_steer = vectors_a_all[STEERING_LAYER].to(dtype=model_dtype)
    vector_b_steer = vectors_b_all[STEERING_LAYER].to(dtype=model_dtype)
    
    # 正規化引導向量
    vector_a_steer /= vector_a_steer.norm()
    vector_b_steer /= vector_b_steer.norm()
    
    # 準備所有分析層的投影向量並正規化
    for layer in ANALYSIS_LAYERS:
        vectors_a_all[layer] = vectors_a_all[layer].to(dtype=model_dtype)
        vectors_b_all[layer] = vectors_b_all[layer].to(dtype=model_dtype)
        vectors_a_all[layer] /= vectors_a_all[layer].norm()
        vectors_b_all[layer] /= vectors_b_all[layer].norm()

    # 準備結果儲存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"batch_steering_results_{timestamp}.csv"
    detailed_results_file = f"detailed_results_{timestamp}.txt"
    
    all_results = []
    
    # 執行批次實驗
    total_experiments = len(PROMPT_LIST)  # 每個題目只需要處理一次
    
    for prompt_idx, prompt in enumerate(PROMPT_LIST):
        print(f"\n{'='*60}")
        print(f"處理題目 {prompt_idx+1}/{len(PROMPT_LIST)}")
        print(f"題目: {prompt}")
        print(f"{'='*60}")
        
        # 執行單個題目的完整分析（一次基準 + 一次引導，分析所有層）
        prompt_results = process_single_prompt_complete_analysis(
            prompt, prompt_idx, model, tokenizer, 
            STEERING_LAYER, ANALYSIS_LAYERS,
            vector_a_steer, vector_b_steer, 
            vectors_a_all, vectors_b_all,
            STEERING_COEFFICIENT, device
        )
        
        all_results.extend(prompt_results)
        
        # 即時顯示結果摘要
        print(f"\n  ✅ 完成題目 {prompt_idx+1}，產生 {len(prompt_results)} 筆分析結果")

    # 儲存結果到 CSV
    print(f"\n儲存結果到 {results_file}")
    with open(results_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'prompt_idx', 'analysis_layer', 'steering_layer', 'steering_coefficient',
            'proj_baseline_on_a', 'proj_baseline_on_b', 
            'proj_steered_on_a', 'proj_steered_on_b',
            'proj_delta_on_a', 'proj_delta_on_b'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in all_results:
            row = {k: v for k, v in result.items() if k in fieldnames}
            writer.writerow(row)

    # 儲存詳細結果（包含文本）
    print(f"儲存詳細結果到 {detailed_results_file}")
    with open(detailed_results_file, 'w', encoding='utf-8') as f:
        f.write("批次引導向量分析詳細結果\n")
        f.write("="*60 + "\n\n")
        
        for result in all_results:
            f.write(f"題目 {result['prompt_idx']+1}, 分析層 {result['analysis_layer']}\n")
            f.write(f"Prompt: {result['prompt']}\n")
            f.write(f"引導層: {result['steering_layer']}, 分析層: {result['analysis_layer']}, 引導係數: {result['steering_coefficient']}\n\n")
            
            f.write("--- 5a. 絕對投影分析 (測量最終『思維狀態』) ---\n")
            f.write(f"基準投影: A={result['proj_baseline_on_a']:.4f}, B={result['proj_baseline_on_b']:.4f}\n")
            f.write(f"引導後投影: A={result['proj_steered_on_a']:.4f}, B={result['proj_steered_on_b']:.4f}\n\n")
            
            f.write("--- 5b. 差值投影分析 (測量引導向量的『因果效應』) ---\n")
            f.write(f"激活變化量在 A 上的投影: {result['proj_delta_on_a']:.4f}\n")
            f.write(f"激活變化量在 B 上的投影: {result['proj_delta_on_b']:.4f}\n\n")
            
            f.write("--- 生成文本對比 ---\n")
            f.write(f"【基準】: {result['baseline_text']}\n")
            f.write(f"【引導後】: {result['steered_text']}\n")
            f.write("\n" + "="*60 + "\n\n")

    print(f"\n✅ 實驗完成！")
    print(f"📊 CSV 結果: {results_file}")
    print(f"📝 詳細結果: {detailed_results_file}")
    print(f"🔢 總實驗次數: {len(all_results)}")
    print(f"📋 題目數量: {len(PROMPT_LIST)}")
    print(f"🎯 分析層數量: {len(ANALYSIS_LAYERS)}")

if __name__ == "__main__":
    main()