import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Helper Function ---
def a_proj_b(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """計算向量 a 在向量 b 上的純量投影。"""
    # 確保兩個向量有相同的 dtype
    if a.dtype != b.dtype:
        a = a.to(b.dtype)
    
    b_norm = b.norm()
    return torch.tensor(0.0, dtype=b.dtype, device=b.device) if b_norm == 0 else torch.dot(a, b) / b_norm

# --- 1. 實驗參數設定 ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct" # 或者您使用的 Llama 模型
VECTOR_A_PATH = "Llama-3.1-8B-Instruct/multi_role/creative_professional_response_avg_diff.pt"
VECTOR_B_PATH = "Llama-3.1-8B-Instruct/multi_role/environmentalist_response_avg_diff.pt"

# 圖層設定
STEERING_LAYER = 20
ANALYSIS_LAYER = 32

# 引導強度係數 (非常重要，需要調整以獲得最佳效果)
STEERING_COEFFICIENT = 2.0

# 測試用的提示
PROMPT = "Human: What are some innovative solutions for urban sustainability? Assistant:"

# --- 2. 載入模型與向量 ---
print("載入模型與分詞器...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
model.eval()

print("載入概念向量...")
# 確保與模型使用相同的 dtype
model_dtype = torch.bfloat16

# Steering vectors from the steering layer
vectors_L_steer = torch.load(VECTOR_A_PATH, map_location=device)
vector_a_steer = vectors_L_steer[STEERING_LAYER].to(dtype=model_dtype)
vector_b_steer = torch.load(VECTOR_B_PATH, map_location=device)[STEERING_LAYER].to(dtype=model_dtype)

# Projection vectors from the analysis layer
vector_a_proj = vectors_L_steer[ANALYSIS_LAYER].to(dtype=model_dtype) # 可以從同一個檔案讀取
vector_b_proj = torch.load(VECTOR_B_PATH, map_location=device)[ANALYSIS_LAYER].to(dtype=model_dtype)

# 標準化以只關注方向
vector_a_steer /= vector_a_steer.norm()
vector_b_steer /= vector_b_steer.norm()
vector_a_proj /= vector_a_proj.norm()
vector_b_proj /= vector_b_proj.norm()

# --- 3. 基準測試 (無引導) ---
print("\n--- 執行基準測試 (無引導) ---")
inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
with torch.no_grad():
    outputs_baseline = model.generate(
        **inputs, 
        max_new_tokens=512, 
        output_hidden_states=True, 
        return_dict_in_generate=True
    )

# 獲取所有生成 token 的 hidden states，並在層和 token 維度上取平均
# 結構: outputs.hidden_states[token_idx][layer_idx]
# 我們對所有新生成的 token 在第 ANALYSIS_LAYER 的激活取平均
prompt_len = inputs.input_ids.shape[1]

# 只取新生成的 token 的 hidden states (跳過 prompt tokens)
generated_hidden_states = []
for token_idx, token_hidden_states in enumerate(outputs_baseline.hidden_states):
    if token_idx >= prompt_len:  # 只考慮新生成的 tokens
        # token_hidden_states 是 tuple，每個元素對應一層
        # 我們只需要 ANALYSIS_LAYER 層的激活
        layer_activation = token_hidden_states[ANALYSIS_LAYER]  # shape: [batch_size, seq_len, hidden_dim]
        # 取最後一個 token 的激活 (剛生成的 token)
        generated_hidden_states.append(layer_activation[:, -1, :])  # shape: [batch_size, hidden_dim]

if generated_hidden_states:
    # 將所有生成 token 的激活平均
    activation_L_analysis_baseline = torch.stack(generated_hidden_states).mean(dim=0).squeeze()  # shape: [hidden_dim]
else:
    # 如果沒有生成任何 token，使用 prompt 的最後一個 token
    activation_L_analysis_baseline = outputs_baseline.hidden_states[0][ANALYSIS_LAYER][:, -1, :].squeeze()

proj_baseline_on_a = a_proj_b(activation_L_analysis_baseline, vector_a_proj)
proj_baseline_on_b = a_proj_b(activation_L_analysis_baseline, vector_b_proj)
print(f"基準投影 (L{ANALYSIS_LAYER}): A={proj_baseline_on_a.item():.4f}, B={proj_baseline_on_b.item():.4f}")


# --- 4. 引導測試 ---
print(f"\n--- 執行引導測試 (在 L{STEERING_LAYER} 層施加 A+B 引導) ---")

# 定義引導向量
steering_vec = (vector_a_steer + vector_b_steer) * STEERING_COEFFICIENT

# 定義 Forward Hook 函數
def steering_hook(module, args, kwargs, output):
    # output[0] 是 hidden_states tensor
    modified_output = output[0] + steering_vec
    return (modified_output,) + output[1:]

# 將 Hook 掛載到目標圖層 (注意 python 的 0-indexed)
hook_handle = model.model.layers[STEERING_LAYER].register_forward_hook(steering_hook, with_kwargs=True)
print(f"✅ Forward Hook 已成功掛載到第 {STEERING_LAYER} 層。")

try:
    with torch.no_grad():
        outputs_steered = model.generate(
            **inputs, 
            max_new_tokens=512, 
            output_hidden_states=True, 
            return_dict_in_generate=True
        )
finally:
    # 確保無論是否出錯，都移除 Hook
    hook_handle.remove()
    print("✅ Forward Hook 已被移除。")

# 提取並分析被引導後的激活值
generated_hidden_states_steered = []
for token_idx, token_hidden_states in enumerate(outputs_steered.hidden_states):
    if token_idx >= prompt_len:  # 只考慮新生成的 tokens
        layer_activation = token_hidden_states[ANALYSIS_LAYER]  # shape: [batch_size, seq_len, hidden_dim]
        generated_hidden_states_steered.append(layer_activation[:, -1, :])  # shape: [batch_size, hidden_dim]

if generated_hidden_states_steered:
    activation_L_analysis_steered = torch.stack(generated_hidden_states_steered).mean(dim=0).squeeze()
else:
    activation_L_analysis_steered = outputs_steered.hidden_states[0][ANALYSIS_LAYER][:, -1, :].squeeze()


proj_steered_on_a = a_proj_b(activation_L_analysis_steered, vector_a_proj)
proj_steered_on_b = a_proj_b(activation_L_analysis_steered, vector_b_proj)
print(f"引導後投影 (L{ANALYSIS_LAYER}): A={proj_steered_on_a.item():.4f}, B={proj_steered_on_b.item():.4f}")


# --- 5. 結果分析 ---
print("\n--- 實驗結果分析 ---")
print(f"引導層: {STEERING_LAYER}, 分析層: {ANALYSIS_LAYER}, 引導係數: {STEERING_COEFFICIENT}")
print(f"對 A 的投影變化: {proj_baseline_on_a.item():.4f} -> {proj_steered_on_a.item():.4f} (變化量: {proj_steered_on_a.item() - proj_baseline_on_a.item():+.4f})")
print(f"對 B 的投影變化: {proj_baseline_on_b.item():.4f} -> {proj_steered_on_b.item():.4f} (變化量: {proj_steered_on_b.item() - proj_baseline_on_b.item():+.4f})")

print("\n生成的文本對比:")
print("【基準】:", tokenizer.decode(outputs_baseline.sequences[0][prompt_len:]))
print("【引導後】:", tokenizer.decode(outputs_steered.sequences[0][prompt_len:]))