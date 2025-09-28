import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich import print

# --- 輔助函數 ---
def a_proj_b(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """計算向量 a 在向量 b 上的純量投影。"""
    if a.dtype != b.dtype:
        a = a.to(b.dtype)
    
    b_norm = b.norm()
    return torch.tensor(0.0, dtype=b.dtype, device=b.device) if b_norm == 0 else torch.dot(a, b) / b_norm

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
VECTOR_A_PATH = "Llama-3.1-8B-Instruct/multi_role/creative_professional_response_avg_diff.pt"
VECTOR_B_PATH = "Llama-3.1-8B-Instruct/multi_role/environmentalist_response_avg_diff.pt"

STEERING_LAYER = 20
ANALYSIS_LAYER = 20

STEERING_COEFFICIENT = 2.0

AUT_LIST = [
    "What are some creative use for Fork? The goal is to come up with creative ideas, which are ideas that strike people as clever, unusual, interesting, uncommon, humorous, innovative, or different. Present a list of 5 creative and diverse uses for Fork."
    "What are some creative use for Jar? The goal is to come up with creative ideas, which are ideas that strike people as clever, unusual, interesting, uncommon, humorous, innovative, or different. Present a list of 5 creative and diverse uses for Jar."
    ]

INS_LIST = [
    "Name all the round things you can think of.",
    "Name all the things you can think of that will make a noise.",
    "Name all the things you can think of that have a screen.",
]

SIMI_LIST = [
    "Tell me all the ways in which a kite and a balloon are alike.",
    "Tell me all the ways in which a pencil and a pen are alike.",
    "Tell me all the ways in which a chair and a couch are alike.",
]

SCI_LIST = [
    "If you can take a spaceship to travel in outer space and go to a planet, what scientific questions do you want to research? For example, are there any living things on the planet?",
    "Please think up as many possible improvements as you can to a regular bicycle, making it more interesting, more useful and more beautiful. For example, make the tires reflective, so they can be seen in the dark."
]

# PROMPT = "Human: What are some innovative solutions for urban sustainability? Assistant:"
# PROMPT_LIST = "What is your plan to save the planet?"
# PROMPT = "What's your thought about AI application?"

PROMPT_LIST = AUT_LIST + INS_LIST + SIMI_LIST + SCI_LIST

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
model.eval()

model_dtype = torch.bfloat16
vectors_L_steer = torch.load(VECTOR_A_PATH, map_location=device)
vector_a_steer = vectors_L_steer[STEERING_LAYER].to(dtype=model_dtype)
vector_b_steer = torch.load(VECTOR_B_PATH, map_location=device)[STEERING_LAYER].to(dtype=model_dtype)

vector_a_proj = vectors_L_steer[ANALYSIS_LAYER].to(dtype=model_dtype)
vector_b_proj = torch.load(VECTOR_B_PATH, map_location=device)[ANALYSIS_LAYER].to(dtype=model_dtype)

vector_a_steer /= vector_a_steer.norm()
vector_b_steer /= vector_b_steer.norm()
vector_a_proj /= vector_a_proj.norm()
vector_b_proj /= vector_b_proj.norm()

inputs = tokenizer(PROMPT, return_tensors="pt").to(device)                                                                                                                    
with torch.no_grad():
    outputs_baseline = model.generate(
        **inputs, 
        max_new_tokens=512, 
        output_hidden_states=True, 
        return_dict_in_generate=True
    )

prompt_len = inputs.input_ids.shape[1]
generated_hidden_states_baseline = []
for token_idx, token_hidden_states in enumerate(outputs_baseline.hidden_states):
    if token_idx >= prompt_len:
        layer_activation = token_hidden_states[ANALYSIS_LAYER]
        generated_hidden_states_baseline.append(layer_activation[:, -1, :])

if generated_hidden_states_baseline:
    activation_L_analysis_baseline = torch.stack(generated_hidden_states_baseline).mean(dim=0).squeeze()
else:
    activation_L_analysis_baseline = outputs_baseline.hidden_states[0][ANALYSIS_LAYER][:, -1, :].squeeze()


# --- 4. 引導測試 ---
print(f"\n--- 執行引導測試 (在 L{STEERING_LAYER} 層施加 A+B 引導) ---")
steering_vec = (vector_a_steer + vector_b_steer) * STEERING_COEFFICIENT

def steering_hook(module, args, kwargs, output):
    modified_output = output[0] + steering_vec
    return (modified_output,) + output[1:]

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
    hook_handle.remove()
    print("✅ Forward Hook 已被移除。")

generated_hidden_states_steered = []
for token_idx, token_hidden_states in enumerate(outputs_steered.hidden_states):
    if token_idx >= prompt_len:
        layer_activation = token_hidden_states[ANALYSIS_LAYER]
        generated_hidden_states_steered.append(layer_activation[:, -1, :])

if generated_hidden_states_steered:
    activation_L_analysis_steered = torch.stack(generated_hidden_states_steered).mean(dim=0).squeeze()
else:
    activation_L_analysis_steered = outputs_steered.hidden_states[0][ANALYSIS_LAYER][:, -1, :].squeeze()


# --- 5. 結果分析 ---
print("\n" + "="*20 + " 實驗結果分析 " + "="*20)
print(f"Prompt: {PROMPT}")
print(f"引導層: {STEERING_LAYER}, 分析層: {ANALYSIS_LAYER}, 引導係數: {STEERING_COEFFICIENT}")

# --- 5a. 絕對投影分析 (測量「最終目的地」) ---
print("\n--- 5a. 絕對投影分析 (測量最終『思維狀態』) ---")
proj_baseline_on_a = a_proj_b(activation_L_analysis_baseline, vector_a_proj)
proj_baseline_on_b = a_proj_b(activation_L_analysis_baseline, vector_b_proj)
proj_steered_on_a = a_proj_b(activation_L_analysis_steered, vector_a_proj)
proj_steered_on_b = a_proj_b(activation_L_analysis_steered, vector_b_proj)

print(f"基準投影: A={proj_baseline_on_a.item():.4f}, B={proj_baseline_on_b.item():.4f}")
print(f"引導後投影: A={proj_steered_on_a.item():.4f}, B={proj_steered_on_b.item():.4f}")

# --- 5b. 差值投影分析 (測量「羅盤」的影響) ---
print("\n--- 5b. 差值投影分析 (測量引導向量的『因果效應』) ---")

# 計算引導前後的激活「變化量」 (delta)
activation_delta = activation_L_analysis_steered - activation_L_analysis_baseline

# 測量這個「變化量」在 A 和 B 方向上的投影
proj_delta_on_a = a_proj_b(activation_delta, vector_a_proj)
proj_delta_on_b = a_proj_b(activation_delta, vector_b_proj)
print(f"激活變化量在 A 上的投影: {proj_delta_on_a.item():.4f}")
print(f"激活變化量在 B 上的投影: {proj_delta_on_b.item():.4f}")

# --- 6. 生成文本對比 ---
print("\n" + "="*20 + " 生成文本對比 " + "="*20)
print("【基準】:", tokenizer.decode(outputs_baseline.sequences[0][prompt_len:]))
print("【引導後】:", tokenizer.decode(outputs_steered.sequences[0][prompt_len:]))
