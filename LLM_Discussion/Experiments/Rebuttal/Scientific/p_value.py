import json
import numpy as np
from scipy import stats

def load_json(path, metric="originality_scores"):
    with open(path, "r") as f:
        data = json.load(f)
    return data["summary"][metric]

qwen_files = [
    "qwen_single_simple_eval_results.json",
    "Scientific_qwen_1_1_Qwen_SingleRole_Temp1_qwen_0924-0030_100_chat_log_simple_eval_results.json",
    "qwen_multi_simple_eval_results.json",
    "Scientific_multi_debate_roleplay_4_5_qwen2-5-7b-instruct_Environmentalist-CreativeProfessional-Futurist-Futurist_chat_log_2025-09-26-14-00-38_100_simple_eval_results.json",
    "qwen_vector_simple_eval_results.json"
]

llama_files = [
    "llama_single_simple_eval_results.json",
    "Scientific_llama_1_1_Llama_SingleRole_Temp1_llama_0924-0242_100_chat_log_simple_eval_results.json",
    "llama_multi_simple_eval_results.json",
    "Scientific_multi_debate_roleplay_4_5_llama-3-1-8b-instruct_Environmentalist-CreativeProfessional-Futurist-Futurist_chat_log_2025-09-27-15-32-21_100_simple_eval_results.json",
    "llama_vector_simple_eval_results.json"
]

gemma_files = [
    "gemma_single_simple_eval_results.json",
    "Scientific_gemma_1_1_Gemma_SingleRole_Temp1_gemma_0927-0036_100_chat_log_simple_eval_results.json",
    "gemma_multi_simple_eval_results.json",
    "Scientific_multi_debate_roleplay_4_5_gemma-3-4b-it_Environmentalist-CreativeProfessional-Futurist-Futurist_chat_log_2025-10-01-00-36-23_100_simple_eval_results.json",
    "gemma_vector_simple_eval_results.json"
]



# --- 執行成對 t-test ---
def results(model_file, metric="elaboration_scores"):

    ori_sa = load_json(model_file[0], metric)
    ori_sa_t1 = load_json(model_file[1], metric)
    ori_samrp = load_json(model_file[2], metric)
    ori_llmd = load_json(model_file[3], metric)
    ori_billy = load_json(model_file[4], metric)

    group_size = 4
    ori_llmd = [ sum(ori_llmd[i:i+group_size]) / len(ori_llmd[i:i+group_size])
                for i in range(0, len(ori_llmd), group_size) ]

    # * 比較 BILLY vs SA
    t_statistic, p_value = stats.ttest_rel(ori_billy, ori_sa)
    print(f"比較 BILLY vs SA:")
    print(f"T-statistic: {t_statistic:.4f}")
    print(f"P-value: {p_value:.8f}")
    if p_value < 0.001:
        print("差異達到極度顯著水準 (p < 0.001)")
    elif p_value < 0.05:  # 因為已經排除了 < 0.001 的情況，所以這裡隱含了 p >= 0.001
        print("差異達到統計顯著水準 (p < 0.05)")
    else:
        print("差異未達到統計顯著水準")
    print("-" * 30)

    # * 比較 BILLY vs SA-Temp-1
    t_statistic, p_value = stats.ttest_rel(ori_billy, ori_sa_t1)
    print(f"比較 BILLY vs SA-Temp-1:")
    print(f"T-statistic: {t_statistic:.4f}")
    print(f"P-value: {p_value:.8f}")
    if p_value < 0.001:
        print("差異達到極度顯著水準 (p < 0.001)")
    elif p_value < 0.05:  # 因為已經排除了 < 0.001 的情況，所以這裡隱含了 p >= 0.001
        print("差異達到統計顯著水準 (p < 0.05)")
    else:
        print("差異未達到統計顯著水準")
    print("-" * 30)

    # * 比較 BILLY vs SA-MRP
    t_statistic, p_value = stats.ttest_rel(ori_billy, ori_samrp)
    print(f"比較 BILLY vs SA-MRP:")
    print(f"T-statistic: {t_statistic:.4f}")
    print(f"P-value: {p_value:.8f}")
    if p_value < 0.001:
        print("差異達到極度顯著水準 (p < 0.001)")
    elif p_value < 0.05:  # 因為已經排除了 < 0.001 的情況，所以這裡隱含了 p >= 0.001
        print("差異達到統計顯著水準 (p < 0.05)")
    else:
        print("差異未達到統計顯著水準")
    print("-" * 30)

    # * 比較 BILLY vs LLM-D
    t_statistic, p_value = stats.ttest_rel(ori_billy, ori_llmd)
    print(f"比較 BILLY vs LLM-D:")
    print(f"T-statistic: {t_statistic:.4f}")
    print(f"P-value: {p_value:.8f}")
    if p_value < 0.001:
        print("差異達到極度顯著水準 (p < 0.001)")
    elif p_value < 0.05:  # 因為已經排除了 < 0.001 的情況，所以這裡隱含了 p >= 0.001
        print("差異達到統計顯著水準 (p < 0.05)")
    else:
        print("差異未達到統計顯著水準")
    print("-" * 30)

## originality_scores, elaboration_scores
results(qwen_files, "originality_scores")
print("\n====\n")
results(qwen_files, "elaboration_scores")
print("\n====\n")
results(llama_files, "originality_scores")
print("\n====\n")
results(llama_files, "elaboration_scores")
print("\n====\n")
results(gemma_files, "originality_scores")
print("\n====\n")
results(gemma_files, "elaboration_scores")