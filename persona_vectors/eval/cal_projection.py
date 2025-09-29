import os
import asyncio
import yaml
from typing import Dict, List
import json
import pandas as pd
from tqdm import tqdm
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]

def save_jsonl(data, path):
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

def cos_sim(a, b):
    return (a*b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1))

def a_proj_b(a, b):
    return (a * b).sum(dim=-1) / b.norm(dim=-1)

def main(file_path, vector_path_list=[], layer_list=[], projection_type="proj", model_name="Qwen/Qwen2.5-7B-Instruct", overwrite=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    """Evaluate a model on all questions form the evaluation yaml file"""

    metric_model_name = os.path.basename(model_name) + "_"
    if not isinstance(vector_path_list, list):
        vector_path_list = [vector_path_list]
    if not isinstance(layer_list, list):
        layer_list = [layer_list]
    vector_dict = {}
    layer_dict = {}
    for vector_path in vector_path_list:
        vector = torch.load(vector_path, weights_only=False)
        for layer in layer_list:
            vector_name = vector_path.split("/")[-1].split(".")[0]
            metric_name = f"{metric_model_name}{vector_name}_{projection_type}_layer{layer}"
            vector_dict[metric_name] = vector[layer]
            layer_dict[metric_name] = layer

    # --- 關鍵修改開始：重構數據讀取邏輯 ---
    is_csv = False
    if file_path.endswith(".csv"):
        is_csv = True
        data = pd.read_csv(file_path)
        # 假設 CSV 有 'prompt' 和 'answer' 欄位
        prompts = data["prompt"].tolist()
        answers = data["answer"].tolist()
        
        # 檢查是否需要覆蓋
        vector_dict_keys = list(vector_dict.keys())
        for metric_name in vector_dict_keys:
            if metric_name in data.columns and not overwrite:
                print(f"Metric {metric_name} already exists, skipping...")
                vector_dict.pop(metric_name)

    else: # 處理 JSON 或 JSONL
        with open(file_path, 'r') as f:
            try:
                # 嘗試讀取單一的大 JSON 物件 (您的新格式)
                json_data = json.load(f)
                if "conversations" in json_data:
                    data = json_data["conversations"]
                    # 從新格式中提取 prompts 和 answers
                    prompts = [d.get("user_input", d.get("question", "")) for d in data]
                    answers = [d.get("response", "") for d in data]
                else: # 假設是 JSONL
                    f.seek(0)
                    data = [json.loads(line) for line in f]
                    # 假設 JSONL 有 'messages' 欄位
                    prompts = [tokenizer.apply_chat_template(d["messages"][:-1], tokenize=False, add_generation_prompt=True) for d in data]
                    answers = [d["messages"][-1]["content"] for d in data]
            except json.JSONDecodeError: # 處理 JSONL 的備用方案
                f.seek(0)
                data = [json.loads(line) for line in f]
                prompts = [tokenizer.apply_chat_template(d["messages"][:-1], tokenize=False, add_generation_prompt=True) for d in data]
                answers = [d["messages"][-1]["content"] for d in data]
        
        # 檢查是否需要覆蓋
        vector_dict_keys = list(vector_dict.keys())
        if data: # 確保 data 不是空的
            for metric_name in vector_dict_keys:
                if metric_name in data[0] and not overwrite:
                    print(f"Metric {metric_name} already exists, skipping...")
                    vector_dict.pop(metric_name)
    # --- 關鍵修改結束 ---


    if len(vector_dict) == 0:
        print("No metrics to calculate, exiting...")
        return
    else:
        print(f"Calculating {len(vector_dict)} metrics:")
        for metric_name in vector_dict.keys():
            print(f"{metric_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    projections = {k:[] for k in vector_dict.keys()}
    for prompt, answer in tqdm(zip(prompts, answers), total=len(prompts), desc=f"Calculating"):
        # 確保 answer 不是 None 或 float
        if not isinstance(answer, str):
            answer = "" # 如果 answer 是空的或非字串，設為空字串

        inputs = tokenizer(prompt + answer, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        outputs = model(**inputs, output_hidden_states=True)
        
        for metric_name in vector_dict.keys():
            layer = layer_dict[metric_name]
            vector = vector_dict[metric_name]
            
            # 確保有回應 token 可供計算
            if inputs['input_ids'].shape[1] > prompt_len:
                response_avg = outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1).detach().cpu()
                last_prompt = outputs.hidden_states[layer][:, prompt_len-1, :].detach().cpu()
                if projection_type == "proj":
                    projection = a_proj_b(response_avg, vector).item()
                elif projection_type == "prompt_last_proj":
                    projection = a_proj_b(last_prompt, vector).item()
                else:
                    projection = cos_sim(response_avg, vector).item()
            else: # 如果沒有回應 token，則投影為 0
                projection = 0.0
                
            projections[metric_name].append(projection)
    

    if is_csv:
        df = pd.read_csv(file_path) # 重新載入以確保對齊
        for metric_name in vector_dict.keys():
            df[metric_name] = projections[metric_name]
        df.to_csv(file_path, index=False)
    else:
        for i, d in enumerate(data):
            for metric_name in vector_dict.keys():
                d[metric_name] = projections[metric_name][i]
        
        # 決定是存回大的 JSON 還是 JSONL
        if "conversations" in locals().get('json_data', {}):
             json_data["conversations"] = data
             with open(file_path, 'w') as f:
                 json.dump(json_data, f, indent=2)
        else:
             save_jsonl(data, file_path)

    print(f"Projection results saved to {file_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--vector_path_list", type=str, nargs="+", default=[])
    parser.add_argument("--layer_list", type=int, nargs="+", default=[])
    parser.add_argument("--projection_type", type=str, default="proj", choices=["proj", "prompt_last_proj", "cos_sim"])
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.file_path, args.vector_path_list, args.layer_list, args.projection_type, args.model_name, args.overwrite)