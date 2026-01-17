import argparse
import sys
import os
import gc
import time
from datetime import timedelta
import json
import torch
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModelForCausalLM
import pytz
import torch._dynamo as dynamo

torch.set_float32_matmul_precision('high')  # gemma

# -----------------------------
# Dynamo 配置
# -----------------------------
dynamo.config.cache_size_limit = 64  # 可依模型大小調整
dynamo.config.suppress_errors = True

def maybe_reset_dynamo_cache(threshold=0.9):
    """
    如果 Dynamo cache 使用量超過 threshold，自動 reset
    """
    if hasattr(dynamo, "_cache") and len(dynamo._cache) / dynamo.config.cache_size_limit > threshold:
        print("[Dynamo] Cache 使用過高，正在重置...")
        dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()

# -----------------------------
# ModelRunner 類別
# -----------------------------
class ModelRunner:

    _loaded_models = {}  # 🔑 模型快取
    def __init__(self, model_name="gemma"):
        self.model_name = model_name
        
        # 模型名稱對應
        model_mapping = {
            "qwen": "Qwen/Qwen2.5-7B-Instruct",
            "llama": "meta-llama/Llama-3.1-8B-Instruct",
            "gemma": "google/gemma-3-4b-it"
        }
        
        # 自動對應完整模型名稱
        if model_name.lower() in model_mapping:
            self.model_name = model_mapping[model_name.lower()]
        else:
            self.model_name = model_name
        
        if self.model_name in ModelRunner._loaded_models:
            print(f"⚡ 使用快取模型: {self.model_name}")
            self.tokenizer, self.model = ModelRunner._loaded_models[self.model_name]
        else:
            print(f"🤖 載入模型: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            ModelRunner._loaded_models[self.model_name] = (self.tokenizer, self.model)
        
        # 判斷模型類型
        if "qwen" in self.model_name.lower():
            self.model_type = "qwen"
        elif "llama" in self.model_name.lower():
            self.model_type = "llama"
        elif "gemma" in self.model_name.lower():
            self.model_type = "gemma"
        else:
            self.model_type = "unknown"
        
        print(f"🔍 檢測到模型類型: {self.model_type}")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("✅ 模型載入完成")

        
    # def generate_response(self, prompt, max_tokens=4096):
    def generate_response(self, prompt, max_tokens=2048):
        """使用原始模型產生回應，並自動管理 Dynamo cache"""
        try:
            # 生成前檢查快取
            maybe_reset_dynamo_cache()
            
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 編碼輸入
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt",
                truncation=True,
                # max_length=8192
                max_length=6144
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # 生成回應
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # 解碼回應
            # response = self.tokenizer.decode(
            #     outputs[0][inputs['input_ids'].shape[1]:], 
            #     skip_special_tokens=True
            # )
            # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            # print("=== Full Decode ===")
            # print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))

            # print("=== Only New Tokens Decode ===")
            # print(self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))
            
            # 清理 GPU 與 Python 物件
            del outputs, inputs
            torch.cuda.empty_cache()
            gc.collect()

            return response.strip()
            
        except Exception as e:
            print(f"❌ 生成回應時發生錯誤: {e}")
            return None
