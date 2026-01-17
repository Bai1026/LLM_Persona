
### input 2048, output 2048 撐得住
### 在更大就會跑非常慢

### 更大一點的話會有以下錯誤
'''
W0928 15:54:35.390000 303740 site-packages/torch/_dynamo/convert_frame.py:906] [0/9] torch._dynamo hit config.cache_size_limit (
8)                                                                                                                              
W0928 15:54:35.390000 303740 site-packages/torch/_dynamo/convert_frame.py:906] [0/9]    function: 'forward' (/root/miniconda/env
s/persona_vec/lib/python3.10/site-packages/transformers/models/gemma3/modeling_gemma3.py:1275)                                  
W0928 15:54:35.390000 303740 site-packages/torch/_dynamo/convert_frame.py:906] [0/9]    last reason: 0/7: Cache line invalidate$
 because L['past_key_values'].key_cache[33] got deallocated                                                                     
W0928 15:54:35.390000 303740 site-packages/torch/_dynamo/convert_frame.py:906] [0/9] To log all recompilation reasons, use TORCH
_LOGS="recompiles".                                                                                                             
W0928 15:54:35.390000 303740 site-packages/torch/_dynamo/convert_frame.py:906] [0/9] To diagnose recompilation issues, see https
://pytorch.org/docs/main/torch.compiler_troubleshooting.html.                                                                   
❌ 生成回應時發生錯誤: cache_size_limit reached
'''



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

torch.set_float32_matmul_precision('high') ## gemma

class ModelRunner:

    _loaded_models = {}  # 🔑 快取
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
            # 如果輸入的已經是完整名稱，直接使用
            self.model_name = model_name
        
        if self.model_name in ModelRunner._loaded_models:
            print(f"⚡ 使用快取模型: {self.model_name}")
            self.tokenizer, self.model = ModelRunner._loaded_models[self.model_name]
        else:
            # 載入模型和分詞器
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
            # 存到快取
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

        
    def generate_response(self, prompt, max_tokens=2048):
        """使用原始模型產生回應"""
        try:
            # 格式化對話
            # print(f"📣 📣 📣:\n{prompt}📣 📣 📣\n")
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
                max_length=6146
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
                    repetition_penalty=1.3, ## gemma
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id, ## gemma
                )
            
            # 解碼回應
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            # print(f"🤖 🤖 🤖:\n{response}🤖 🤖 🤖\n")
            
            del outputs
            torch.cuda.empty_cache()
            gc.collect()

            return response.strip()
            
        except Exception as e:
            print(f"❌ 生成回應時發生錯誤: {e}")
            return None