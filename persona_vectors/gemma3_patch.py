"""
為 Gemma-3 模型修正 eval_persona.py 中的產生參數
"""
import torch
from vllm import SamplingParams

def get_gemma3_sampling_params(tokenizer, temperature=1, top_p=1, max_tokens=1000, min_tokens=10):
    """為 Gemma-3 模型獲取最佳化的 SamplingParams"""
    
    # Gemma-3 的特殊 stop tokens
    stop_tokens = []
    
    # 加入標準的結束 token
    if tokenizer.eos_token:
        stop_tokens.append(tokenizer.eos_token)
    
    # 加入 Gemma 特有的對話結束 token
    gemma_stop_tokens = ["<end_of_turn>", "</s>", "<eos>"]
    for token in gemma_stop_tokens:
        if token not in stop_tokens:
            stop_tokens.append(token)
    
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=stop_tokens,
        min_tokens=min_tokens,  # 增加最小 token 數量
        repetition_penalty=1.1,  # 加入重複懲罰
        length_penalty=1.0       # 加入長度懲罰
    )

def is_gemma_model(model_name: str) -> bool:
    """判斷是否為 Gemma 模型"""
    return "gemma" in model_name.lower()

def patch_gemma3_sampling():
    """為 Gemma-3 修補 eval_persona.py 中的採樣函式"""
    
    import eval.eval_persona as eval_persona_module
    
    # 備份原始函式
    original_sample = eval_persona_module.sample
    
    def sample_with_gemma3_support(model, tokenizer, conversations, top_p=1, max_tokens=1000, temperature=1, min_tokens=1, lora_path=None):
        """支援 Gemma-3 的採樣函式"""
        
        # 檢查是否為 Gemma 模型
        model_name = getattr(model, 'model_config', {}).get('model', '')
        if not model_name:
            # 嘗試從 tokenizer 獲取模型名稱
            model_name = getattr(tokenizer, 'name_or_path', '')
        
        if is_gemma_model(model_name):
            print(f"🎯 偵測到 Gemma 模型，使用最佳化參數: {model_name}")
            
            # 使用 Gemma-3 最佳化的參數
            sampling_params = get_gemma3_sampling_params(
                tokenizer=tokenizer,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                min_tokens=max(min_tokens, 10)  # 至少 10 個 token
            )
        else:
            # 使用原始參數
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                skip_special_tokens=True,
                stop=[tokenizer.eos_token] if tokenizer.eos_token else [],
                min_tokens=min_tokens
            )
        
        # 準備提示文字
        texts = []
        for i, messages in enumerate(conversations):
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                texts.append(text)
            except Exception as e:
                print(f"⚠️ 聊天模板處理失敗，使用簡單格式: {e}")
                # 回退到簡單格式
                if isinstance(messages, list) and len(messages) > 0:
                    if isinstance(messages[-1], dict) and 'content' in messages[-1]:
                        texts.append(messages[-1]['content'])
                    else:
                        texts.append(str(messages[-1]))
                else:
                    texts.append(str(messages))
        
        generate_kwargs = {
            "sampling_params": sampling_params,
            "use_tqdm": True
        }
        
        try:
            if lora_path:
                from vllm.lora.request import LoRARequest
                completions = model.generate(texts, **generate_kwargs, lora_request=LoRARequest("default", 1, lora_path=lora_path))
            else:
                completions = model.generate(texts, **generate_kwargs)
            
            answers = [completion.outputs[0].text for completion in completions]
            
            # 除錯：檢查是否有空回應
            empty_count = sum(1 for answer in answers if not answer.strip())
            if empty_count > 0:
                print(f"⚠️ 偵測到 {empty_count}/{len(answers)} 個空回應")
                # 顯示前幾個範例
                for i, (text, answer) in enumerate(zip(texts[:3], answers[:3])):
                    print(f"範例 {i+1}:")
                    print(f"  輸入: {text[:100]}...")
                    print(f"  輸出: '{answer}'")
            
            return texts, answers
            
        except Exception as e:
            print(f"❌ 產生失敗: {e}")
            # 回退策略：返回空答案但保持格式一致
            return texts, ["" for _ in texts]
    
    # 替換原始函式
    eval_persona_module.sample = sample_with_gemma3_support
    print("✅ Gemma-3 採樣支援已啟用")
    
    return original_sample

if __name__ == "__main__":
    print("🔧 修補 eval_persona.py 以支援 Gemma-3...")
    patch_gemma3_sampling()
    print("✅ 修補完成")
