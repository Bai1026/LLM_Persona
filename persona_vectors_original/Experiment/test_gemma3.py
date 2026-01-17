import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def debug_gemma3_model():
    """除錯 Gemma-3 模型的結構和回應問題"""
    
    # 設定環境變數來協助除錯
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    model_name = "google/gemma-3-4b-it"
    print(f"🔍 載入模型: {model_name}")
    
    try:
        # 載入分詞器
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 嘗試使用更穩定的載入方式
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # 使用 bfloat16 而非 float16
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
        )
        
        # 設定 pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"📊 模型類型: {type(model).__name__}")
        print(f"📊 模型設備: {next(model.parameters()).device}")
        print(f"📊 模型精度: {next(model.parameters()).dtype}")
        print(f"📊 分詞器資訊: pad_token={tokenizer.pad_token}, eos_token={tokenizer.eos_token}")
        
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        # 回退到 CPU 和 float32
        print("🔄 嘗試回退到 CPU 和 float32...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"📊 回退模型類型: {type(model).__name__}")
        print(f"📊 回退模型設備: {next(model.parameters()).device}")
        print(f"📊 回退模型精度: {next(model.parameters()).dtype}")
    
    # 檢查模型結構
    print("\n🏗️ 模型結構:")
    print(f"  - 總層數: {len(model.model.language_model.layers)}")
    print(f"  - 第一層類型: {type(model.model.language_model.layers[0]).__name__}")
    
    # 測試基本推理（使用最保守的參數）
    print("\n🧪 測試基本推理:")
    test_prompt = "Hello"
    
    try:
        inputs = tokenizer(
            test_prompt, 
            return_tensors="pt",
            add_special_tokens=True,
            max_length=8192,  # 明確設定最大長度
            truncation=True,
            padding=False
        )
        
        # 移動到相同設備
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 加入注意力遮罩
        if 'attention_mask' not in inputs:
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
        
        with torch.no_grad():
            # 使用最簡單的貪婪解碼
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=10,
                do_sample=False,  # 貪婪解碼
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                output_scores=False,
                return_dict_in_generate=False
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"✅ 輸入: {test_prompt}")
            print(f"✅ 輸出: {response}")
            
    except Exception as e:
        print(f"❌ 基本推理失敗: {e}")
        print("🔍 嘗試更簡單的測試...")
        
        # 更簡單的測試 - 直接前向傳播
        try:
            inputs = tokenizer("Hello", return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = model(**inputs).logits
                next_token_id = torch.argmax(logits[0, -1, :], dim=-1)
                next_token = tokenizer.decode(next_token_id)
                print(f"✅ 前向傳播成功，下一個 token: '{next_token}'")
                
        except Exception as e2:
            print(f"❌ 前向傳播也失敗: {e2}")
            return None, None
    
    # 測試簡單的對話格式
    print("\n🎭 測試簡單對話:")
    try:
        conversation = [
            {"role": "user", "content": "What is art?"},
        ]
        
        prompt = tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"✅ 對話測試成功")
            print(f"   完整回應: {response}")
            
    except Exception as e:
        print(f"❌ 對話測試失敗: {e}")
    
    return model, tokenizer

def test_different_precision():
    """測試不同精度設定"""
    model_name = "google/gemma-3-4b-it"
    
    precisions = [
        ("float32", torch.float32),
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
    ]
    
    for name, dtype in precisions:
        print(f"\n🧮 測試 {name} 精度:")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto" if dtype != torch.float32 else "cpu"
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 簡單測試
            inputs = tokenizer("Hello", return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
                
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"✅ {name} 成功: {response}")
            
            # 清理記憶體
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ {name} 失敗: {e}")

if __name__ == "__main__":
    print("🔬 開始 Gemma-3 診斷...")
    
    # 先測試不同精度
    test_different_precision()
    
    print("\n" + "="*50)
    print("🎯 詳細模型測試:")
    
    # 詳細測試
    model, tokenizer = debug_gemma3_model()
    
    if model is not None:
        print("\n✅ 模型載入成功，可以進行後續的 eval_persona 整合")
    else:
        print("\n❌ 模型載入失敗，需要進一步調查")