#!/usr/bin/env python3
"""
簡單的 Gemma-3-4b 聊天腳本
直接輸入問題，模型回應
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SimpleGemma3Chat:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """載入 Gemma-3-4b 模型"""
        model_name = "google/gemma-3-4b-it"
        print(f"🔄 載入模型: {model_name}")
        print(f"📱 使用設備: {self.device}")
        
        try:
            # 載入分詞器
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 載入模型
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True
                )
            
            print("✅ 模型載入成功！")
            return True
            
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            return False
    
    def chat(self, user_input: str, max_new_tokens: int = 8096):
        """與模型對話"""
        if self.model is None or self.tokenizer is None:
            print("❌ 模型尚未載入")
            return None
        
        # 構造對話格式
        conversation = [
            {"role": "user", "content": user_input}
        ]
        
        try:
            # 應用聊天模板
            prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 編碼
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            # 移動到正確設備
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # 生成回應
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 解碼回應
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取模型回應部分
            response = full_response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            print(f"❌ 生成回應失敗: {e}")
            return None

def main():
    """主函式"""
    print("🤖 Gemma-3-4b 簡單聊天機器人")
    print("輸入 'quit' 或 'exit' 結束對話")
    print("=" * 50)
    
    # 初始化聊天機器人
    chat_bot = SimpleGemma3Chat()
    
    # 載入模型
    if not chat_bot.load_model():
        print("無法載入模型，程式結束")
        return
    
    print("\n✅ 準備就緒！開始對話...")
    print("=" * 50)
    
    # 對話迴圈
    while True:
        try:
            # 獲取使用者輸入
            user_input = input("\n👤 您: ").strip()
            
            # 檢查結束條件
            if user_input.lower() in ['quit', 'exit', '退出', '結束']:
                print("👋 再見！")
                break
            
            # 跳過空輸入
            if not user_input:
                continue
            
            # 生成回應
            print("🤖 思考中...")
            response = chat_bot.chat(user_input)
            
            if response:
                print(f"🤖 Gemma: {response}")
            else:
                print("❌ 無法生成回應，請重試")
                
        except KeyboardInterrupt:
            print("\n👋 程式被中斷，再見！")
            break
        except Exception as e:
            print(f"❌ 發生錯誤: {e}")

if __name__ == "__main__":
    main()
