import torch
import os

# 禁用 torch.compile 以避免 CUDA graph 衝突
os.environ["TORCH_COMPILE"] = "0"
torch._dynamo.config.disable = True

from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from pathlib import Path

# 添加父目錄到路徑，讓模組可以找到
sys.path.append(str(Path(__file__).parent.parent))

from activation_steer import ActivationSteerer
import argparse

class PersonaChatbot:
    def __init__(self, model_name, vector_path, layer_idx=20, steering_coef=2.0):
        self.model_name = model_name
        self.layer_idx = layer_idx
        self.steering_coef = steering_coef
        
        # 載入模型
        print(f"🤖 載入模型: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager"  # 使用 eager attention 避免編譯問題
        )
        
        # 禁用模型的編譯優化
        if hasattr(self.model, '_is_compiled'):
            self.model._is_compiled = False
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 載入persona向量
        print(f"📊 載入向量: {vector_path}")
        self.persona_vectors = torch.load(vector_path, weights_only=False)
        
        self.conversation_history = []
    
    def set_steering_params(self, layer_idx=None, coef=None):
        """動態調整steering參數"""
        if layer_idx is not None:
            self.layer_idx = layer_idx
        if coef is not None:
            self.steering_coef = coef
        print(f"🔧 更新參數: layer={self.layer_idx}, coef={self.steering_coef}")
    
    # TODO: 看 max_token 是否需要修改, creativity 需要大量輸出
    def generate_response(self, user_input, max_tokens=1000):
        """產生模型回應"""
        # 更新對話歷史
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # 建構prompt
        prompt = self.tokenizer.apply_chat_template(
            self.conversation_history, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]
        
        # 使用steering產生回應
        with ActivationSteerer(
            self.model, 
            self.persona_vectors[self.layer_idx], 
            coeff=self.steering_coef, 
            layer_idx=self.layer_idx-1,
            positions="response"
        ):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, 512),   # 進一步限制長度
                    do_sample=True,
                    # temperature=0.6,                       # 降低創意度
                    top_p=0.8,                            # 降低隨機性
                    repetition_penalty=1.2,               # 增強重複懲罰
                    no_repeat_ngram_size=4,               # 避免4-gram重複
                    # early_stopping=True,                  # 重新啟用早期停止
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    # length_penalty=0.8,                   # 鼓勵較短回應
                )
        
        # 解碼回應
        response = self.tokenizer.decode(
            outputs[0][input_length:], 
            skip_special_tokens=True
        ).strip()
        
        # 清理記憶體和CUDA快取
        del outputs
        torch.cuda.empty_cache()
        
        # 更新對話歷史
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # 限制歷史長度
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        return response
    
    def chat(self):
        """開始互動對話"""
        print("✅ 聊天機器人準備就緒！")
        print("💬 指令說明:")
        print("  - 直接輸入文字開始對話")
        print("  - '/layer X' 調整目標層數")
        print("  - '/coef X' 調整steering強度") 
        print("  - '/reset' 清除對話歷史")
        print("  - '/quit' 結束對話")
        print("=" * 50)
        
        while True:
            try:
                user_input = input(f"\n🧑 You: ").strip()
                
                if not user_input:
                    continue
                
                # 處理指令
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    break
                
                # 產生回應
                print("🤖 Model: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 對話已中斷！")
                break
            except Exception as e:
                print(f"\n❌ 錯誤: {e}")
    
    def _handle_command(self, command):
        """處理用戶指令"""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == '/layer' and len(parts) == 2:
            try:
                layer = int(parts[1])
                self.set_steering_params(layer_idx=layer)
            except ValueError:
                print("❌ 層數必須是整數")
        
        elif cmd == '/coef' and len(parts) == 2:
            try:
                coef = float(parts[1])
                self.set_steering_params(coef=coef)
            except ValueError:
                print("❌ 係數必須是數字")
        
        elif cmd == '/reset':
            self.conversation_history = []
            print("🔄 對話歷史已清除")
        
        elif cmd == '/quit':
            print("👋 再見！")
            exit()
        
        else:
            print("❌ 未知指令")

def main():
    parser = argparse.ArgumentParser(description="Persona Chatbot")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="模型名稱")
    parser.add_argument("--vector_path", required=True, help="persona向量路徑")
    parser.add_argument("--layer", type=int, default=20, help="目標層數")
    parser.add_argument("--coef", type=float, default=2.0, help="steering係數")
    
    args = parser.parse_args()
    
    chatbot = PersonaChatbot(
        model_name=args.model,
        vector_path=args.vector_path,
        layer_idx=args.layer,
        steering_coef=args.coef
    )
    
    chatbot.chat()

if __name__ == "__main__":
    main()