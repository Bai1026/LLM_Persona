from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from typing import List, Dict
import gc

app = Flask(__name__)

class PureModelChatbot:
    """純粹的 Qwen/Llama 聊天機器人，不使用 vector"""
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.conversation_history = []
        
        print(f"🚀 正在載入模型: {model_name}")
        
        # 載入 tokenizer 和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"✅ 模型載入完成")
        print(f"📱 裝置: {self.model.device}")
    
    def generate_response(self, user_input: list, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """產生回應"""
        try:
            # 建構對話格式
            # messages = self.conversation_history + [{"role": "user", "content": user_input}]
            ## Amy 修改 -------
            # messages = [{"role": "user", "content": user_input}]
            messages = user_input
            ## ---------------
            
            # 使用 tokenizer 的 chat template
            if hasattr(self.tokenizer, 'apply_chat_template'):
                formatted_input = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                # 如果沒有 chat template，使用簡單格式
                formatted_input = self._format_conversation(messages)
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_input, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                # max_length=2048
                # max_length=32768 ### 新增 --------
            )
            
            # 移到正確的裝置
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            ### Amy 新增 --------
            # input_length = inputs['input_ids'].shape[1]
            # available_tokens = self.tokenizer.model_max_length - input_length
            # actual_max_tokens = min(max_tokens, available_tokens)
            # if actual_max_tokens <= 0:
            #     return "抱歉，輸入過長，無法生成回應。"
            ### -------------
            
            # 產生回應
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    # max_new_tokens=max_tokens,
                    max_new_tokens=8196, ### Amy 新增 --------
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # 解碼回應
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # 更新對話歷史
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # 限制對話歷史長度
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            ### Amy 新增 --------------
            del outputs
            torch.cuda.empty_cache()
            gc.collect()
            ## -------------------
            return response
            
        except Exception as e:
            print(f"❌ 產生回應時發生錯誤: {e}")
            return f"抱歉，處理您的訊息時發生錯誤: {str(e)}"
    
    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """格式化對話（當沒有 chat template 時使用）"""
        formatted = ""
        for msg in messages:
            if msg["role"] == "user":
                formatted += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                formatted += f"Assistant: {msg['content']}\n"
        formatted += "Assistant: "
        return formatted
    
    def reset_conversation(self):
        """重設對話歷史"""
        self.conversation_history = []
        print("🔄 對話歷史已清除")
    
    def get_status(self) -> Dict:
        """取得狀態資訊"""
        return {
            "model_name": self.model_name,
            "device": str(self.model.device),
            "conversation_length": len(self.conversation_history),
            "memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }

# 全域變數
chatbot = None

@app.route('/chat', methods=['POST'])
def chat_api():
    """聊天 API 端點"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "缺少 message 參數"}), 400
        
        user_input = data['message']
        max_tokens = data.get('max_tokens', 512)
        temperature = data.get('temperature', 0.7)
        
        print(f"🧑 User: {user_input}")
        response = chatbot.generate_response(user_input, max_tokens, temperature)
        print(f"🤖 Assistant: {response}")
        
        return jsonify({
            "message": user_input,
            "response": response,
            "status": "success"
        })
        
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_conversation():
    """重設對話歷史"""
    try:
        chatbot.reset_conversation()
        return jsonify({"status": "conversation reset"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """取得模型狀態"""
    try:
        status = chatbot.get_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康檢查"""
    return jsonify({"status": "healthy", "model_loaded": chatbot is not None})

def main():
    global chatbot
    
    parser = argparse.ArgumentParser(description="純粹模型 API 服務")
    parser.add_argument("--model", 
                       default="Qwen/Qwen2.5-7B-Instruct",
                       help="模型名稱 (例如: Qwen/Qwen2.5-7B-Instruct 或 meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--device", 
                       default="auto",
                       help="裝置 (auto, cpu, cuda, cuda:0 等)")
    parser.add_argument("--host", 
                       default="127.0.0.1", 
                       help="API 主機位址")
    parser.add_argument("--port", 
                       type=int, 
                       default=5000, 
                       help="API 連接埠")
    
    args = parser.parse_args()
    
    # 初始化聊天機器人
    print("🚀 正在初始化純粹模型 API 服務...")
    print(f"📱 使用模型: {args.model}")
    print(f"🔧 裝置: {args.device}")
    
    try:
        chatbot = PureModelChatbot(
            model_name=args.model,
            device=args.device
        )
        
        print(f"✅ API 服務啟動於 http://{args.host}:{args.port}")
        print("📋 可用端點:")
        print(f"  POST /chat - 傳送訊息")
        print(f"    參數: {{\"message\": \"你的訊息\", \"max_tokens\": 512, \"temperature\": 0.7}}")
        print(f"  POST /reset - 重設對話")
        print(f"  GET /status - 取得狀態")
        print(f"  GET /health - 健康檢查")
        
        app.run(host=args.host, port=args.port, debug=False)
        
    except Exception as e:
        print(f"❌ 初始化失敗: {e}")
        return 1

if __name__ == "__main__":
    main()
