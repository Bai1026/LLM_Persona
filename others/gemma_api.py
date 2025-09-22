#!/usr/bin/env python3
"""
Gemma API 服務
提供 HTTP API 介面讓外部程式呼叫 Gemma 模型
"""

from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
import threading
import time
from datetime import datetime

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gemma_api.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)

class GemmaAPIService:
    """Gemma API 服務類別"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conversation_history = {}  # 儲存對話歷史
        self.model_loaded = False
        
    def load_model(self):
        """載入 Gemma 模型"""
        model_name = "google/gemma-3-4b-it"
        logging.info(f"🔄 載入模型: {model_name}")
        logging.info(f"📱 使用設備: {self.device}")
        
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
            
            self.model_loaded = True
            logging.info("✅ 模型載入成功！")
            return True
            
        except Exception as e:
            logging.error(f"❌ 模型載入失敗: {e}")
            return False
    
    def generate_response(self, user_input: str, max_tokens: int = 1000, session_id: str = "default"):
        """生成回應"""
        if not self.model_loaded:
            return None, "模型尚未載入"
        
        try:
            # 構造對話格式
            conversation = [
                {"role": "user", "content": user_input}
            ]
            
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
                    max_new_tokens=max_tokens,
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
            
            # 儲存對話歷史
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []
            
            self.conversation_history[session_id].append({
                "user": user_input,
                "assistant": response,
                "timestamp": datetime.now().isoformat()
            })
            
            return response, None
            
        except Exception as e:
            error_msg = f"生成回應失敗: {str(e)}"
            logging.error(error_msg)
            return None, error_msg
    
    def reset_conversation(self, session_id: str = "default"):
        """重設對話歷史"""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
        return True

# 建立全域服務實例
gemma_service = GemmaAPIService()

@app.route('/status', methods=['GET'])
def status():
    """檢查服務狀態"""
    return jsonify({
        "status": "running",
        "model_loaded": gemma_service.model_loaded,
        "device": gemma_service.device,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/chat', methods=['POST'])
def chat():
    """聊天端點"""
    try:
        data = request.get_json()
        
        # 檢查必要參數
        if not data or 'user_input' not in data:
            return jsonify({"error": "缺少 user_input 參數"}), 400
        
        user_input = data['user_input']
        max_tokens = data.get('max_tokens', 1000)
        session_id = data.get('session_id', 'default')
        
        # 生成回應
        response, error = gemma_service.generate_response(user_input, max_tokens, session_id)
        
        if error:
            return jsonify({"error": error}), 500
        
        return jsonify({
            "response": response,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"聊天端點錯誤: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    """重設對話歷史"""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', 'default')
        
        gemma_service.reset_conversation(session_id)
        
        return jsonify({
            "message": f"會話 {session_id} 已重設",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"重設端點錯誤: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    """獲取對話歷史"""
    try:
        session_id = request.args.get('session_id', 'default')
        
        history = gemma_service.conversation_history.get(session_id, [])
        
        return jsonify({
            "session_id": session_id,
            "history": history,
            "count": len(history),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"歷史端點錯誤: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """健康檢查"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

def load_model_async():
    """異步載入模型"""
    logging.info("🚀 開始載入 Gemma 模型...")
    success = gemma_service.load_model()
    if success:
        logging.info("✅ Gemma 模型載入完成，API 服務已就緒")
    else:
        logging.error("❌ Gemma 模型載入失敗")

def main():
    """主函式"""
    print("🤖 Gemma API 服務啟動中...")
    
    # 在背景執行緒中載入模型
    model_thread = threading.Thread(target=load_model_async, daemon=True)
    model_thread.start()
    
    # 設定 Flask 應用
    port = int(os.environ.get('GEMMA_API_PORT', 8002))
    host = os.environ.get('GEMMA_API_HOST', '0.0.0.0')
    
    print(f"🌐 API 服務將在 http://{host}:{port} 啟動")
    print("📡 可用端點:")
    print("  - GET  /status   - 檢查服務狀態")
    print("  - POST /chat     - 聊天對話")
    print("  - POST /reset    - 重設對話")
    print("  - GET  /history  - 獲取對話歷史")
    print("  - GET  /health   - 健康檢查")
    print("\n等待模型載入完成...")
    
    # 啟動 Flask 應用
    app.run(
        host=host,
        port=port,
        debug=False,
        threaded=True
    )

if __name__ == "__main__":
    main()
