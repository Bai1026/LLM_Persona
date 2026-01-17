#!/usr/bin/env python3
"""
除錯版本的 API - 增加更詳細的錯誤處理和記憶體監控
"""

from flask import Flask, request, jsonify
import argparse
from rich import print
import traceback
import torch
import gc
import psutil
import os
from multi_persona_handler import MultiPersonaChatbot

app = Flask(__name__)
chatbot = None

def log_memory_usage():
    """記錄記憶體使用情況"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    print(f"🔧 記憶體使用: {memory_info.rss / 1024 / 1024:.1f} MB")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_memory_cached = torch.cuda.memory_reserved() / 1024 / 1024
        print(f"🔧 GPU 記憶體: {gpu_memory:.1f} MB allocated, {gpu_memory_cached:.1f} MB cached")

@app.route('/chat', methods=['POST'])
def chat_api():
    """API 端點：接收使用者輸入並回傳模型回應"""
    try:
        print("🔧 開始處理請求...")
        log_memory_usage()
        
        data = request.get_json()
        if not data or 'user_input' not in data:
            return jsonify({"error": "缺少 user_input 參數"}), 400
        
        user_input = data['user_input']
        max_tokens = data.get('max_tokens', 16384)
        
        print(f"🧑 User: {user_input}")
        print(f"🔧 Max tokens: {max_tokens}")
        
        # 清理 GPU 記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        print("🔧 開始生成回應...")
        response = chatbot.generate_response(user_input, max_tokens)
        print(f"🤖 Model: {response}")
        
        log_memory_usage()
        
        return jsonify({
            "user_input": user_input,
            "response": response,
            "current_weights": chatbot.current_weights,
            "status": "success"
        })
        
    except Exception as e:
        print(f"❌ 詳細錯誤資訊:")
        print(f"   錯誤類型: {type(e).__name__}")
        print(f"   錯誤訊息: {str(e)}")
        print(f"   錯誤追蹤:")
        traceback.print_exc()
        
        # 記錄記憶體狀況
        log_memory_usage()
        
        # 嘗試釋放記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }), 500

@app.route('/set_persona_weights', methods=['POST'])
def set_persona_weights():
    """設定 persona 權重"""
    try:
        data = request.get_json()
        weights = data.get('weights')
        
        if not weights:
            return jsonify({"error": "缺少 weights 參數"}), 400
        
        chatbot.update_persona_weights(weights)
        
        return jsonify({
            "current_weights": chatbot.current_weights,
            "status": "weights updated"
        })
    except Exception as e:
        print(f"❌ 設定權重錯誤: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/set_persona_mode', methods=['POST'])
def set_persona_mode():
    """設定 persona 模式"""
    try:
        data = request.get_json()
        mode = data.get('mode')
        
        if not mode:
            return jsonify({"error": "缺少 mode 參數"}), 400
        
        chatbot.set_persona_mode(mode)
        
        return jsonify({
            "mode": mode,
            "current_weights": chatbot.current_weights,
            "status": "mode updated"
        })
    except Exception as e:
        print(f"❌ 設定模式錯誤: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_conversation():
    """重設對話歷史"""
    try:
        chatbot.reset_conversation()
        print("🔄 對話歷史已重設")
        
        # 清理記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        log_memory_usage()
        return jsonify({"status": "conversation reset"})
    except Exception as e:
        print(f"❌ 重設錯誤: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """取得當前狀態"""
    try:
        log_memory_usage()
        return jsonify({
            "model_name": chatbot.base_chatbot.model_name,
            "layer_idx": chatbot.base_chatbot.layer_idx,
            "steering_coef": chatbot.base_chatbot.steering_coef,
            "current_weights": chatbot.current_weights,
            "num_personas": len(chatbot.persona_handler.personas),
            "conversation_length": len(chatbot.base_chatbot.conversation_history)
        })
    except Exception as e:
        print(f"❌ 狀態查詢錯誤: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/memory', methods=['GET'])
def get_memory_status():
    """取得記憶體狀態"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        result = {
            "ram_usage_mb": memory_info.rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent()
        }
        
        if torch.cuda.is_available():
            result.update({
                "gpu_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "gpu_cached_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "gpu_total_mb": torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            })
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def main():
    global chatbot
    
    parser = argparse.ArgumentParser(description="Debug Multi-Persona Chatbot API")
    parser.add_argument("--model", default="google/gemma-2-9b-it", help="模型名稱")
    parser.add_argument("--vector_paths", nargs='+', required=True, help="多個 persona 向量路徑")
    parser.add_argument("--layer", type=int, default=20, help="目標層數")
    parser.add_argument("--coef", type=float, default=2.0, help="steering 係數")
    parser.add_argument("--fusion_method", default="weighted_average", 
                       choices=["weighted_average", "concatenate", "attention", "dynamic"],
                       help="向量融合方法")
    parser.add_argument("--host", default="127.0.0.1", help="API 主機位址")
    parser.add_argument("--port", type=int, default=5000, help="API 連接埠")
    
    args = parser.parse_args()
    
    try:
        # 初始化多重 persona 聊天機器人
        print("🚀 正在初始化多重 Persona API 服務...")
        print(f"📁 載入 persona 向量: {args.vector_paths}")
        
        log_memory_usage()
        
        chatbot = MultiPersonaChatbot(
            model_name=args.model,
            vector_paths=args.vector_paths,
            layer_idx=args.layer,
            steering_coef=args.coef,
            fusion_method=args.fusion_method
        )
        
        print(f"✅ API 服務啟動於 http://{args.host}:{args.port}")
        print("📋 可用端點:")
        print(f"  POST /chat - 傳送訊息")
        print(f"  POST /set_persona_weights - 設定權重")
        print(f"  POST /set_persona_mode - 設定模式")
        print(f"  POST /reset - 重設對話")
        print(f"  GET /status - 取得狀態")
        print(f"  GET /memory - 記憶體狀態")
        
        log_memory_usage()
        
        app.run(host=args.host, port=args.port, debug=False)
        
    except Exception as e:
        print(f"❌ 初始化失敗: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
