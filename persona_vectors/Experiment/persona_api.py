from flask import Flask, request, jsonify
import argparse
from rich import print
from multi_persona_handler import MultiPersonaChatbot

app = Flask(__name__)
chatbot = None

@app.route('/chat', methods=['POST'])
def chat_api():
    """API 端點：接收使用者輸入並回傳模型回應"""
    try:
        data = request.get_json()
        if not data or 'user_input' not in data:
            return jsonify({"error": "缺少 user_input 參數"}), 400
        
        user_input = data['user_input']
        max_tokens = data.get('max_tokens', 16384)
        
        print(f"🧑 User: {user_input}")
        response = chatbot.generate_response(user_input, max_tokens)
        print(f"🤖 Model: {response}")
        
        return jsonify({
            "user_input": user_input,
            "response": response,
            "current_weights": chatbot.current_weights,
            "status": "success"
        })
        
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        return jsonify({"error": str(e)}), 500

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
    """取得當前狀態"""
    return jsonify({
        "model_name": chatbot.base_chatbot.model_name,
        "layer_idx": chatbot.base_chatbot.layer_idx,
        "steering_coef": chatbot.base_chatbot.steering_coef,
        "current_weights": chatbot.current_weights,
        "num_personas": len(chatbot.persona_handler.personas),
        "conversation_length": len(chatbot.base_chatbot.conversation_history)
    })

def main():
    global chatbot
    
    parser = argparse.ArgumentParser(description="Multi-Persona Chatbot API")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="模型名稱")
    parser.add_argument("--vector_paths", nargs='+', required=True, help="多個 persona 向量路徑")
    parser.add_argument("--layer", type=int, default=20, help="目標層數")
    parser.add_argument("--coef", type=float, default=2.0, help="steering 係數")
    parser.add_argument("--fusion_method", default="weighted_average", 
                       choices=["weighted_average", "concatenate", "attention", "dynamic"],
                       help="向量融合方法")
    parser.add_argument("--host", default="127.0.0.1", help="API 主機位址")
    parser.add_argument("--port", type=int, default=5000, help="API 連接埠")
    
    args = parser.parse_args()
    
    # 初始化多重 persona 聊天機器人
    print("🚀 正在初始化多重 Persona API 服務...")
    print(f"📁 載入 persona 向量: {args.vector_paths}")
    
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
    
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()

# from flask import Flask, request, jsonify
# from interactive_chat import PersonaChatbot
# import argparse
# from rich import print

# app = Flask(__name__)
# chatbot = None

# @app.route('/chat', methods=['POST'])
# def chat_api():
#     """API 端點：接收使用者輸入並回傳模型回應"""
#     try:
#         # 從請求中取得使用者輸入
#         data = request.get_json()
#         if not data or 'user_input' not in data:
#             return jsonify({"error": "缺少 user_input 參數"}), 400
        
#         user_input = data['user_input']
#         max_tokens = data.get('max_tokens', 16384)
        
#         # 使用現有的 generate_response 函式
#         print(f"🧑 User: {user_input}")
#         response = chatbot.generate_response(user_input, max_tokens)
#         print(f"🤖 Model: {response}")
        
#         # 回傳 JSON 回應
#         return jsonify({
#             "user_input": user_input,
#             "response": response,
#             "status": "success"
#         })
        
#     except Exception as e:
#         print(f"❌ 錯誤: {e}")
#         return jsonify({"error": str(e)}), 500

# @app.route('/reset', methods=['POST'])
# def reset_conversation():
#     """重設對話歷史"""
#     try:
#         chatbot.conversation_history = []
#         print("🔄 對話歷史已清除")
#         return jsonify({"status": "conversation reset"})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/settings', methods=['POST'])
# def update_settings():
#     """更新 steering 參數"""
#     try:
#         data = request.get_json()
#         layer_idx = data.get('layer_idx')
#         coef = data.get('coef')
        
#         chatbot.set_steering_params(layer_idx, coef)
        
#         return jsonify({
#             "layer_idx": chatbot.layer_idx,
#             "steering_coef": chatbot.steering_coef,
#             "status": "settings updated"
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/status', methods=['GET'])
# def get_status():
#     """取得當前狀態"""
#     return jsonify({
#         "model_name": chatbot.model_name,
#         "layer_idx": chatbot.layer_idx,
#         "steering_coef": chatbot.steering_coef,
#         "conversation_length": len(chatbot.conversation_history)
#     })

# def main():
#     global chatbot
    
#     parser = argparse.ArgumentParser(description="Persona Chatbot API")
#     parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="模型名稱")
#     parser.add_argument("--vector_path", required=True, help="persona向量路徑")
#     parser.add_argument("--layer", type=int, default=20, help="目標層數")
#     parser.add_argument("--coef", type=float, default=2.0, help="steering係數")
#     parser.add_argument("--host", default="127.0.0.1", help="API 主機位址")
#     parser.add_argument("--port", type=int, default=5000, help="API 連接埠")
    
#     args = parser.parse_args()
    
#     # 初始化聊天機器人
#     print("🚀 正在初始化 API 服務...")
#     chatbot = PersonaChatbot(
#         model_name=args.model,
#         vector_path=args.vector_path,
#         layer_idx=args.layer,
#         steering_coef=args.coef
#     )
    
#     print(f"✅ API 服務啟動於 http://{args.host}:{args.port}")
#     print("📋 可用端點:")
#     print(f"  POST /chat - 傳送訊息")
#     print(f"  POST /reset - 重設對話")
#     print(f"  POST /settings - 更新參數")
#     print(f"  GET /status - 取得狀態")
    
#     app.run(host=args.host, port=args.port, debug=False)

# if __name__ == "__main__":
#     main()