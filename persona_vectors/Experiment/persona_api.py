from flask import Flask, request, jsonify
import argparse
import re
from rich import print
from multi_persona_handler import MultiPersonaChatbot

app = Flask(__name__)
chatbot = None

# def clean_repetitive_response(text):
#     """清理重複的回應內容"""
#     if not text:
#         return text
    
#     # 先處理明顯的截斷位置（如果回應看起來不完整）
#     if len(text) > 500 and (text.endswith(',') or text.endswith(' the') or text.endswith(' of')):
#         # 找到最後一個完整句子
#         last_period = text.rfind('.')
#         last_exclamation = text.rfind('!')
#         last_question = text.rfind('?')
#         last_complete = max(last_period, last_exclamation, last_question)
        
#         if last_complete > len(text) * 0.5:  # 如果找到的句子不會太短
#             text = text[:last_complete + 1]
    
#     # 偵測重複的短語模式
#     words = text.split()
#     if len(words) < 10:
#         return text
    
#     # 檢查重複的短語（2-8個詞）
#     for pattern_length in range(2, min(8, len(words) // 4)):
#         for start in range(len(words) - pattern_length * 2):
#             pattern = words[start:start + pattern_length]
            
#             # 計算這個模式的出現次數
#             occurrences = []
#             for i in range(len(words) - pattern_length + 1):
#                 if words[i:i + pattern_length] == pattern:
#                     occurrences.append(i)
            
#             # 如果模式重複2次以上，在第二次出現處截斷
#             if len(occurrences) >= 2 and occurrences[1] - occurrences[0] == pattern_length:
#                 cutoff_index = occurrences[1]
#                 cleaned_text = ' '.join(words[:cutoff_index])
#                 # 確保以完整句子結尾
#                 if not cleaned_text.endswith(('.', '!', '?')):
#                     last_complete = max(cleaned_text.rfind('.'), 
#                                       cleaned_text.rfind('!'), 
#                                       cleaned_text.rfind('?'))
#                     if last_complete > 0:
#                         cleaned_text = cleaned_text[:last_complete + 1]
#                 return cleaned_text.strip()
    
#     # 檢查單詞重複（特別是創意寫作中的重複）
#     word_counts = {}
#     for i, word in enumerate(words):
#         if len(word) > 3:  # 只檢查較長的詞
#             if word in word_counts:
#                 word_counts[word].append(i)
#             else:
#                 word_counts[word] = [i]
    
#     # 如果某個詞出現太頻繁，可能是重複
#     for word, positions in word_counts.items():
#         if len(positions) > 5:  # 同一個詞出現超過5次
#             # 在第4次出現處截斷
#             if len(positions) >= 4:
#                 cutoff = positions[3]
#                 if cutoff < len(words) * 0.8:  # 不要截得太短
#                     truncated = ' '.join(words[:cutoff])
#                     # 找到最後的完整句子
#                     last_period = truncated.rfind('.')
#                     if last_period > len(truncated) * 0.5:
#                         return truncated[:last_period + 1]
    
#     return text

@app.route('/chat', methods=['POST'])
def chat_api():
    """API 端點：接收使用者輸入並回傳模型回應"""
    try:
        data = request.get_json()
        if not data or 'user_input' not in data:
            return jsonify({"error": "缺少 user_input 參數"}), 400
        
        user_input = data['user_input']
        max_tokens = data.get('max_tokens', 1000)  # 預設值改為更合理的1000
        max_tokens = min(max_tokens, 2048)  # 強制限制最大值
        
        print(f"🧑 User: {user_input}")
        response = chatbot.generate_response(user_input, max_tokens)
        
        # 檢查並清理重複內容
        # response = clean_repetitive_response(response)
        
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
    
    # parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="模型名稱")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="模型名稱")
    # parser.add_argument("--model", default="google/gemma-2-9b-it", help="模型名稱")
    # parser.add_argument("--model", default="google/gemma-3-4b-it", help="模型名稱")
    

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