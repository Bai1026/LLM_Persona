from flask import Flask, request, jsonify
import argparse
import re
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
        max_tokens = data.get('max_tokens', 1000)  # 預設值改為更合理的1000
        # TODO: change the max_tokens value for llm discussion
        # max_tokens = min(max_tokens, 2048)  # 強制限制最大值
        max_tokens = min(max_tokens, 8192)  # 強制限制最大值
        
        print(f"🧑 User: {user_input}")

        # 🔧 在產生回應前清理記憶體
        import torch
        torch.cuda.empty_cache()

        response = chatbot.generate_response(user_input, max_tokens)
        
        # Role Prompt
        ROLE_PROMPT = False
        role_prompt = """
You need to think and answer this question from three different professional perspectives:

1. Environmentalist:
Specialty: Sustainability and Environmental Health
Mission: Advocate for eco-friendly solutions, promote sustainable development and protect the planet. Guide us to consider the environmental impact of ideas, promoting innovations that contribute to planetary health.

2. Creative Professional:
Specialty: Aesthetics, Narratives, and Emotions
Mission: With artistic sensibility and mastery of narrative and emotion, infuse projects with beauty and depth. Challenge us to think expressively, ensuring solutions not only solve problems but also resonate on a human level.

3. Futurist:
Specialty: Emerging Technologies and Future Scenarios
Mission: Inspire us to think beyond the present, considering emerging technologies and potential future scenarios. Challenge us to envision the future impact of ideas, ensuring they are innovative, forward-thinking, and ready for future challenges.

Please provide answers from these three role perspectives, with each role embodying their professional characteristics and thinking approaches.
"""
        if ROLE_PROMPT:
            user_input += "\n\n" + role_prompt
        else:
            user_input = user_input

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
        # 🔧 錯誤時也要清理記憶體
        import torch
        torch.cuda.empty_cache()
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
        # "conversation_length": len(chatbot.base_chatbot.conversation_history)
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
