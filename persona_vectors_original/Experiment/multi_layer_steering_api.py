from flask import Flask, request, jsonify
import argparse
import json
import os
from rich import print
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from activation_steer import ActivationSteererMultiple
import numpy as np

app = Flask(__name__)
chatbot = None

class MultiLayerPersonaChatbot:
    """
    多層級 Persona 聊天機器人
    支援不同 persona 在不同層進行 activation steering
    """
    
    def __init__(self, model_name, persona_configs, debug=False):
        """
        初始化多層級 Persona 聊天機器人
        
        Args:
            model_name: 模型名稱
            persona_configs: lista 包含每個 persona 的配置
                格式: [
                    {
                        "name": "environmentalist",
                        "vector_path": "path/to/vector.npy", 
                        "layer_idx": 15,
                        "coeff": 2.0,
                        "positions": "all"
                    },
                    ...
                ]
            debug: 是否開啟偵錯模式
        """
        self.model_name = model_name
        self.persona_configs = persona_configs
        self.debug = debug
        
        # 載入模型和 tokenizer
        print(f"🤖 正在載入模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 載入 persona 向量
        self.personas = {}
        self.steering_instructions = []
        
        for config in persona_configs:
            name = config["name"]
            vector_path = config["vector_path"]
            
            print(f"📁 載入 {name} 向量: {vector_path}")
            
            if not os.path.exists(vector_path):
                raise FileNotFoundError(f"向量檔案不存在: {vector_path}")
            
            # 支援多種檔案格式
            if vector_path.endswith('.pt'):
                vector = torch.load(vector_path, map_location='cpu')
                if isinstance(vector, torch.Tensor):
                    vector = vector.numpy()
                else:
                    raise ValueError(f"PyTorch 檔案 {vector_path} 不包含 tensor")
            elif vector_path.endswith('.npy'):
                vector = np.load(vector_path)
            elif vector_path.endswith('.npz'):
                npz_file = np.load(vector_path)
                # 假設 npz 檔案只有一個陣列，取第一個
                vector = npz_file[list(npz_file.keys())[0]]
            else:
                raise ValueError(f"不支援的檔案格式: {vector_path}")
            self.personas[name] = {
                "vector": vector,
                "config": config
            }
            
            # 建立 steering 指令
            instruction = {
                "steering_vector": vector,
                "layer_idx": config.get("layer_idx", 20),
                "coeff": config.get("coeff", 0.0),  # 預設為 0，可動態調整
                "positions": config.get("positions", "all")
            }
            self.steering_instructions.append(instruction)
        
        # 對話歷史
        self.conversation_history = []
        
        print(f"✅ 成功載入 {len(self.personas)} 個 persona")
        self._print_persona_info()
    
    def _print_persona_info(self):
        """列印 persona 資訊"""
        print("\n📋 Persona 配置資訊:")
        for name, persona in self.personas.items():
            config = persona["config"]
            print(f"  • {name}: 層 {config.get('layer_idx', 20)}, "
                  f"係數 {config.get('coeff', 0.0)}, "
                  f"位置 {config.get('positions', 'all')}")
    
    def update_persona_weights(self, weights):
        """
        更新 persona 權重（係數）
        
        Args:
            weights: dict, 格式 {"persona_name": coeff, ...}
        """
        for i, (name, persona) in enumerate(self.personas.items()):
            if name in weights:
                new_coeff = float(weights[name])
                self.steering_instructions[i]["coeff"] = new_coeff
                persona["config"]["coeff"] = new_coeff
                
        if self.debug:
            print(f"📊 更新權重: {weights}")
    
    def get_current_weights(self):
        """取得當前權重"""
        weights = {}
        for name, persona in self.personas.items():
            weights[name] = persona["config"]["coeff"]
        return weights
    
    def set_persona_mode(self, mode):
        """
        設定 persona 模式
        
        Args:
            mode: 預設模式名稱 (single, balanced, creative, analytical 等)
        """
        if mode == "single_environmentalist":
            self.update_persona_weights({"environmentalist": 2.0, "creative": 0.0})
        elif mode == "single_creative":
            self.update_persona_weights({"environmentalist": 0.0, "creative": 2.0})
        elif mode == "balanced":
            self.update_persona_weights({"environmentalist": 1.0, "creative": 1.0})
        elif mode == "creative_focus":
            self.update_persona_weights({"environmentalist": 0.5, "creative": 2.0})
        elif mode == "analytical":
            self.update_persona_weights({"environmentalist": 2.0, "creative": 0.5})
        elif mode == "off":
            # 關閉所有 steering
            weights = {name: 0.0 for name in self.personas.keys()}
            self.update_persona_weights(weights)
        else:
            raise ValueError(f"未知模式: {mode}")
        
        if self.debug:
            print(f"🎭 設定模式: {mode}")
    
    def generate_response(self, user_input, max_tokens=1000):
        """生成回應"""
        # 添加到對話歷史
        self.conversation_history.append(f"Human: {user_input}")
        
        # 建構 prompt
        if len(self.conversation_history) > 10:  # 限制歷史長度
            recent_history = self.conversation_history[-10:]
        else:
            recent_history = self.conversation_history
        
        prompt = "\n".join(recent_history) + "\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048
        ).to(self.model.device)
        
        # 使用 multi-layer steering
        with ActivationSteererMultiple(
            self.model, 
            self.steering_instructions, 
            debug=self.debug
        ):
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
        
        # 解碼回應
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # 添加到對話歷史
        self.conversation_history.append(f"Assistant: {response}")
        
        return response
    
    def reset_conversation(self):
        """重設對話歷史"""
        self.conversation_history = []
        if self.debug:
            print("🔄 對話歷史已重設")

# Flask API 端點
@app.route('/chat', methods=['POST'])
def chat_api():
    """API 端點：接收使用者輸入並回傳模型回應"""
    try:
        data = request.get_json()
        if not data or 'user_input' not in data:
            return jsonify({"error": "缺少 user_input 參數"}), 400
        
        user_input = data['user_input']
        max_tokens = data.get('max_tokens', 1000)
        max_tokens = min(max_tokens, 2048)  # 限制最大值
        
        print(f"🧑 User: {user_input}")
        response = chatbot.generate_response(user_input, max_tokens)
        print(f"🤖 Model: {response}")
        
        return jsonify({
            "user_input": user_input,
            "response": response,
            "current_weights": chatbot.get_current_weights(),
            "persona_configs": {name: persona["config"] for name, persona in chatbot.personas.items()},
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
            "current_weights": chatbot.get_current_weights(),
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
            "current_weights": chatbot.get_current_weights(),
            "status": "mode updated"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/update_layer_config', methods=['POST'])
def update_layer_config():
    """更新特定 persona 的層配置"""
    try:
        data = request.get_json()
        persona_name = data.get('persona_name')
        new_layer = data.get('layer_idx')
        
        if not persona_name or new_layer is None:
            return jsonify({"error": "缺少 persona_name 或 layer_idx 參數"}), 400
        
        if persona_name not in chatbot.personas:
            return jsonify({"error": f"未找到 persona: {persona_name}"}), 400
        
        # 找到對應的指令索引
        for i, (name, _) in enumerate(chatbot.personas.items()):
            if name == persona_name:
                chatbot.steering_instructions[i]["layer_idx"] = new_layer
                chatbot.personas[name]["config"]["layer_idx"] = new_layer
                break
        
        return jsonify({
            "persona_name": persona_name,
            "new_layer": new_layer,
            "persona_configs": {name: persona["config"] for name, persona in chatbot.personas.items()},
            "status": "layer updated"
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
    persona_info = {}
    for name, persona in chatbot.personas.items():
        config = persona["config"]
        persona_info[name] = {
            "layer_idx": config.get("layer_idx", 20),
            "coeff": config.get("coeff", 0.0),
            "positions": config.get("positions", "all"),
            "vector_shape": persona["vector"].shape
        }
    
    return jsonify({
        "model_name": chatbot.model_name,
        "num_personas": len(chatbot.personas),
        "persona_info": persona_info,
        "current_weights": chatbot.get_current_weights(),
        "conversation_length": len(chatbot.conversation_history)
    })

@app.route('/available_modes', methods=['GET'])
def get_available_modes():
    """取得可用的模式"""
    modes = [
        "single_environmentalist",
        "single_creative", 
        "balanced",
        "creative_focus",
        "analytical",
        "off"
    ]
    return jsonify({"available_modes": modes})

def main():
    global chatbot
    
    parser = argparse.ArgumentParser(description="Multi-Layer Persona Steering API")
    
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="模型名稱")
    parser.add_argument("--config", required=True, help="Persona 配置檔案 (JSON)")
    parser.add_argument("--host", default="127.0.0.1", help="API 主機位址")
    parser.add_argument("--port", type=int, default=5000, help="API 連接埠")
    parser.add_argument("--debug", action="store_true", help="開啟偵錯模式")
    
    args = parser.parse_args()
    
    # 載入 persona 配置
    if not os.path.exists(args.config):
        print(f"❌ 配置檔案不存在: {args.config}")
        return
    
    with open(args.config, 'r', encoding='utf-8') as f:
        persona_configs = json.load(f)
    
    # 初始化多層級 persona 聊天機器人
    print("🚀 正在初始化多層級 Persona Steering API 服務...")
    print(f"📁 載入 persona 配置: {args.config}")
    
    chatbot = MultiLayerPersonaChatbot(
        model_name=args.model,
        persona_configs=persona_configs,
        debug=args.debug
    )
    
    print(f"✅ API 服務啟動於 http://{args.host}:{args.port}")
    print("📋 可用端點:")
    print(f"  POST /chat - 傳送訊息")
    print(f"  POST /set_persona_weights - 設定權重")
    print(f"  POST /set_persona_mode - 設定預設模式")
    print(f"  POST /update_layer_config - 更新層配置")
    print(f"  POST /reset - 重設對話")
    print(f"  GET /status - 取得狀態")
    print(f"  GET /available_modes - 取得可用模式")
    
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()