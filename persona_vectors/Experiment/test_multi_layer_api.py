#!/usr/bin/env python3
"""
Multi-Layer Persona Steering API 使用範例
展示如何使用 API 進行多層級 persona steering
"""

import requests
import json
import time

# API 基礎 URL
BASE_URL = "http://127.0.0.1:5000"

def test_api():
    """測試 API 功能"""
    
    print("🧪 測試多層級 Persona Steering API")
    print("=" * 50)
    
    # 1. 檢查狀態
    print("\n1️⃣ 檢查 API 狀態...")
    try:
        response = requests.get(f"{BASE_URL}/status")
        if response.status_code == 200:
            status = response.json()
            print("✅ API 狀態正常")
            print(f"   模型: {status['model_name']}")
            print(f"   Persona 數量: {status['num_personas']}")
            print("   Persona 資訊:")
            for name, info in status['persona_info'].items():
                print(f"     • {name}: 層 {info['layer_idx']}, 係數 {info['coeff']}")
        else:
            print(f"❌ API 無法連接: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ 連接錯誤: {e}")
        return
    
    # 2. 測試基本對話
    print("\n2️⃣ 測試基本對話...")
    test_message = "What are the key challenges facing humanity in the next 50 years?"
    
    chat_data = {
        "user_input": test_message,
        "max_tokens": 300
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=chat_data)
    if response.status_code == 200:
        result = response.json()
        print("✅ 對話成功")
        print(f"🤖 回應: {result['response'][:200]}...")
        print(f"📊 當前權重: {result['current_weights']}")
    else:
        print(f"❌ 對話失敗: {response.status_code}")
        return
    
    # 3. 測試權重調整
    print("\n3️⃣ 測試權重調整...")
    
    # 設定環保主義者權重較高
    weights_data = {
        "weights": {
            "environmentalist": 3.0,
            "creative": 0.5
        }
    }
    
    response = requests.post(f"{BASE_URL}/set_persona_weights", json=weights_data)
    if response.status_code == 200:
        result = response.json()
        print("✅ 權重調整成功")
        print(f"📊 新權重: {result['current_weights']}")
        
        # 用同樣問題測試差異
        response = requests.post(f"{BASE_URL}/chat", json=chat_data)
        if response.status_code == 200:
            result = response.json()
            print(f"🤖 環保重點回應: {result['response'][:200]}...")
    
    # 4. 測試預設模式
    print("\n4️⃣ 測試預設模式...")
    
    mode_data = {"mode": "creative_focus"}
    response = requests.post(f"{BASE_URL}/set_persona_mode", json=mode_data)
    if response.status_code == 200:
        result = response.json()
        print("✅ 模式設定成功")
        print(f"📊 創意模式權重: {result['current_weights']}")
        
        # 測試創意回應
        creative_question = "Describe a futuristic city where art and nature coexist perfectly."
        chat_data["user_input"] = creative_question
        
        response = requests.post(f"{BASE_URL}/chat", json=chat_data)
        if response.status_code == 200:
            result = response.json()
            print(f"🎨 創意回應: {result['response'][:200]}...")
    
    # 5. 測試層配置更新
    print("\n5️⃣ 測試動態層配置...")
    
    layer_data = {
        "persona_name": "creative",
        "layer_idx": 25
    }
    
    response = requests.post(f"{BASE_URL}/update_layer_config", json=layer_data)
    if response.status_code == 200:
        result = response.json()
        print("✅ 層配置更新成功")
        print(f"🔧 {layer_data['persona_name']} 已移至第 {layer_data['layer_idx']} 層")
    
    # 6. 查看可用模式
    print("\n6️⃣ 查看可用模式...")
    
    response = requests.get(f"{BASE_URL}/available_modes")
    if response.status_code == 200:
        modes = response.json()
        print("✅ 可用模式:")
        for mode in modes['available_modes']:
            print(f"   • {mode}")
    
    # 7. 重設對話
    print("\n7️⃣ 重設對話...")
    
    response = requests.post(f"{BASE_URL}/reset")
    if response.status_code == 200:
        print("✅ 對話已重設")
    
    print("\n✨ API 測試完成！")

def interactive_chat():
    """互動式聊天"""
    
    print("🤖 多層級 Persona 聊天模式")
    print("輸入 'quit' 結束，輸入 'help' 查看指令")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n🧑 You: ")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'help':
                print_help()
                continue
            elif user_input.lower().startswith('mode '):
                mode = user_input.split(' ', 1)[1]
                set_mode(mode)
                continue
            elif user_input.lower().startswith('weight '):
                # 格式: weight persona_name value
                parts = user_input.split(' ')
                if len(parts) == 3:
                    persona_name, weight = parts[1], float(parts[2])
                    set_weight(persona_name, weight)
                else:
                    print("❌ 格式錯誤。使用: weight <persona_name> <value>")
                continue
            elif user_input.lower() == 'status':
                show_status()
                continue
            
            # 發送聊天請求
            chat_data = {
                "user_input": user_input,
                "max_tokens": 500
            }
            
            response = requests.post(f"{BASE_URL}/chat", json=chat_data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n🤖 Assistant: {result['response']}")
                print(f"📊 當前權重: {result['current_weights']}")
            else:
                print(f"❌ 請求失敗: {response.status_code}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ 錯誤: {e}")
    
    print("\n👋 再見！")

def print_help():
    """列印幫助資訊"""
    print("""
🔧 可用指令:
  help              - 顯示此幫助
  status            - 顯示當前狀態
  mode <mode_name>  - 設定模式 (balanced, creative_focus, off 等)
  weight <persona> <value> - 設定特定 persona 權重
  quit              - 結束程式

💡 範例:
  mode balanced
  weight environmentalist 2.5
  weight creative 0.5
""")

def set_mode(mode):
    """設定模式"""
    try:
        response = requests.post(f"{BASE_URL}/set_persona_mode", json={"mode": mode})
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 模式已設為 '{mode}'")
            print(f"📊 新權重: {result['current_weights']}")
        else:
            print(f"❌ 模式設定失敗: {response.status_code}")
    except Exception as e:
        print(f"❌ 錯誤: {e}")

def set_weight(persona_name, weight):
    """設定特定 persona 權重"""
    try:
        weights = {persona_name: weight}
        response = requests.post(f"{BASE_URL}/set_persona_weights", json={"weights": weights})
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {persona_name} 權重已設為 {weight}")
            print(f"📊 當前權重: {result['current_weights']}")
        else:
            print(f"❌ 權重設定失敗: {response.status_code}")
    except Exception as e:
        print(f"❌ 錯誤: {e}")

def show_status():
    """顯示狀態"""
    try:
        response = requests.get(f"{BASE_URL}/status")
        if response.status_code == 200:
            status = response.json()
            print(f"\n📊 當前狀態:")
            print(f"   模型: {status['model_name']}")
            print(f"   對話長度: {status['conversation_length']}")
            print(f"   Persona 資訊:")
            for name, info in status['persona_info'].items():
                print(f"     • {name}: 層 {info['layer_idx']}, 係數 {info['coeff']}, 位置 {info['positions']}")
        else:
            print(f"❌ 狀態查詢失敗: {response.status_code}")
    except Exception as e:
        print(f"❌ 錯誤: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_api()
    else:
        interactive_chat()