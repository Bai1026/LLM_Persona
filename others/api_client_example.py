#!/usr/bin/env python3
"""
純粹模型 API 使用範例
"""

import requests
import json
import time

class ModelAPIClient:
    def __init__(self, base_url: str = "http://127.0.0.1:5000"):
        self.base_url = base_url
    
    def chat(self, message: str, max_tokens: int = 512, temperature: float = 0.7):
        """傳送聊天訊息"""
        url = f"{self.base_url}/chat"
        data = {
            "message": message,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(url, json=data)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def reset(self):
        """重設對話"""
        url = f"{self.base_url}/reset"
        try:
            response = requests.post(url)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def status(self):
        """取得狀態"""
        url = f"{self.base_url}/status"
        try:
            response = requests.get(url)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def health(self):
        """健康檢查"""
        url = f"{self.base_url}/health"
        try:
            response = requests.get(url)
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def main():
    # 建立客戶端
    client = ModelAPIClient()
    
    # 健康檢查
    print("🔍 檢查 API 狀態...")
    health = client.health()
    print(f"健康狀態: {health}")
    
    if not health.get("model_loaded"):
        print("❌ 模型未載入，請先啟動 API 服務")
        return
    
    # 取得狀態
    status = client.status()
    print(f"\n📊 模型狀態:")
    print(f"  模型: {status.get('model_name')}")
    print(f"  裝置: {status.get('device')}")
    print(f"  對話長度: {status.get('conversation_length')}")
    
    # 互動式聊天
    print("\n💬 開始聊天 (輸入 'quit' 結束, 'reset' 重設對話, 'status' 查看狀態)")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n🧑 你: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 再見！")
                break
            elif user_input.lower() == 'reset':
                result = client.reset()
                print(f"🔄 {result}")
                continue
            elif user_input.lower() == 'status':
                status = client.status()
                print(f"📊 狀態: {status}")
                continue
            elif not user_input:
                continue
            
            # 傳送訊息
            print("🤖 思考中...")
            response = client.chat(user_input)
            
            if "error" in response:
                print(f"❌ 錯誤: {response['error']}")
            else:
                print(f"🤖 助手: {response['response']}")
                
        except KeyboardInterrupt:
            print("\n👋 再見！")
            break
        except Exception as e:
            print(f"❌ 錯誤: {e}")

if __name__ == "__main__":
    main()
