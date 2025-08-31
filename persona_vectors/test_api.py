import requests
import json
from rich import print

class ChatbotClient:
    def __init__(self, base_url="http://127.0.0.1:5000"):
        self.base_url = base_url
    
    def send_message(self, user_input, max_tokens=1000):
        """傳送訊息給聊天機器人"""
        url = f"{self.base_url}/chat"
        data = {
            "user_input": user_input,
            "max_tokens": max_tokens
        }
        
        response = requests.post(url, json=data)
        return response.json()
    
    def reset_conversation(self):
        """重設對話歷史"""
        url = f"{self.base_url}/reset"
        response = requests.post(url)
        return response.json()
    
    def update_settings(self, layer_idx=None, coef=None):
        """更新設定"""
        url = f"{self.base_url}/settings"
        data = {}
        if layer_idx is not None:
            data["layer_idx"] = layer_idx
        if coef is not None:
            data["coef"] = coef
        
        response = requests.post(url, json=data)
        return response.json()
    
    def get_status(self):
        """取得狀態"""
        url = f"{self.base_url}/status"
        response = requests.get(url)
        return response.json()

# 使用範例
if __name__ == "__main__":
    client = ChatbotClient()
    
    # 傳送訊息
    result = client.send_message("What are some creative use for fork? The goal is to come up with creative ideas")
    print(f"回應: {result['response']}")
    
    # 查看狀態
    status = client.get_status()
    print(f"狀態: {status}")