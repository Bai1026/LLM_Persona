#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI API 金鑰測試程式碼
用於測試您的 API 金鑰是否有效以及可用的模型
"""

import json
import sys
from openai import OpenAI

# 設定編碼
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def load_config():
    """載入設定檔"""
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ 找不到 config.json 檔案！")
        return None
    except json.JSONDecodeError:
        print("❌ config.json 格式錯誤！")
        return None

def test_api_connection(config):
    """測試 API 連線"""
    print("🔗 測試 API 連線...")
    
    try:
        # 建立客戶端
        client = OpenAI(
            api_key=config['api_key'],
            base_url=config['base_url']
        )
        
        # 測試簡單的對話完成 - 使用英文避免編碼問題
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # OpenAI 的模型
            messages=[
                {"role": "user", "content": "Hello! Please reply with 'API test successful'"}
            ],
            max_tokens=50
        )
        
        print("✅ API 連線成功！")
        print(f"📝 回應：{response.choices[0].message.content}")
        print(f"💰 使用的 tokens：{response.usage.total_tokens}")
        return True
        
    except Exception as e:
        print(f"❌ API 連線失敗：{str(e)}")
        return False

def test_embedding_api(config):
    """測試嵌入 API"""
    print("\n🔗 測試嵌入 API...")
    
    try:
        # 建立客戶端
        client = OpenAI(
            api_key=config['embedding_api_key'],
            base_url=config['embedding_base_url']
        )
        
        # 測試文字嵌入 - 使用英文
        response = client.embeddings.create(
            model=config['embedding_model'],  # text-embedding-ada-002
            input="This is a test text for embedding"
        )
        
        embedding_vector = response.data[0].embedding
        print("✅ 嵌入 API 連線成功！")
        print(f"📊 嵌入向量維度：{len(embedding_vector)}")
        print(f"💰 使用的 tokens：{response.usage.total_tokens}")
        return True
        
    except Exception as e:
        print(f"❌ 嵌入 API 連線失敗：{str(e)}")
        return False

def list_available_models(config):
    """列出可用的模型"""
    print("\n📋 列出可用模型...")
    
    try:
        client = OpenAI(
            api_key=config['api_key'],
            base_url=config['base_url']
        )
        
        models = client.models.list()
        print("✅ 可用模型清單：")
        
        # 篩選常用的模型
        common_models = []
        for model in models.data:
            if any(keyword in model.id.lower() for keyword in ['gpt-4', 'gpt-3.5', 'embedding']):
                common_models.append(model.id)
        
        for model in common_models[:10]:
            print(f"   • {model}")
        
        if len(common_models) > 10:
            print(f"   ... 還有 {len(common_models) - 10} 個相關模型")
            
        return True
        
    except Exception as e:
        print(f"❌ 無法取得模型清單：{str(e)}")
        return False

def test_account_info(config):
    """測試帳戶資訊（如果支援）"""
    print("\n💳 檢查帳戶資訊...")
    
    try:
        client = OpenAI(
            api_key=config['api_key'],
            base_url=config['base_url']
        )
        
        # 嘗試取得帳戶資訊（某些 API 端點可能不支援）
        models = client.models.list()
        print(f"✅ API 金鑰有效，可存取 {len(models.data)} 個模型")
        return True
        
    except Exception as e:
        print(f"❌ 無法取得帳戶資訊：{str(e)}")
        return False

def main():
    """主函式"""
    print("🚀 開始測試 OpenAI API 金鑰...")
    print("=" * 50)
    
    # 載入設定
    config = load_config()
    if not config:
        return
    
    # 顯示設定資訊（隱藏 API 金鑰）
    print(f"📍 Base URL: {config['base_url']}")
    api_key_display = config['api_key'][:20] + "..." if len(config['api_key']) > 20 else config['api_key']
    print(f"🔑 API Key: {api_key_display}")
    print(f"📦 Embedding Model: {config.get('embedding_model', 'N/A')}")
    print("=" * 50)
    
    # 執行測試
    tests_passed = 0
    total_tests = 4
    
    # 測試 1: 基本 API 連線
    if test_api_connection(config):
        tests_passed += 1
    
    # 測試 2: 嵌入 API
    if test_embedding_api(config):
        tests_passed += 1
    
    # 測試 3: 模型清單
    if list_available_models(config):
        tests_passed += 1
    
    # 測試 4: 帳戶資訊
    if test_account_info(config):
        tests_passed += 1
    
    # 顯示結果
    print("\n" + "=" * 50)
    print(f"📊 測試結果：{tests_passed}/{total_tests} 通過")
    
    if tests_passed == total_tests:
        print("🎉 所有測試通過！您的 OpenAI API 設定正確。")
        print("💡 建議：請確保您的帳戶有足夠的信用額度。")
    elif tests_passed > 0:
        print("⚠️  部分測試通過，請檢查失敗的項目。")
    else:
        print("💥 所有測試失敗，請檢查您的 OpenAI API 金鑰。")
        print("💡 建議：")
        print("   1. 確認 API 金鑰是否正確")
        print("   2. 檢查帳戶餘額")
        print("   3. 確認 API 金鑰權限")

if __name__ == "__main__":
    main()