from rich import print
from flask import Flask, request, jsonify
from datetime import datetime
import json
import traceback
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import os
import re
from utils.log import colored_print

from self_play import discuss

load_dotenv(find_dotenv())

app = Flask(__name__)

# 初始化 OpenAI 客戶端
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
)

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    簡單的 OpenAI 相容 API 端點，純粹用於測試
    """
    try:
        # 解析請求資料
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": {
                    "message": "請求主體必須是有效的 JSON",
                    "type": "invalid_request_error"
                }
            }), 400
        
        # 驗證必要欄位
        if 'messages' not in data:
            return jsonify({
                "error": {
                    "message": "缺少必要欄位: messages",
                    "type": "invalid_request_error"
                }
            }), 400
        
        messages = data['messages']
        model = data.get('model', 'gpt-4o-mini')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 1000)
        
        print(f'收到 API 請求 - 模型: {model}, 訊息數量: {len(messages)}')
        print(f'訊息內容: {json.dumps(messages, ensure_ascii=False, indent=2)}')

        # 簡單的回應生成 - 直接呼叫 OpenAI API
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # 使用固定模型進行測試
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            generator_output = response.choices[0].message.content
            # generator_output, log = discuss(messages)
            
        except Exception as openai_error:
            print(f'OpenAI API 錯誤: {str(openai_error)}')
        
        # return generator_output
        # 建立 OpenAI 格式的回應
        api_response = {
            "id": f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "object": "chat.completion", 
            "created": int(datetime.now().timestamp()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generator_output
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": estimate_token_count(messages),
                "completion_tokens": estimate_token_count([{"role": "assistant", "content": generator_output}]),
                "total_tokens": estimate_token_count(messages) + estimate_token_count([{"role": "assistant", "content": generator_output}])
            }
        }
        
        print(f'API 回應產生完成 - 長度: {len(generator_output)} 字元')
        
        return jsonify(api_response)
        
    except Exception as e:
        print(f'API 錯誤: {str(e)}')
        print(f'錯誤追蹤: {traceback.format_exc()}')
        
        return jsonify({
            "error": {
                "message": f"內部伺服器錯誤: {str(e)}",
                "type": "internal_server_error"
            }
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康檢查端點"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Simple Test API"
    })

@app.route('/models', methods=['GET'])
def list_models():
    """列出可用模型（OpenAI 相容）"""
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "test-model-v1",
                "object": "model", 
                "created": int(datetime.now().timestamp()),
                "owned_by": "custom"
            }
        ]
    })

def estimate_token_count(messages):
    """
    簡單的 token 數量估計函式
    """
    total_chars = sum(len(msg.get('content', '')) for msg in messages)
    # 粗略估計：1 token ≈ 4 字元（中文可能更少）
    return max(1, total_chars // 3)

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": {
            "message": "找不到請求的端點",
            "type": "not_found_error"
        }
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": {
            "message": "內部伺服器錯誤",
            "type": "internal_server_error"
        }
    }), 500

if __name__ == '__main__':
    print('🚀 啟動簡單測試 API 伺服器...')
    print('📡 API 端點: http://localhost:6969/v1/chat/completions')
    print('💊 健康檢查: http://localhost:6969/health')
    print('📋 模型列表: http://localhost:6969/models')
    
    app.run(
        host='0.0.0.0',
        port=6969,
        debug=True,
        threaded=True
    )