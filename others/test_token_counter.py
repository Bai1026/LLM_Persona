#!/usr/bin/env python3
"""
Token 計算器使用範例
"""

import json
from token_counter import TokenCounter, TokenStats

def create_sample_simple_format():
    """建立簡單格式的範例資料"""
    return [
        {
            "item": "Fork",
            "PersonaAPI": [
                {
                    "role": "user",
                    "content": "Please provide 5 innovative and original uses for 'Fork'. You are working on a creative task; as a result, answer as diversely and creatively as you can."
                },
                {
                    "role": "assistant",
                    "content": "Certainly! Let's reimagine the concept of \"fork\" in five entirely new, innovative, and unexpected ways:\n\n1. **Solar-Powered Fork with Urban Forests**: Imagine a fork that houses a miniature urban forest...\n\n2. **Dance Fork**: A fork that transforms into a dance floor..."
                }
            ]
        }
    ]

def create_sample_discussion_format():
    """建立 discussion 格式的範例資料"""
    return {
        "What are some creative use for Fork?": {
            "QWEN Agent 6 - Environmentalist": [
                {
                    "role": "user",
                    "content": "You are an Environmentalist. Initiate a discussion about creative uses for Fork."
                },
                {
                    "role": "assistant", 
                    "content": "As an Environmentalist, I believe we should think about sustainable uses for forks..."
                }
            ],
            "QWEN Agent 4 - Creative Professional": [
                {
                    "role": "user",
                    "content": "You are a Creative Professional. What are creative uses for forks?"
                },
                {
                    "role": "assistant",
                    "content": "As a Creative Professional, I see forks as artistic tools..."
                }
            ]
        }
    }

def test_token_counter():
    """測試 token 計算器"""
    print("🧪 測試 Token 計算器")
    print("="*50)
    
    counter = TokenCounter()
    
    # 測試簡單文字
    test_text = "Hello, this is a test message."
    tokens = counter.count_tokens(test_text)
    print(f"📝 測試文字: '{test_text}'")
    print(f"🔢 Token 數量: {tokens}")
    print()
    
    # 測試簡單格式
    print("📊 測試簡單格式:")
    simple_data = create_sample_simple_format()
    simple_stats = counter.analyze_simple_format(simple_data)
    print(f"   Input: {simple_stats.input_tokens}, Output: {simple_stats.output_tokens}")
    print()
    
    # 測試 discussion 格式
    print("📊 測試 Discussion 格式:")
    discussion_data = create_sample_discussion_format()
    discussion_stats = counter.analyze_discussion_format(discussion_data)
    print(f"   Input: {discussion_stats.input_tokens}, Output: {discussion_stats.output_tokens}")
    print()
    
    # 格式檢測
    print("🔍 格式檢測測試:")
    print(f"   簡單格式檢測: {counter.detect_format(simple_data)}")
    print(f"   Discussion 格式檢測: {counter.detect_format(discussion_data)}")

if __name__ == "__main__":
    test_token_counter()
