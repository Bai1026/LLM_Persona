#!/bin/bash
"""
Gemma API 啟動腳本
"""

# 設定環境變數
export GEMMA_API_HOST="0.0.0.0"
export GEMMA_API_PORT="8002"

# 啟動 API 服務
echo "🚀 啟動 Gemma API 服務..."
echo "📡 服務位址: http://${GEMMA_API_HOST}:${GEMMA_API_PORT}"

python gemma_api.py
