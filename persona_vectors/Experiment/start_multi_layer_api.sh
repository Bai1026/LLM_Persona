#!/bin/bash

# Multi-Layer Persona Steering API 啟動腳本

echo "🚀 啟動多層級 Persona Steering API"
echo "=================================="

# 預設參數
MODEL="meta-llama/Llama-3.1-8B-Instruct"
CONFIG="persona_config_example.json"
HOST="127.0.0.1"
PORT=5000
DEBUG=false

# 解析命令列參數
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --help|-h)
            echo "使用方法: $0 [選項]"
            echo ""
            echo "選項:"
            echo "  --model MODEL     指定模型名稱 (預設: meta-llama/Llama-3.1-8B-Instruct)"
            echo "  --config CONFIG   指定配置檔案 (預設: persona_config_example.json)"
            echo "  --host HOST       指定主機位址 (預設: 127.0.0.1)"
            echo "  --port PORT       指定連接埠 (預設: 5000)"
            echo "  --debug           開啟偵錯模式"
            echo "  --help, -h        顯示此幫助"
            echo ""
            echo "範例:"
            echo "  $0 --model google/gemma-2-9b-it --port 5001"
            echo "  $0 --config my_config.json --debug"
            exit 0
            ;;
        *)
            echo "未知參數: $1"
            echo "使用 --help 查看幫助"
            exit 1
            ;;
    esac
done

echo "📋 啟動參數:"
echo "   模型: $MODEL"
echo "   配置: $CONFIG"
echo "   主機: $HOST"
echo "   連接埠: $PORT"
echo "   偵錯: $DEBUG"

# 檢查配置檔案
if [ ! -f "$CONFIG" ]; then
    echo "❌ 錯誤: 配置檔案不存在: $CONFIG"
    echo "請建立配置檔案或使用 --config 指定正確路徑"
    exit 1
fi

echo ""
echo "⏳ 正在啟動服務..."

# 建構 Python 指令
CMD="python multi_layer_steering_api.py"
CMD="$CMD --model $MODEL"
CMD="$CMD --config $CONFIG"
CMD="$CMD --host $HOST"
CMD="$CMD --port $PORT"

if [ "$DEBUG" = true ]; then
    CMD="$CMD --debug"
fi

echo "🔧 執行指令: $CMD"
echo ""

# 執行 API 服務
exec $CMD