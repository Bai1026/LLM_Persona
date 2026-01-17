#!/bin/bash

# 純粹模型 API 啟動腳本

echo "🚀 純粹模型 API 啟動腳本"
echo "========================"

# 預設參數
MODEL="Qwen/Qwen2.5-7B-Instruct"
DEVICE="auto"
HOST="127.0.0.1"
PORT="5003"

# 解析命令列參數
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
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
        --llama)
            MODEL="meta-llama/Llama-3.1-8B-Instruct"
            shift
            ;;
        --qwen)
            MODEL="Qwen/Qwen2.5-7B-Instruct"
            shift
            ;;
        --help|-h)
            echo "使用方法: $0 [選項]"
            echo "選項:"
            echo "  --model MODEL     指定模型名稱"
            echo "  --device DEVICE   指定裝置 (auto, cpu, cuda)"
            echo "  --host HOST       指定主機位址 (預設: 127.0.0.1)"
            echo "  --port PORT       指定連接埠 (預設: 5000)"
            echo "  --llama           使用 Llama-3.1-8B-Instruct"
            echo "  --qwen            使用 Qwen2.5-7B-Instruct"
            echo "  --help, -h        顯示此說明"
            exit 0
            ;;
        *)
            echo "未知參數: $1"
            echo "使用 --help 取得說明"
            exit 1
            ;;
    esac
done

echo "📱 模型: $MODEL"
echo "🔧 裝置: $DEVICE"
echo "🌐 主機: $HOST"
echo "🔌 連接埠: $PORT"
echo ""

# 檢查 Python 檔案是否存在
if [ ! -f "pure_model_api.py" ]; then
    echo "❌ 找不到 pure_model_api.py"
    exit 1
fi

# 啟動 API
echo "🚀 正在啟動 API..."
python pure_model_api.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --host "$HOST" \
    --port "$PORT"
