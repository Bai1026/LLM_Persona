# 純粹模型 API 服務

這是一個不使用 vector 的純粹 Qwen/Llama 模型 API 服務。

## 功能特色

- 🚀 支援 Qwen 和 Llama 模型
- 💬 RESTful API 介面
- 🔄 對話歷史管理
- 📊 狀態監控
- 🔧 可調整參數

## 安裝相依套件

```bash
pip install -r requirements_pure_api.txt
```

## 快速開始

### 方法 1: 使用啟動腳本

```bash
# 使用 Qwen 模型
./start_api.sh --qwen

# 使用 Llama 模型
./start_api.sh --llama

# 自定義參數
./start_api.sh --model "your-model" --port 8080
```

### 方法 2: 直接執行

```bash
# 使用 Qwen
python pure_model_api.py --model "Qwen/Qwen2.5-7B-Instruct"

# 使用 Llama
python pure_model_api.py --model "meta-llama/Llama-3.1-8B-Instruct"

# 自定義參數
python pure_model_api.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --device "cuda:0" \
    --host "0.0.0.0" \
    --port 5000
```

## API 端點

### 1. 聊天 (POST /chat)

```bash
curl -X POST http://127.0.0.1:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "你好！",
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

回應：

```json
{
  "message": "你好！",
  "response": "你好！很高興見到你！有什麼我可以幫助你的嗎？",
  "status": "success"
}
```

### 2. 重設對話 (POST /reset)

```bash
curl -X POST http://127.0.0.1:5000/reset
```

### 3. 取得狀態 (GET /status)

```bash
curl http://127.0.0.1:5000/status
```

回應：

```json
{
  "model_name": "Qwen/Qwen2.5-7B-Instruct",
  "device": "cuda:0",
  "conversation_length": 4,
  "memory_allocated": 7516192768
}
```

### 4. 健康檢查 (GET /health)

```bash
curl http://127.0.0.1:5000/health
```

## 使用客戶端範例

```bash
python api_client_example.py
```

這會啟動一個互動式客戶端，你可以：

- 直接輸入訊息聊天
- 輸入 `reset` 重設對話
- 輸入 `status` 查看狀態
- 輸入 `quit` 結束

## 參數說明

### 命令列參數

- `--model`: 模型名稱 (預設: Qwen/Qwen2.5-7B-Instruct)
- `--device`: 裝置 (預設: auto)
- `--host`: API 主機位址 (預設: 127.0.0.1)
- `--port`: API 連接埠 (預設: 5000)

### 聊天 API 參數

- `message`: 使用者訊息 (必須)
- `max_tokens`: 最大生成 token 數 (預設: 512)
- `temperature`: 溫度參數 (預設: 0.7)

## 支援的模型

### Qwen 系列

- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct`
- `Qwen/Qwen2.5-32B-Instruct`

### Llama 系列

- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.1-70B-Instruct`

## 注意事項

1. 第一次使用會自動下載模型檔案
2. 建議使用 GPU 以獲得更好的效能
3. 記憶體需求依模型大小而定
4. 對話歷史會自動限制在 20 輪以內

## 錯誤處理

API 會回傳適當的 HTTP 狀態碼和錯誤訊息：

- 400: 請求參數錯誤
- 500: 伺服器內部錯誤

錯誤回應格式：

```json
{
  "error": "錯誤訊息"
}
```
