# Multi-Layer Persona Steering API

## 概述

這是一個支援多層級 Persona Activation Steering 的 Flask API 服務。可以讓不同的 persona 在模型的不同層進行 steering，實現更精細的控制。

## 主要功能

- **多層級 Steering**: 不同 persona 可以在不同的 transformer 層進行 activation steering
- **動態權重調整**: 即時調整各個 persona 的影響係數
- **預設模式**: 提供多種預設的 persona 組合模式
- **靈活配置**: 支援 JSON 配置檔案自定義 persona 設定
- **RESTful API**: 完整的 REST API 介面

## 檔案結構

```
/workspace/LLM_Persona/persona_vectors/Experiment/
├── multi_layer_steering_api.py      # 主要 API 服務
├── activation_steer.py              # Activation Steering 核心邏輯
├── persona_config_example.json      # 範例配置檔案
├── test_multi_layer_api.py          # API 測試和互動工具
├── start_multi_layer_api.sh         # 啟動腳本
└── README_multi_layer_api.md        # 本說明文件
```

## 快速開始

### 1. 準備 Persona 向量

確保你有準備好的 persona 向量檔案（.npy 格式）：

```
persona_vectors/
├── environmentalist_vector.npy
├── creative_vector.npy
└── futurist_vector.npy
```

### 2. 建立配置檔案

建立一個 JSON 配置檔案，定義每個 persona 的設定：

```json
[
  {
    "name": "environmentalist",
    "vector_path": "../persona_vectors/environmentalist_vector.npy",
    "layer_idx": 15,
    "coeff": 2.0,
    "positions": "all"
  },
  {
    "name": "creative",
    "vector_path": "../persona_vectors/creative_vector.npy",
    "layer_idx": 20,
    "coeff": 1.5,
    "positions": "response"
  },
  {
    "name": "futurist",
    "vector_path": "../persona_vectors/futurist_vector.npy",
    "layer_idx": 25,
    "coeff": 1.0,
    "positions": "all"
  }
]
```

配置參數說明：

- `name`: Persona 名稱
- `vector_path`: 向量檔案路徑
- `layer_idx`: 目標層索引（從 0 開始）
- `coeff`: Steering 係數（權重）
- `positions`: Steering 位置（"all", "prompt", "response"）

### 3. 啟動服務

使用啟動腳本：

```bash
./start_multi_layer_api.sh --config persona_config_example.json
```

或直接執行 Python：

```bash
python multi_layer_steering_api.py --config persona_config_example.json --model meta-llama/Llama-3.1-8B-Instruct
```

### 4. 測試 API

```bash
# 執行自動測試
python test_multi_layer_api.py test

# 啟動互動聊天
python test_multi_layer_api.py
```

## API 端點

### POST /chat

發送訊息並取得回應

**請求**：

```json
{
  "user_input": "What are the key challenges facing humanity?",
  "max_tokens": 500
}
```

**回應**：

```json
{
    "user_input": "...",
    "response": "...",
    "current_weights": {"environmentalist": 2.0, "creative": 1.5, "futurist": 1.0},
    "persona_configs": {...},
    "status": "success"
}
```

### POST /set_persona_weights

設定 persona 權重

**請求**：

```json
{
  "weights": {
    "environmentalist": 3.0,
    "creative": 0.5,
    "futurist": 2.0
  }
}
```

### POST /set_persona_mode

設定預設模式

**請求**：

```json
{
  "mode": "creative_focus"
}
```

可用模式：

- `single_environmentalist`: 只啟用環保主義者
- `single_creative`: 只啟用創意專家
- `single_futurist`: 只啟用未來主義者
- `balanced`: 平衡模式
- `creative_focus`: 創意重點
- `analytical`: 分析模式
- `off`: 關閉所有 steering

### POST /update_layer_config

動態更新 persona 的層配置

**請求**：

```json
{
  "persona_name": "futurist",
  "layer_idx": 30
}
```

### POST /reset

重設對話歷史

### GET /status

取得當前狀態

**回應**：

```json
{
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "num_personas": 3,
    "persona_info": {
        "environmentalist": {
            "layer_idx": 15,
            "coeff": 2.0,
            "positions": "all",
            "vector_shape": [4096]
        }
    },
    "current_weights": {...},
    "conversation_length": 4
}
```

### GET /available_modes

取得可用的預設模式

## 使用範例

### Python 客戶端

```python
import requests

BASE_URL = "http://127.0.0.1:5000"

# 發送訊息
response = requests.post(f"{BASE_URL}/chat", json={
    "user_input": "Describe a sustainable city of the future",
    "max_tokens": 300
})

result = response.json()
print(result['response'])

# 調整權重
requests.post(f"{BASE_URL}/set_persona_weights", json={
    "weights": {"environmentalist": 3.0, "futurist": 2.0, "creative": 1.0}
})

# 設定模式
requests.post(f"{BASE_URL}/set_persona_mode", json={
    "mode": "balanced"
})
```

### 互動式聊天

```bash
python test_multi_layer_api.py

# 在聊天中使用指令：
# mode balanced          # 設定平衡模式
# weight creative 2.5    # 調整創意權重
# status                 # 查看狀態
# quit                   # 結束
```

## 進階功能

### 自定義 Persona 配置

你可以建立自己的配置檔案：

```json
[
  {
    "name": "scientist",
    "vector_path": "/path/to/scientist_vector.npy",
    "layer_idx": 10,
    "coeff": 1.8,
    "positions": "prompt"
  },
  {
    "name": "artist",
    "vector_path": "/path/to/artist_vector.npy",
    "layer_idx": 25,
    "coeff": 2.2,
    "positions": "response"
  }
]
```

### 多模型支援

API 支援多種模型：

```bash
# Llama 模型
./start_multi_layer_api.sh --model meta-llama/Llama-3.1-8B-Instruct

# Gemma 模型
./start_multi_layer_api.sh --model google/gemma-2-9b-it

# Qwen 模型
./start_multi_layer_api.sh --model Qwen/Qwen2.5-7B-Instruct
```

### 偵錯模式

開啟偵錯模式查看詳細資訊：

```bash
./start_multi_layer_api.sh --debug --config my_config.json
```

## 注意事項

1. **記憶體需求**: 大型模型需要足夠的 GPU 記憶體
2. **向量相容性**: 確保 persona 向量與模型的 hidden_size 相符
3. **層索引範圍**: layer_idx 必須在模型層數範圍內
4. **係數調整**: 建議從小係數開始測試，避免過度 steering

## 故障排除

### 常見錯誤

1. **向量檔案不存在**

   - 檢查配置檔案中的路徑是否正確

2. **層索引超出範圍**

   - 確認 layer_idx 在模型層數範圍內

3. **記憶體不足**

   - 嘗試使用較小的模型或增加 GPU 記憶體

4. **API 連接失敗**
   - 檢查服務是否正常啟動
   - 確認連接埠沒有被占用

### 效能最佳化

1. **向量預載**: 向量會在初始化時載入到 GPU
2. **對話長度限制**: 自動限制對話歷史長度避免記憶體溢出
3. **Token 限制**: 強制限制最大回應長度

## 擴展和客製化

你可以：

1. **添加新的 Persona**: 在配置檔案中新增條目
2. **自定義模式**: 修改 `set_persona_mode` 函式
3. **調整 API**: 新增自定義端點
4. **整合其他功能**: 結合其他 NLP 工具

這個 API 為多層級 persona steering 提供了完整的解決方案，讓你能靈活地控制模型在不同層的行為表現。
