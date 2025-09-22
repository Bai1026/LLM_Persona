# Token 計算器

這是一個專門用來計算對話檔案中 input 和 output tokens 的工具集。

## 功能特色

- 🔢 精確計算 input tokens (user 訊息) 和 output tokens (assistant 訊息)
- 📊 支援兩種格式：簡單對話格式和複雜 Discussion 格式
- 📁 支援單檔案和批次處理
- 📈 提供詳細統計報告
- 💾 可輸出 CSV 和 JSON 格式結果

## 安裝相依套件

```bash
pip install -r requirements_token_counter.txt
```

## 使用方法

### 1. 單檔案計算

```bash
# 基本使用
python token_counter.py your_file.json

# 指定模型
python token_counter.py your_file.json --model gpt-3.5-turbo

# 輸出結果到檔案
python token_counter.py your_file.json --output result.json
```

### 2. 批次處理

```bash
# 處理整個目錄
python batch_token_counter.py /path/to/json/files/

# 遞迴搜尋子目錄
python batch_token_counter.py /path/to/json/files/ --recursive

# 指定檔案模式
python batch_token_counter.py /path/to/json/files/ --pattern "*_chat_log.json"

# 輸出 CSV 和 JSON 結果
python batch_token_counter.py /path/to/json/files/ --output-csv results.csv --output-json results.json
```

### 3. 測試程式

```bash
# 執行測試
python test_token_counter.py
```

## 支援的檔案格式

### 格式 1: 簡單對話格式

```json
[
  {
    "item": "Fork",
    "PersonaAPI": [
      {
        "role": "user",
        "content": "Please provide 5 innovative uses for Fork."
      },
      {
        "role": "assistant",
        "content": "Here are 5 creative uses..."
      }
    ]
  }
]
```

### 格式 2: Discussion 格式

```json
{
  "What are some creative use for Fork?": {
    "QWEN Agent 6 - Environmentalist": [
      {
        "role": "user",
        "content": "You are an Environmentalist..."
      },
      {
        "role": "assistant",
        "content": "As an Environmentalist..."
      }
    ],
    "QWEN Agent 4 - Creative Professional": [
      {
        "role": "user",
        "content": "You are a Creative Professional..."
      },
      {
        "role": "assistant",
        "content": "As a Creative Professional..."
      }
    ]
  }
}
```

## 輸出說明

### 終端機輸出

```
📊 檔案 example.json 的 Token 統計結果
============================================================
📥 Input Tokens (user):       1,234
📤 Output Tokens (assistant): 5,678
📊 Total Tokens:              6,912
============================================================
📊 Input 比例:   17.9%
📊 Output 比例:  82.1%
```

### CSV 輸出欄位

- 檔案名稱
- 檔案路徑
- Input_Tokens
- Output_Tokens
- Total_Tokens
- Input\_比例(%)
- Output\_比例(%)

### JSON 輸出格式

```json
{
  "file": "example.json",
  "model": "gpt-4",
  "input_tokens": 1234,
  "output_tokens": 5678,
  "total_tokens": 6912,
  "input_ratio": 17.9,
  "output_ratio": 82.1
}
```

## 範例使用場景

### 計算 PersonaAPI 對話的 tokens

```bash
python token_counter.py /path/to/AUT_persona_api_0913-1905_100.json
```

### 批次計算 Discussion 格式檔案

```bash
python batch_token_counter.py /path/to/discussion/files/ \
    --pattern "*_chat_log.json" \
    --output-csv discussion_tokens.csv \
    --recursive
```

### 比較不同模型的 token 計算

```bash
# GPT-4 計算
python token_counter.py file.json --model gpt-4 --output gpt4_result.json

# GPT-3.5 計算
python token_counter.py file.json --model gpt-3.5-turbo --output gpt35_result.json
```

## 注意事項

1. **模型選擇**: 不同模型使用不同的 tokenizer，結果會有差異
2. **檔案編碼**: 確保 JSON 檔案使用 UTF-8 編碼
3. **記憶體使用**: 大檔案可能佔用較多記憶體
4. **格式檢測**: 程式會自動檢測檔案格式，但請確保格式正確

## 錯誤處理

- 檔案不存在或無法讀取會顯示錯誤訊息
- 格式不支援會提示未知格式
- JSON 解析錯誤會顯示具體錯誤位置
- 批次處理時個別檔案錯誤不會中斷整個流程
