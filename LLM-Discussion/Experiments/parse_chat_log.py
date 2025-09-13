#!/usr/bin/env python3
"""
獨立腳本：從 chat_log.json 解析出評估版本的 JSON 檔案
使用方法: python parse_chat_log.py -i input_chat_log.json -o output_eval.json -t AUT
"""

import argparse
import json
import re
from pathlib import Path

def extract_responses(content):
    """提取回應內容 - 與 persona_conversation.py 相同的邏輯"""
    import re
    
    # 使用正規表達式找到所有編號項目及其完整內容
    # 匹配格式如：1. **標題**: 描述內容...
    pattern = r'(\d+\.\s*\*\*[^*]+\*\*:?\s*(?:[^\n]+(?:\n(?!\d+\.\s*\*\*)[^\n]*)*)?)'
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
    
    responses = []
    for match in matches:
        # 清理並格式化每個項目
        clean_response = match.strip()
        # 移除開頭的數字和點號，但保留完整內容
        clean_response = re.sub(r'^\d+\.\s*', '', clean_response)
        if clean_response:
            responses.append(clean_response)
    
    # 如果正規表達式沒有匹配到，回退到原始邏輯
    if not responses:
        lines = content.split('\n')
        current_item = ""
        
        for line in lines:
            line = line.strip()
            
            # 檢查是否是新項目的開始
            if line and (line.startswith('-') or line.startswith('•') or 
                        any(line.startswith(f"{i}.") for i in range(1, 20))):
                # 如果有前一個項目，先儲存
                if current_item:
                    clean_response = current_item.lstrip('-•0123456789. ').strip()
                    if clean_response:
                        responses.append(clean_response)
                
                # 開始新項目
                current_item = line
            elif current_item and line:
                # 繼續當前項目的內容
                current_item += "\n" + line
        
        # 處理最後一個項目
        if current_item:
            clean_response = current_item.lstrip('-•0123456789. ').strip()
            if clean_response:
                responses.append(clean_response)
    
    return responses

def parse_chat_log_to_eval_format(chat_log_data, task_type):
    """將 chat_log 格式轉換為評估格式"""
    eval_results = []
    
    for item_key, conversations in chat_log_data.items():
        if "PersonaAPI" in conversations:
            conversation = conversations["PersonaAPI"]
            
            # 找到助手的回應
            assistant_content = None
            for msg in conversation:
                if msg.get("role") == "assistant":
                    assistant_content = msg.get("content", "")
                    break
            
            if assistant_content:
                # 提取回應
                extracted_responses = extract_responses(assistant_content)
                
                # 根據任務類型建立不同的結構
                if task_type == "AUT":
                    eval_results.append({
                        "item": item_key,
                        "uses": extracted_responses,
                        "Agent": "PersonaAPI"
                    })
                elif task_type == "Scientific":
                    eval_results.append({
                        "question": item_key,
                        "answer": extracted_responses,
                        "Agent": "PersonaAPI"
                    })
                elif task_type in ["Instances", "Similarities"]:
                    eval_results.append({
                        "question": item_key,
                        "answer": extracted_responses,
                        "Agent": "PersonaAPI"
                    })
                else:
                    print(f"❌ 不支援的任務類型: {task_type}")
                    continue
            else:
                # 如果沒有找到助手回應，建立空項目
                if task_type == "AUT":
                    eval_results.append({
                        "item": item_key,
                        "uses": [],
                        "Agent": "PersonaAPI"
                    })
                elif task_type == "Scientific":
                    eval_results.append({
                        "question": item_key,
                        "answer": [],
                        "Agent": "PersonaAPI"
                    })
                elif task_type in ["Instances", "Similarities"]:
                    eval_results.append({
                        "question": item_key,
                        "answer": [],
                        "Agent": "PersonaAPI"
                    })
    
    return eval_results

def auto_detect_task_type(chat_log_data):
    """自動偵測任務類型"""
    # 檢查第一個對話內容來猜測任務類型
    for item_key, conversations in chat_log_data.items():
        if "PersonaAPI" in conversations:
            conversation = conversations["PersonaAPI"]
            for msg in conversation:
                if msg.get("role") == "user":
                    content = msg.get("content", "").lower()
                    if "uses for" in content or "用途" in content:
                        return "AUT"
                    elif "scientific" in content or "科學" in content:
                        return "Scientific"
                    elif "instances" in content or "examples" in content or "範例" in content:
                        return "Instances"
                    elif "similarities" in content or "相似" in content:
                        return "Similarities"
            break
    
    # 預設返回 AUT
    return "AUT"

def generate_output_filename(input_filename, task_type):
    """根據輸入檔名產生輸出檔名"""
    input_path = Path(input_filename)
    
    # 移除 _chat_log 後綴
    base_name = input_path.stem
    if base_name.endswith("_chat_log"):
        base_name = base_name[:-9]  # 移除 "_chat_log"
    
    # 如果檔名不是標準格式，嘗試建立標準格式
    if not base_name.startswith(task_type):
        base_name = f"{task_type}_{base_name}"
    
    return f"{base_name}.json"

def main():
    parser = argparse.ArgumentParser(description="從 chat_log.json 解析出評估版本的 JSON 檔案")
    # parser.add_argument("-i", "--input", required=True, help="輸入的 chat_log.json 檔案路徑")
    # parser.add_argument("-o", "--output", help="輸出的評估 JSON 檔案路徑（可選，會自動產生）")
    parser.add_argument("-t", "--task", choices=["AUT", "Scientific", "Instances", "Similarities"], 
                       help="任務類型（可選，會自動偵測）")
    parser.add_argument("--dry-run", action="store_true", help="只顯示解析結果，不寫入檔案")
    
    args = parser.parse_args()
    
    
    # 檢查輸入檔案是否存在
    # input_path = Path(args.input)
    input_path = Path("../Results/AUT/Output/persona_agent/AUT_persona_api_0913-1905_100_chat_log.json")
    output_path = Path("../Results/AUT/Output/persona_agent/AUT_persona_api_0913-1905_100.json")
    
    if not input_path.exists():
        print(f"❌ 輸入檔案不存在: {args.input}")
        return 1
    
    # 載入 chat_log 資料
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            chat_log_data = json.load(f)
        print(f"✅ 成功載入 chat_log: {input_path}")
    except Exception as e:
        print(f"❌ 載入檔案失敗: {e}")
        return 1
    
    # 確定任務類型
    task_type = args.task
    if not task_type:
        task_type = auto_detect_task_type(chat_log_data)
        print(f"🔍 自動偵測任務類型: {task_type}")
    else:
        print(f"📋 使用指定任務類型: {task_type}")
    
    # 解析資料
    eval_results = parse_chat_log_to_eval_format(chat_log_data, task_type)
    
    print(f"📊 解析結果:")
    print(f"  - 項目總數: {len(eval_results)}")
    
    # 統計有效項目
    valid_items = 0
    empty_items = 0
    
    for item in eval_results:
        if task_type == "AUT":
            if item.get("uses"):
                valid_items += 1
            else:
                empty_items += 1
        else:
            if item.get("answer"):
                valid_items += 1
            else:
                empty_items += 1
    
    print(f"  - 有效項目: {valid_items}")
    print(f"  - 空白項目: {empty_items}")
    
    # 顯示前幾個項目的範例
    print(f"\n📋 前 3 個項目範例:")
    for i, item in enumerate(eval_results[:3]):
        if task_type == "AUT":
            item_name = item.get("item", "Unknown")
            uses_count = len(item.get("uses", []))
            print(f"  {i+1}. {item_name}: {uses_count} uses")
            # 顯示第一個 use 的開頭
            if item.get("uses"):
                first_use = item["uses"][0]
                preview = first_use[:50] + "..." if len(first_use) > 50 else first_use
                print(f"     └─ {preview}")
        else:
            question = item.get("question", "Unknown")
            answer_count = len(item.get("answer", []))
            print(f"  {i+1}. {question}: {answer_count} answers")
    
    # 如果是 dry-run，只顯示結果不寫入檔案
    if args.dry_run:
        print(f"\n🔍 Dry-run 模式，不寫入檔案")
        print(f"完整資料預覽:")
        print(json.dumps(eval_results[:2], indent=2, ensure_ascii=False))
        return 0
    
    # 確定輸出檔案路徑
    # if args.output:
    #     output_path = Path(args.output)
    # else:
    #     output_filename = generate_output_filename(args.input, task_type)
    #     output_path = input_path.parent / output_filename
    
    # 寫入檔案
    try:
        # 確保輸出目錄存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 評估格式檔案已儲存: {output_path}")
        print(f"📁 檔案大小: {output_path.stat().st_size} bytes")
        
        return 0
        
    except Exception as e:
        print(f"❌ 寫入檔案失敗: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
