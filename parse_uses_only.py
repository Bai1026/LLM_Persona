#!/usr/bin/env python3
"""
純粹的 chat_log 解析腳本
從 chat_log 檔案中提取 uses 資料
"""

import json
import re
import argparse
import os

def extract_responses(content):
    """提取回應內容，從 assistant 的回答中解析出具體的 uses"""
    import re
    
    # 先處理被用戶輸入截斷的內容
    # 移除 "user" 開始的部分和後續內容
    if '\nuser\n' in content:
        content = content.split('\nuser\n')[0]
    
    # 優化的正規表達式，能處理多種格式
    patterns = [
        # 格式1: 1. **Title**:** (內容) - 處理標題後有額外冒號和星號的情況
        r'(\d+)\.\s*\*\*([^*]+?)\*\*:?\*?:?\s*(.*?)(?=\d+\.\s*\*\*|$)',
        # 格式2: **1.** **Title** (新格式)
        r'\*\*(\d+)\.\*\*\s*\*\*([^*]+?)\*\*\s*(.*?)(?=\*\*\d+\.\*\*|$)',
        # 格式3: **1. Title:** (內容)
        r'\*\*(\d+)\.\s*([^*]+?)\*\*:?\s*(.*?)(?=\*\*\d+\.|$)',
        # 格式4: 1. **Title:** (內容)  
        r'(\d+)\.\s*\*\*([^*]+?)\*\*:?\s*(.*?)(?=\d+\.\s*\*\*|$)',
        # 格式5: 數字開頭的一般項目
        r'(\d+)\.\s*([^\n]*?)\n(.*?)(?=\d+\.|$)'
    ]
    
    responses = []
    
    for pattern in patterns:
        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
        if matches:
            for match in matches:
                if len(match) == 3:  # (number, title, content)
                    number = match[0].strip()
                    title = match[1].strip()
                    body = match[2].strip()
                    
                    # 建構完整項目
                    full_item = f"**{title}**"
                    if body:
                        full_item += f": {body}"
                    responses.append(full_item)
            
            if responses:  # 如果找到匹配，就不嘗試其他模式
                break
    
    # 如果上面的模式都沒匹配到，嘗試更寬泛的分割方法
    if not responses:
        # 按照多個換行符分割，尋找可能的項目
        sections = re.split(r'\n\n+', content)
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            # 檢查是否包含數字編號的項目
            if re.search(r'^\*?\*?\d+\.', section.strip(), re.MULTILINE):
                # 進一步分割這個section中的項目
                items = re.split(r'\n(?=\*?\*?\d+\.)', section)
                for item in items:
                    item = item.strip()
                    if item and re.match(r'^\*?\*?\d+\.', item):
                        # 清理格式但保留完整內容
                        clean_item = re.sub(r'^\*?\*?(\d+)\.\s*', '', item)
                        if clean_item:
                            responses.append(clean_item)
    
    # 如果還是沒有找到，使用原有的邏輯作為最後的回退
    if not responses:
        lines = content.split('\n')
        current_item = ""
        
        for line in lines:
            line = line.strip()
            
            # 檢查是否是新項目的開始
            if line and (line.startswith('-') or line.startswith('•') or 
                        any(line.startswith(f"{i}.") for i in range(1, 20)) or
                        any(line.startswith(f"**{i}.") for i in range(1, 20))):
                # 如果有前一個項目，先儲存
                if current_item:
                    clean_response = current_item.lstrip('-•*0123456789. ').strip()
                    if clean_response:
                        responses.append(clean_response)
                
                # 開始新項目
                current_item = line
            elif current_item and line:
                # 繼續當前項目的內容
                current_item += "\n" + line
        
        # 處理最後一個項目
        if current_item:
            clean_response = current_item.lstrip('-•*0123456789. ').strip()
            if clean_response:
                responses.append(clean_response)
    
    return responses[:10]  # 限制最多10個回應

def parse_chat_log_to_uses(input_file, output_file=None):
    """
    從 chat_log 檔案中解析出 uses 資料
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        final_results = []
        
        for item_name, conversations in data.items():
            # 跳過空的項目
            if not conversations:
                continue
                
            # 找到第一個有對話內容的 agent
            agent_name = None
            assistant_content = None
            
            for agent, conversation_list in conversations.items():
                if conversation_list:
                    agent_name = agent
                    # 找到 assistant 的回答
                    for conversation in conversation_list:
                        if conversation.get("role") == "assistant":
                            assistant_content = conversation.get("content", "")
                            break
                    if assistant_content:
                        break
            
            if assistant_content and agent_name:
                # 解析 uses 資料
                extracted_uses = extract_responses(assistant_content)
                if extracted_uses:  # 只有當解析出內容時才加入
                    final_results.append({
                        "item": item_name,
                        "uses": extracted_uses,
                        "Agent": agent_name
                    })
        
        # 決定輸出檔案名稱
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_parsed_uses.json"
        
        # 寫入解析出的 uses 檔案
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 成功解析：{input_file}")
        print(f"✅ 輸出檔案：{output_file}")
        print(f"📊 解析出 {len(final_results)} 個項目")
        
        # 顯示解析出的項目名稱
        if final_results:
            print("🔍 解析的項目：")
            for result in final_results[:5]:  # 只顯示前5個
                print(f"   - {result['item']}")
            if len(final_results) > 5:
                print(f"   ... 還有 {len(final_results) - 5} 個項目")
        
        return True, output_file
        
    except Exception as e:
        print(f"❌ 解析失敗：{e}")
        return False, None

def main():
    parser = argparse.ArgumentParser(description='從 chat_log 檔案中解析 uses 資料')
    parser.add_argument('input_file', help='輸入的 chat_log 檔案路徑')
    parser.add_argument('-o', '--output', help='輸出檔案路徑 (預設：在輸入檔案名稱後加上 _parsed_uses)')
    
    args = parser.parse_args()
    
    # 檢查輸入檔案是否存在
    if not os.path.exists(args.input_file):
        print(f"❌ 輸入檔案不存在：{args.input_file}")
        return
    
    # 執行解析
    success, output_file = parse_chat_log_to_uses(args.input_file, args.output)
    
    if success:
        print(f"\n🎉 解析完成！")
    else:
        print(f"\n💔 解析失敗！")

if __name__ == "__main__":
    # 如果沒有命令列參數，處理當前目錄的預設檔案
    import sys
    if len(sys.argv) == 1:
        # input_file = "/workspace/LLM_Persona/LLM-Discussion/Results/AUT/Output/persona_agent/AUT_persona_api_0918-0527_100_chat_log.json"
        input_file = "/workspace/LLM_Persona/LLM-Discussion/Results/Scientific/Output/persona_agent/Scientific_persona_api_0918-2154_100_chat_log.json"
        
        if os.path.exists(input_file):
            print(f"🔧 使用預設檔案：{input_file}")
            success, output_file = parse_chat_log_to_uses(input_file)
        else:
            print(f"❌ 找不到預設檔案：{input_file}")
            print("💡 請提供檔案路徑，例如：")
            print("   python parse_uses_only.py /path/to/chat_log.json")
    else:
        main()
