#!/usr/bin/env python3
"""
純粹的 chat_log 解析腳本
從 chat_log 檔案中提取 uses 資料
"""

import json
import re
import argparse
import os

def clean_repetitive_content(text):
    """清理重複的內容"""
    if not text:
        return text
    
    # 首先處理特殊的重複模式：「Each of these bowls, now, a bowl of the impossible...」
    impossible_pattern = r'Each of these bowls, now, a bowl of the impossible.*'
    match = re.search(impossible_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        text = text[:match.start()].strip()
    
    # 檢測並移除明顯的重複模式
    # 例如：「a bowl of the impossible, a bowl of the impossible, ...」
    
    # 找出重複的短語模式
    # 先分割成句子或短語
    phrases = re.split(r'[,，。！!?？]', text)
    
    # 如果有很多重複的短語，截取到第一次重複出現的地方
    seen_phrases = set()
    clean_phrases = []
    repetition_threshold = 3  # 如果同一個短語出現超過3次，就截斷
    phrase_counts = {}
    
    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase:
            continue
            
        # 計算這個短語出現的次數
        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        # 如果這個短語已經出現太多次，停止添加
        if phrase_counts[phrase] <= repetition_threshold:
            clean_phrases.append(phrase)
        else:
            # 發現重複，停止處理後續內容
            break
    
    # 重新組合文本
    cleaned = ', '.join(clean_phrases)
    
    # 移除明顯的重複模式（更精確的方法）
    # 檢測連續重複的單詞或短語
    words = cleaned.split()
    if len(words) > 20:  # 只處理較長的文本
        # 檢查最後部分是否有重複
        last_part = ' '.join(words[-20:])  # 檢查最後20個單詞
        # 如果發現重複模式，截取到重複開始的地方
        for i in range(1, 10):  # 檢查1-9個單詞的重複模式
            pattern = ' '.join(words[-i:])
            count = last_part.count(pattern)
            if count >= 3 and len(pattern.split()) >= 2:  # 至少2個單詞重複3次以上
                # 找到重複開始的位置
                before_repetition = cleaned.split(pattern)[0]
                if before_repetition:
                    cleaned = before_repetition.rstrip(' ,，')
                break
    
    return cleaned

def extract_responses(content):
    """提取回應內容，從 assistant 的回答中解析出具體的 uses"""
    import re
    
    # 首先嘗試提取完整的回應，包括被 user 中斷後的內容
    enhanced_content = content
    
    # 檢查是否有 \nuser\n 中斷
    user_match = re.search(r'\nuser\n', content)
    if user_match:
        # 獲取被截斷前的內容
        before_user = content[:user_match.start()]
        after_user_section = content[user_match.end():]
        
        # 跳過用戶的輸入，找到後續的回應
        lines_after_user = after_user_section.split('\n')
        assistant_response_lines = []
        skip_user_input = True
        
        for line in lines_after_user:
            line = line.strip()
            # 跳過用戶的指令部分
            if skip_user_input:
                if line.startswith(('Continue', 'Now', 'Imagine', 'Tell me', 'What', 'How')):
                    continue
                elif line == '' or len(line) < 10:
                    continue
                else:
                    skip_user_input = False
            
            # 收集助理的回應
            if not skip_user_input and line:
                assistant_response_lines.append(line)
        
        # 如果找到後續的助理回應，將其合併
        if assistant_response_lines:
            additional_response = ' '.join(assistant_response_lines)
            # 檢查 before_user 是否以不完整的句子結尾
            if before_user.rstrip().endswith(('like', '(', 'such as', 'including', 'with')):
                # 嘗試智慧地連接內容
                enhanced_content = before_user.rstrip() + ' ' + additional_response
            else:
                enhanced_content = before_user
        else:
            enhanced_content = before_user
    
    # 移除其他可能的截斷標記
    cleanup_patterns = [
        r'\nContinue.*',
        r'\nNow.*',
        r'\nImagine.*'
    ]
    
    for pattern in cleanup_patterns:
        match = re.search(pattern, enhanced_content, re.DOTALL | re.IGNORECASE)
        if match:
            enhanced_content = enhanced_content[:match.start()]
    
    content = enhanced_content
    
    # 優化的正規表達式，能處理多種格式
    patterns = [
        # 格式1: 1. **Title** (可選冒號) - 改進版本，能處理混合格式
        r'(\d+)\.\s*\*\*([^*]+?)\*\*:?\s*\n?(.*?)(?=\d+\.\s*\*\*|$)',
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
    
    for i, pattern in enumerate(patterns):
        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
        print(f"DEBUG: Pattern {i+1} found {len(matches)} matches")  # 除錯資訊
        if matches:
            for j, match in enumerate(matches):
                if len(match) == 3:  # (number, title, content)
                    number = match[0].strip()
                    title = match[1].strip()
                    body = match[2].strip()
                    
                    print(f"DEBUG: Processing match {j+1}: '{title}', body length: {len(body)}")
                    
                    # 清理重複的內容
                    original_body_length = len(body)
                    body = clean_repetitive_content(body)
                    cleaned_body_length = len(body)
                    
                    print(f"DEBUG: Body length after cleaning: {original_body_length} -> {cleaned_body_length}")
                    
                    # 建構完整項目
                    full_item = f"**{title}**"
                    if body:
                        full_item += f": {body}"
                    responses.append(full_item)
                    print(f"DEBUG: Added response: {title}")  # 除錯資訊
            
            if responses:  # 如果找到匹配，就不嘗試其他模式
                print(f"DEBUG: Total responses collected: {len(responses)}")  # 除錯資訊
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
        input_file = "/workspace/LLM_Persona/LLM-Discussion/Results/Similarities/Output/persona_agent/Similarities_persona_api_0920-2032_100_chat_log.json"
        
        if os.path.exists(input_file):
            print(f"🔧 使用預設檔案：{input_file}")
            success, output_file = parse_chat_log_to_uses(input_file)
        else:
            print(f"❌ 找不到預設檔案：{input_file}")
            print("💡 請提供檔案路徑，例如：")
            print("   python parse_uses_only.py /path/to/chat_log.json")
    else:
        main()
