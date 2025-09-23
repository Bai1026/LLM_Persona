import argparse
import sys
import os
import json
import requests
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
import pytz

class PersonaAPIRunner:
    """使用 Persona API 進行單次回應的類別"""
    
    def __init__(self, api_url, dataset_file, task_type, prompt_id):
        self.api_url = api_url
        self.dataset_file = dataset_file
        self.task_type = task_type
        self.prompt_id = prompt_id
        
    def test_api_connection(self):
        """測試 API 連線"""
        try:
            response = requests.get(f"{self.api_url}/status")
            if response.status_code == 200:
                print("✅ API 連線成功")
                return True
            else:
                print("❌ API 連線失敗")
                return False
        except Exception as e:
            print(f"❌ API 連線錯誤: {e}")
            return False
    
    def reset_conversation(self):
        """重設對話歷史"""
        try:
            response = requests.post(f"{self.api_url}/reset")
            return response.status_code == 200
        except:
            return False
    
    def call_persona_api(self, user_input, max_tokens=1000):
        """呼叫 Persona API"""
        try:
            data = {
                "user_input": user_input,
                "max_tokens": max_tokens
            }
            response = requests.post(f"{self.api_url}/chat", json=data)
            
            if response.status_code == 200:
                result = response.json()
                return result["response"]
            else:
                print(f"❌ API 請求失敗: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ API 請求錯誤: {e}")
            return None
    
    def load_dataset(self):
        """載入資料集"""
        with open(self.dataset_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def construct_prompt(self, item):
        """建構提示詞"""
        # 將 LLM Discussion 的協作提示詞改為單一代理版本
        base_prompts = {
            1: "You are working on a creative task; as a result, answer as diversely and creatively as you can.",
            2: "You're in a brainstorming session where each idea leads to the next. Embrace the flow of creativity without limits, building on suggestions for unexpected connections.",
            3: "Pretend you are at a think tank where unconventional ideas are the norm. Challenge yourself to think from different perspectives, considering the most unusual or innovative ideas.",
            4: "Engage in a creative thinking process where you contribute unique insights and queries, aiming to delve into uncharted territories of thought. Focus on expanding the scope and depth of your contribution through innovative thinking, counterpoints, and further questioning. The objective is to achieve a broad spectrum of ideas and solutions, promoting continuous learning and innovation.",
            5: "Envision yourself as an expert on a mission to solve a challenge using only your creativity and wit. How would you piece together clues from different perspectives to find the solution? This would be crucial to the success of the mission."
        }
        
        # use the -p parameter
        discussion_prompt = base_prompts.get(self.prompt_id, base_prompts[1])
        
        if self.task_type == "AUT":
            # # 使用與 LLM Discussion 相同的完整任務描述格式, this does not work well for Qwen and Llama
            # task_description = f"What are some creative use for {item}? The goal is to come up with creative ideas, which are ideas that strike people as clever, unusual, interesting, uncommon, humorous, innovative, or different. Present a list of as many creative and diverse uses for {item} as possible."
            # task_prompt = f"Can you answer the following question as creatively as possible: {task_description} {discussion_prompt} Please put your answer in a list format, starting with 1. ... 2. ... 3. ... and so on."

            # Test-1
            task_prompt = f"Please provide 5 innovative and original uses for '{item}'. {discussion_prompt}"

            # Test-2
            # task_prompt = f"Please provide some innovative and original uses for '{item}'. {discussion_prompt}"

            # Task-3
            # task_prompt = f"Please provide 10 innovative and original uses for '{item}'. {discussion_prompt}"

        elif self.task_type == "Scientific":
            task_prompt = f"Can you answer the following scientific question as creatively as possible: {item} {discussion_prompt} Please provide 5 innovative solutions in a list format, starting with 1. ... 2. ... 3. ..."
        elif self.task_type == "Instances":
            task_prompt = f"Can you answer the following question as creatively as possible: {item} {discussion_prompt} Please provide 5 creative examples in a list format, starting with 1. ... 2. ... 3. ... and so on."
        elif self.task_type == "Similarities":
            task_prompt = f"Can you answer the following question as creatively as possible: {item} {discussion_prompt} Please provide 5 creative perspectives in a list format, starting with 1. ... 2. ... 3. ..."
        
        return task_prompt
    
    # # Chinese version of construct_prompt (better for Qwen and Llama)
    # def construct_prompt(self, item):
    #     """建構提示詞"""
    #     base_prompts = {
    #         1: "請盡可能多樣化和創造性地回答。",
    #         2: "無限制地擁抱創造力的流動，提供意想不到的連接。",
    #         3: "請從不同角度思考，考慮最不尋常或創新的想法。",
    #         4: "請提供獨特的見解，專注於創新的想法和解決方案。",
    #         5: "請使用你的創造力和智慧來提供最佳解決方案。"
    #     }
        
    #     discussion_prompt = base_prompts.get(self.prompt_id, base_prompts[1])
        
    #     if self.task_type == "AUT":
    #         task_prompt = f"請為「{item}」提供5個創新和原創的用途。{discussion_prompt}"
    #     elif self.task_type == "Scientific":
    #         task_prompt = f"請為以下科學問題提供5個創新的解決方案：{item}。{discussion_prompt}"
    #     elif self.task_type == "Instances":
    #         task_prompt = f"請為「{item}」提供5個創造性的範例。{discussion_prompt}"
    #     elif self.task_type == "Similarities":
    #         task_prompt = f"請分析以下相似性並提供5個創造性的觀點：{item}。{discussion_prompt}"
        
    #     return task_prompt

    def extract_responses(self, content):
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
            # print(f"DEBUG: Pattern {i+1} found {len(matches)} matches")  # 除錯資訊
            if matches:
                for j, match in enumerate(matches):
                    if len(match) == 3:  # (number, title, content)
                        number = match[0].strip()
                        title = match[1].strip()
                        body = match[2].strip()
                        
                        # print(f"DEBUG: Processing match {j+1}: '{title}', body length: {len(body)}")
                        
                        # 清理重複的內容
                        # original_body_length = len(body)
                        # body = self.clean_repetitive_content(body)
                        # cleaned_body_length = len(body)
                        
                        # print(f"DEBUG: Body length after cleaning: {original_body_length} -> {cleaned_body_length}")
                        
                        # 建構完整項目
                        full_item = f"**{title}**"
                        if body:
                            full_item += f": {body}"
                        responses.append(full_item)
                        # print(f"DEBUG: Added response: {title}")  # 除錯資訊
                
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
    
    def run(self):
        """執行 API 呼叫"""
        if not self.test_api_connection():
            return None
        
        dataset = self.load_dataset()
        
        # 根據任務類型提取資料
        if self.task_type == "Scientific":
            # Scientific 資料集格式：{"Task": [{"Original": "...", "Example": [...]}]}
            examples = []
            for task in dataset.get("Task", []):
                examples.extend(task.get("Example", []))
        elif isinstance(dataset, dict) and "Examples" in dataset:
            # AUT, Instances, Similarities 格式：{"Examples": [...]}
            examples = dataset["Examples"]
        else:
            examples = dataset
        
        all_responses = {}
        final_results = []
        
        print(f"🚀 開始 {self.task_type} 任務，共 {len(examples)} 個項目")
        
        from tqdm import tqdm
        for item_data in tqdm(examples):
            # 處理不同的資料集格式
            if isinstance(item_data, str):
                # 如果是字串格式（Scientific, Instances, Similarities 通常是這種格式）
                item = item_data
            elif isinstance(item_data, dict):
                # 如果是字典格式，獲取項目內容
                if self.task_type == "AUT":
                    item = item_data.get("object", item_data.get("item", ""))
                elif self.task_type == "Scientific":
                    item = item_data.get("question", item_data.get("item", ""))
                else:  # Instances 或 Similarities
                    item = item_data.get("question", item_data.get("item", item_data.get("object", "")))
            else:
                print(f"❌ 不支援的資料格式: {type(item_data)}")
                continue
            
            if not item:
                print(f"❌ 空白項目，跳過")
                continue
                
            print(f"📋 處理項目: {item}")
            
            # 重設對話歷史
            self.reset_conversation()
            
            # 建構提示詞
            prompt = self.construct_prompt(item)
            
            # 呼叫 API
            response = self.call_persona_api(prompt, max_tokens=1000)
            
            if response:
                # 儲存對話記錄（模擬 multi-agent 格式）
                all_responses[item] = {
                    "PersonaAPI": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
                }
                
                # 提取並儲存最終結果
                extracted = self.extract_responses(response)
                if self.task_type == "AUT":
                    final_results.append({
                        "item": item,
                        "uses": extracted,
                        "Agent": "PersonaAPI"
                    })
                elif self.task_type == "Scientific":
                    final_results.append({
                        "question": item,
                        "answer": extracted,
                        "Agent": "PersonaAPI"
                    })
                else:
                    final_results.append({
                        "question": item,
                        "answer": extracted,
                        "Agent": "PersonaAPI"
                    })
            else:
                print(f"❌ 項目 {item} 處理失敗")
        
        # 儲存結果
        output_filename = self.save_results(all_responses, final_results, len(examples))
        print(f"✅ 任務完成，結果已儲存至: {output_filename}")
        
        return output_filename
    
    def save_results(self, all_responses, final_results, amount_of_data):
        """儲存結果檔案"""
        # 使用 UTC+8 時區（台灣時間）
        taipei_tz = pytz.timezone('Asia/Taipei')
        taipei_time = datetime.now(taipei_tz)
        
        current_date = taipei_time.strftime("%m%d")
        formatted_time = taipei_time.strftime("%H%M")
        
        # 建立檔案名稱（符合評估系統格式）
        base_filename = f"{self.task_type}_persona_api_{current_date}-{formatted_time}_{amount_of_data}"
        
        # 評估系統期待的路徑結構：Results/{task}/Output/{agent}_agent/
        results_base_path = Path(__file__).parent.parent / "Results" / self.task_type / "Output" / "persona_agent"
        results_base_path.mkdir(parents=True, exist_ok=True)
        
        # 儲存對話記錄
        chat_log_filename = f"{base_filename}_chat_log.json"
        chat_log_path = results_base_path / chat_log_filename
        
        with open(chat_log_path, 'w', encoding='utf-8') as f:
            json.dump(all_responses, f, indent=2, ensure_ascii=False)
        
        # 儲存最終結果（這是評估系統需要的檔案）
        final_filename = f"{base_filename}.json"
        final_path = results_base_path / final_filename
        
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 對話記錄已儲存: {chat_log_path}")
        print(f"💾 最終結果已儲存: {final_path}")
        
        # 回傳檔案名稱（不含副檔名，供評估使用）
        return final_filename.replace('.json', '')

def main():
    parser = argparse.ArgumentParser(description="使用 Persona API 進行創造性任務評估")
    parser.add_argument("-d", "--dataset", required=True, help="資料集檔案路徑")
    parser.add_argument("-t", "--type", choices=["AUT", "Scientific", "Similarities", "Instances"], 
                       required=True, help="任務類型")
    parser.add_argument("-p", "--prompt", type=int, default=1, help="提示詞編號 (1-5)")
    parser.add_argument("-u", "--api_url", default="http://127.0.0.1:5001", help="Persona API 網址")
    parser.add_argument("-e", "--eval_mode", action="store_true", default=False, help="執行評估模式")
    
    args = parser.parse_args()
    
    from datetime import datetime
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🕒 開始時間: {start_time}")

    # 建立並執行 API 執行器
    runner = PersonaAPIRunner(args.api_url, args.dataset, args.type, args.prompt)
    discussion_output = runner.run()
    
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # if args.eval_mode and discussion_output:
    #     # 整合原有的評估系統
    #     evaluation_root = Path(__file__).parent.parent / 'Evaluation'
    #     sys.path.append(str(evaluation_root))
    #     from auto_grade_final import auto_grade
        
    #     # 呼叫評估
    #     eval_args = SimpleNamespace(
    #         version="4", 
    #         input_file=discussion_output,  # 這裡已經是正確的檔案名稱
    #         type="sampling", 
    #         sample=3, 
    #         task=args.type, 
    #         output="y"
    #     )
    #     auto_grade(eval_args)
    #     print(f"🕒 結束時間: {end_time}")
    #     print(f"Total Time: {start_time} to {end_time}")


if __name__ == "__main__":
    main()

