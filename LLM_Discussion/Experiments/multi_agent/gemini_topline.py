import argparse
import sys
import os
import time
from datetime import timedelta
import json
# import google.generativeai as genai
from google import genai
from google.genai import types
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
import pytz
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

class GoogleToplineRunner:
    """使用 Google Generative AI API 進行創造性任務的基線模型"""
    
    # --- MODIFIED: Changed default model to a Google model ---
    def __init__(self, dataset_file, task_type, prompt_id, model_name="gemini-2.5-pro", multi_role_prompt=True):
        self.dataset_file = dataset_file
        self.task_type = task_type
        self.prompt_id = prompt_id
        self.model_name = model_name
        self.multi_role_prompt = multi_role_prompt
        self.TOTAL_TOKEN_USED = {"Input": 0, "Thinking": 0, "Output": 0}
        
        # --- MODIFIED: Initialize Google Generative AI client ---
        try:
            # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            # self.model = genai.GenerativeModel(self.model_name)
            self.client = genai.Client()
            print(f"✅ Google AI SDK 初始化成功，使用 {self.model_name}")
        except Exception as e:
            print(f"❌ Google AI SDK 初始化失敗: {e}")
            self.model = None
        
    # def test_api_connection(self):
    #     """測試 Google API 連線"""
    #     if not self.model_name:
    #         return False
    #     try:
    #         # 簡單的測試方法是列出模型
    #         models = genai.list_models()
    #         # 檢查是否有任何模型返回
    #         if any(m for m in models):
    #             print(f"✅ Google API 連線成功")
    #             return True
    #         else:
    #             print(f"❌ Google API 連線錯誤: 未能獲取模型列表")
    #             return False
    #     except Exception as e:
    #         print(f"❌ Google API 連線錯誤: {e}")
    #         return False

    def token_calculate(self, response, time_delta):
        # print(response)
        if not response:
            return 
        input_token = response.usage_metadata.prompt_token_count
        output_token = response.usage_metadata.candidates_token_count
        thinking_token = response.usage_metadata.thoughts_token_count
        # print(input_token, output_token, thinking_token)
        self.TOTAL_TOKEN_USED["Input"] += input_token
        self.TOTAL_TOKEN_USED["Output"] += output_token
        self.TOTAL_TOKEN_USED["Thinking"] += thinking_token
        current_input = self.TOTAL_TOKEN_USED["Input"]
        current_out = self.TOTAL_TOKEN_USED["Output"]+self.TOTAL_TOKEN_USED["Thinking"]
        current_fee = self.TOTAL_TOKEN_USED["Input"]*(1.25/1000000) + (self.TOTAL_TOKEN_USED["Output"]+self.TOTAL_TOKEN_USED["Thinking"])*(10/1000000)

        record = f"\n---\nTime Usage: {time_delta}\n"
        record += f"Input token: {input_token}, Thinking token: {thinking_token}, Output token: {output_token}\n"
        record += f"    Current Input: {current_input}, Current Ouput: {current_out}, Current Fee: {current_fee}"

        with open("token_record.txt", "a", encoding="utf-8") as f:
            f.write(record)
        
    # --- MODIFIED: Renamed method and implemented Google API call ---
    def call_google_api(self, prompt):
        start_time = time.time()
        """呼叫 Google Generative AI API"""
        if not self.model_name:
            return None
        try:
            # 設定生成參數
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    # max_output_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    # thinking_config=types.ThinkingConfig(thinking_budget=128), # Disables thinking
                ),
            )
            end_time = time.time()
            total_time = end_time - start_time
            time_delta = timedelta(seconds=total_time)
            self.token_calculate(response, time_delta)
            print(f"Sleep Time ...😴")
            time.sleep(30)
            return response.text
            
        except Exception as e:
            print(f"❌ Google API 請求錯誤: {e}")
            return None
    
    def load_dataset(self):
        """載入資料集"""
        with open(self.dataset_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
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
    #         task_prompt = f"請為以下科學問題提供3個創新的解決方案：{item}。{discussion_prompt}"
    #     elif self.task_type == "Instances":
    #         task_prompt = f"請為「{item}」提供5個創造性的範例。{discussion_prompt}"
    #     elif self.task_type == "Similarities":
    #         task_prompt = f"請分析以下相似性並提供3個創造性的觀點：{item}。{discussion_prompt}"
        
    #     return task_prompt

    def construct_prompt(self, item):
        """建構提示詞"""
        base_prompts = {
            1: "Please answer as diversely and creatively as possible.",
            2: "Embrace the flow of creativity without limits, providing unexpected connections.",
            3: "Please think from different perspectives, considering the most unusual or innovative ideas.",
            4: "Please provide unique insights, focusing on innovative ideas and solutions.",
            5: "Please use your creativity and intelligence to provide the best solutions."
        }
        
        discussion_prompt = base_prompts.get(self.prompt_id, base_prompts[1])
        
        print(f"self.multi_role_prompt: {self.multi_role_prompt}")
        if self.multi_role_prompt == False:
            MULTI_ROLE_PLAY = False
        else: 
            MULTI_ROLE_PLAY = True
        # 多角色扮演提示詞
        if MULTI_ROLE_PLAY:
            print(f"\nMODE: Multi role play\n")
            role_prompts = """
You need to think and answer this question from three different professional perspectives:

1. Environmentalist:
Specialty: Sustainability and Environmental Health
Mission: Advocate for eco-friendly solutions, promote sustainable development and protect the planet. Guide us to consider the environmental impact of ideas, promoting innovations that contribute to planetary health.

2. Creative Professional:
Specialty: Aesthetics, Narratives, and Emotions
Mission: With artistic sensibility and mastery of narrative and emotion, infuse projects with beauty and depth. Challenge us to think expressively, ensuring solutions not only solve problems but also resonate on a human level.

3. Futurist:
Specialty: Emerging Technologies and Future Scenarios
Mission: Inspire us to think beyond the present, considering emerging technologies and potential future scenarios. Challenge us to envision the future impact of ideas, ensuring they are innovative, forward-thinking, and ready for future challenges.

Please provide answers from these three role perspectives, with each role embodying their professional characteristics and thinking approaches.
"""
        else:
            print(f"\nMODE: Single role prompt\n")
            role_prompts = ""
        
        # single agent baseline 或 single agent with multi role play
        if self.task_type == "AUT":
            task_prompt = f"{role_prompts}Please provide 5 innovative and original uses for '{item}'. {discussion_prompt}"
        elif self.task_type == "Scientific":
            task_prompt = f"{role_prompts}Please provide 5 innovative solutions for the following scientific problem: {item}. {discussion_prompt}"
        elif self.task_type == "Instances":
            task_prompt = f"{role_prompts}Please provide 5 creative examples for '{item}'. {discussion_prompt}"
        elif self.task_type == "Similarities":
            task_prompt = f"{role_prompts}Please analyze the following similarity and provide 5 creative perspectives: {item}. {discussion_prompt}"
        
        return task_prompt
    

    def extract_responses(self, content):
        import re
        split_parts = re.split(r'(\d+\.)', content)
        uses = []
        for i in range(1, len(split_parts), 2):
            if i + 1 < len(split_parts):
                content_part = split_parts[i + 1]
                full_item = content_part.strip()
                uses.append(full_item)
        return uses
    
    def extract_responses_multi(self, content):
        import re
        split_parts = re.split(r'(\d+\.)', content)
        uses = []
        for i in range(1, len(split_parts), 2):
            if i + 1 < len(split_parts):
                content_part = split_parts[i + 1].strip()
                if content_part.startswith(("The Environmentalist", "The Creative Professional", "The Futurist", "Environmentalist", "Creative Professional", "Futurist")):
                    continue
                full_item = content_part.strip()
                uses.append(full_item)
        return uses
    
    
    def run(self):
        start_time = time.time()
        
        """執行 OpenAI API 呼叫"""
        # if not self.test_api_connection():
        #     return None
        
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
        
        n = 0
        for item_data in examples:
            # 處理不同的資料集格式
            if isinstance(item_data, str):
                item = item_data
            elif isinstance(item_data, dict):
                if self.task_type == "AUT":
                    item = item_data.get("object", item_data.get("item", ""))
                elif self.task_type == "Scientific":
                    item = item_data.get("question", "")
                else:  # Instances 或 Similarities
                    item = item_data.get("question", item_data.get("item", item_data.get("object", "")))
            else:
                print(f"❌ 不支援的資料格式: {type(item_data)}")
                continue
            
            if not item:
                print(f"❌ 空白項目，跳過")
                continue
                
            print(f"📋 處理項目: {item}")
            
            # 建構提示詞
            prompt = self.construct_prompt(item)
            
            # --- MODIFIED: Call the new Google API method ---
            # response = self.call_google_api(prompt, max_tokens=1000)
            response = self.call_google_api(prompt)
            
            if response:
                # --- MODIFIED: Changed agent name for logging ---
                all_responses[item] = {
                    "Google_Topline": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
                }
                
                # 提取並儲存最終結果
                if self.multi_role_prompt:
                    extracted = self.extract_responses_multi(response)
                else:
                    extracted = self.extract_responses(response)
                # --- MODIFIED: Changed agent name in final results ---
                agent_name = "Google_Topline"
                if self.task_type == "AUT":
                    final_results.append({
                        "item": item,
                        "uses": extracted,
                        "Agent": agent_name
                    })
                elif self.task_type == "Scientific":
                    final_results.append({
                        "question": item,
                        "answer": extracted,
                        "Agent": agent_name
                    })
                else:
                    final_results.append({
                        "question": item,
                        "answer": extracted,
                        "Agent": agent_name
                    })
            else:
                # print(f"❌ 項目 {item} 處理失敗")
                all_responses[item] = {
                    "Google_Topline": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "ERROR"}
                    ]
                }
                
                # 提取並儲存最終結果
                # extracted = self.extract_responses(response)
                # --- MODIFIED: Changed agent name in final results ---
                agent_name = "Google_Topline"
                if self.task_type == "AUT":
                    final_results.append({
                        "item": item,
                        "uses": [],
                        "Agent": agent_name
                    })
                elif self.task_type == "Scientific":
                    final_results.append({
                        "question": item,
                        "answer": [],
                        "Agent": agent_name
                    })
                else:
                    final_results.append({
                        "question": item,
                        "answer": [],
                        "Agent": agent_name
                    })

            if n%5==4:
                output_filename = self.save_results(all_responses, final_results, len(examples))
                print(f"Sleep Time ... 😴")
                time.sleep(60)
            n += 1
        
        # 儲存結果
        output_filename = self.save_results(all_responses, final_results, len(examples))
        print(f"✅ 任務完成，結果已儲存至: {output_filename}")

        end_time = time.time()
        total_time = end_time - start_time
        time_delta = timedelta(seconds=total_time)
        with open("run_time.txt", "a", encoding="utf-8") as f:
            f.write(f"===\n\nFile name: {output_filename}\n")
            f.write(f"總秒數: {total_time}\n")
            f.write(f"時間差: {time_delta}\n\n")
        
        return output_filename
    
    def save_results(self, all_responses, final_results, amount_of_data):
        """儲存結果檔案"""
        # 使用 UTC+8 時區（台灣時間）
        taipei_tz = pytz.timezone('Asia/Taipei')
        taipei_time = datetime.now(taipei_tz)
        
        # 修改為與其他檔案一致的格式：MMDD-HHMM
        current_date = taipei_time.strftime("%m%d")
        formatted_time = taipei_time.strftime("%H%M")
        
        # 建立檔案名稱
        model_name_for_file = self.model_name.replace("-", "_").replace(".", "_")
        # --- MODIFIED: Updated filename to reflect Google model usage ---
        
        if self.multi_role_prompt:
            base_filename = f"{self.task_type}_google_topline_1_1_{model_name_for_file}_MultiRole_{current_date}-{formatted_time}_{amount_of_data}"
        else:
            base_filename = f"{self.task_type}_google_topline_1_1_{model_name_for_file}_SingleRole_{current_date}-{formatted_time}_{amount_of_data}"
        
        # --- MODIFIED: Updated output directory to 'google_agent' ---
        results_base_path = Path(__file__).parent.parent.parent / "Results" / self.task_type / "Output" / "google_agent"
        results_base_path.mkdir(parents=True, exist_ok=True)
        
        # 儲存對話記錄
        chat_log_filename = f"{base_filename}_chat_log.json"
        chat_log_path = results_base_path / chat_log_filename
        
        with open(chat_log_path, 'w', encoding='utf-8') as f:
            json.dump(all_responses, f, indent=2, ensure_ascii=False)
        
        # 儲存最終結果
        final_filename = f"{base_filename}.json"
        final_path = results_base_path / final_filename
        
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 對話記錄已儲存: {chat_log_path}")
        print(f"💾 最終結果已儲存: {final_path}")
        
        return final_filename.replace('.json', '')

def main():
    # --- MODIFIED: Updated description for argparse ---
    parser = argparse.ArgumentParser(description="使用 Google Generative AI API 進行創造性任務評估")
    parser.add_argument("-d", "--dataset", required=True, help="資料集檔案路徑")
    parser.add_argument("-t", "--type", choices=["AUT", "Scientific", "Similarities", "Instances"], 
                       required=True, help="任務類型")
    parser.add_argument("-p", "--prompt", type=int, default=1, help="提示詞編號 (1-5)")
    # --- MODIFIED: Updated default model and help text ---
    parser.add_argument("--model", default="gemini-1.5-flash-latest", help="Google Generative AI 模型名稱")
    parser.add_argument("-e", "--eval_mode", action="store_true", default=False, help="執行評估模式")
    parser.add_argument("-a", "--multi_role_prompt", action="store_true", help="Agent mode")

    args = parser.parse_args()
    
    print(f"args.multi_role_prompt: {args.multi_role_prompt}")
    # --- MODIFIED: Instantiate the new GoogleToplineRunner class ---
    runner = GoogleToplineRunner(args.dataset, args.type, args.prompt, args.model, args.multi_role_prompt)
    topline_output = runner.run()
    # topline_output = "/workspace/LLM-discussion/LLM-discussion-reproduce/Results/AUT/Output/google_agent/AUT_google_topline_1_1_gemini_2_5_pro_MultiRole_0920-1527_100"
    
    if args.eval_mode and topline_output:
        # 整合原有的評估系統
        # evaluation_root = Path(__file__).parent.parent / 'Evaluation'
        evaluation_root = Path(__file__).parent.parent.parent / 'Evaluation'
        sys.path.append(str(evaluation_root))
        from auto_grade_final import auto_grade
        
        # 呼叫評估
        eval_args = SimpleNamespace(
            version="4", 
            input_file=topline_output,
            type="sampling", 
            sample=3, 
            task=args.type, 
            output="y"
        )
        auto_grade(eval_args)

if __name__ == "__main__":
    main()
