import argparse
import sys
import os
import json
import torch
import requests
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModelForCausalLM
import pytz

class VanillaQwenRunner:
    """使用原始模型進行創造性任務（無 Persona Steering）"""
    
    def __init__(self, dataset_file, task_type, prompt_id, model_name="llama"):
        self.dataset_file = dataset_file
        self.task_type = task_type
        self.prompt_id = prompt_id
        
        # 模型名稱對應
        model_mapping = {
            "qwen": "Qwen/Qwen2.5-7B-Instruct",
            "llama": "meta-llama/Llama-3.1-8B-Instruct",
            "gemma": "google/gemma-2-9b-it"
        }
        
        # 自動對應完整模型名稱
        if model_name.lower() in model_mapping:
            self.model_name = model_mapping[model_name.lower()]
        else:
            # 如果輸入的已經是完整名稱，直接使用
            self.model_name = model_name
        
        # 判斷模型類型
        if "qwen" in self.model_name.lower():
            self.model_type = "qwen"
        elif "llama" in self.model_name.lower():
            self.model_type = "llama"
        elif "gemma" in self.model_name.lower():
            self.model_type = "gemma"
        else:
            self.model_type = "unknown"
        
        print(f"🔍 檢測到模型類型: {self.model_type}")
        
        # 如果是 Gemma 模型，使用 API，否則載入本地模型
        if self.model_type == "gemma":
            print(f"🌐 使用 Gemma API (Port 8002) 代替本地模型載入")
            self.api_url = "http://localhost:8002"
            self.use_api = True
            self.tokenizer = None
            self.model = None
            
            # 測試 API 連線
            if not self.test_api_connection():
                print("❌ Gemma API 連線失敗！請確保 API 服務正在執行")
                raise Exception("Gemma API 連線失敗")
            else:
                print("✅ Gemma API 連線成功")
        else:
            print(f"🤖 載入本地模型: {self.model_name}")
            self.use_api = False
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ 模型載入完成")
    
    def load_dataset(self):
        """載入資料集"""
        with open(self.dataset_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def test_api_connection(self):
        """測試 Gemma API 連線"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
#     def construct_prompt(self, item):
#         """建構提示詞"""
#         base_prompts = {
#             1: "請盡可能多樣化和創造性地回答。",
#             2: "無限制地擁抱創造力的流動，提供意想不到的連接。",
#             3: "請從不同角度思考，考慮最不尋常或創新的想法。",
#             4: "請提供獨特的見解，專注於創新的想法和解決方案。",
#             5: "請使用你的創造力和智慧來提供最佳解決方案。"
#         }
        
#         discussion_prompt = base_prompts.get(self.prompt_id, base_prompts[1])
        
#         multi_role_play = True
#         # 多角色扮演提示詞
#         if multi_role_play:
#             role_prompts = """
# 你需要以三個不同的專業角色來思考和回答這個問題：

# 1. 環保主義者 (Environmentalist)：
# 專業領域：永續性與環境健康
# 角色使命：倡導環保解決方案，促進永續發展並保護地球。引導我們考慮想法的環境影響，推動有助於地球健康的創新。

# 2. 創意專業人士 (Creative Professional)：
# 專業領域：美學、敘事與情感
# 角色使命：以藝術敏感度和敘事情感掌握為專案注入美感和深度。挑戰我們進行表達性思考，確保解決方案不僅解決問題，還能在人性層面產生共鳴。

# 3. 未來學家 (Futurist)：
# 專業領域：新興技術與未來情境
# 角色使命：啟發我們超越現在思考，考慮新興技術和潛在未來情境。挑戰我們設想想法的未來影響，確保它們具有創新性、前瞻性，並準備好迎接未來挑戰。

# 請以這三個角色的觀點分別提供回答，每個角色都要體現其專業特色和思考方式。
# """
#         else:
#             role_prompts = ""
        
#         # single agent baseline 或 single agent with multi role play
#         if self.task_type == "AUT":
#             task_prompt = f"{role_prompts}請為「{item}」提供5個創新和原創的用途。{discussion_prompt}"
#         elif self.task_type == "Scientific":
#             task_prompt = f"{role_prompts}請為以下科學問題提供5個創新的解決方案：{item}。{discussion_prompt}"
#         elif self.task_type == "Instances":
#             task_prompt = f"{role_prompts}請為「{item}」提供5個創造性的範例。{discussion_prompt}"
#         elif self.task_type == "Similarities":
#             task_prompt = f"{role_prompts}請分析以下相似性並提供5個創造性的觀點：{item}。{discussion_prompt}"
        
#         return task_prompt

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
        
        MULTI_ROLE_PLAY = True
        # 多角色扮演提示詞
        if MULTI_ROLE_PLAY:
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
        
        print(f"📝 Constructed Prompt: {task_prompt}")
        return task_prompt
        
    def generate_response(self, prompt, max_tokens=1000):
        """使用模型產生回應 - 支援本地模型和 Gemma API"""
        try:
            if self.use_api:  # 使用 Gemma API
                return self.generate_api_response(prompt, max_tokens)
            else:  # 使用本地模型
                return self.generate_local_response(prompt, max_tokens)
        except Exception as e:
            print(f"❌ 生成回應時發生錯誤: {e}")
            return None
    
    def generate_api_response(self, prompt, max_tokens=1000):
        """使用 Gemma API 產生回應"""
        try:
            payload = {
                "user_input": prompt,
                "max_tokens": max_tokens,
                "session_id": "baseline_model_session"
            }
            
            response = requests.post(
                f"{self.api_url}/chat",
                json=payload,
                timeout=60  # 60秒超時
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                print(f"❌ API 呼叫失敗，狀態碼: {response.status_code}")
                print(f"❌ 回應內容: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ API 呼叫錯誤: {e}")
            return None
    
    def generate_local_response(self, prompt, max_tokens=1000):
        """使用本地模型產生回應"""
        try:
            # 格式化對話
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 編碼輸入
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # 生成回應
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # 解碼回應
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"❌ 本地模型生成錯誤: {e}")
            return None

    def extract_responses(self, content):
        """提取回應內容"""
        import re
        
        # 優化的正規表達式，能處理多種格式
        patterns = [
            # 格式1: **1.** **Title** (新增！這是你遇到的問題格式)
            r'\*\*(\d+)\.\*\*\s*\*\*([^*]+?)\*\*\s*(.*?)(?=\*\*\d+\.\*\*|\n\n|$)',
            # 格式2: **1. Title:** (內容)
            r'\*\*(\d+)\.\s*([^*]+?)\*\*:?\s*(.*?)(?=\*\*\d+\.|$)',
            # 格式3: 1. **Title:** (內容)  
            r'(\d+)\.\s*\*\*([^*]+?)\*\*:?\s*(.*?)(?=\d+\.\s*\*\*|$)',
            # 格式4: 數字開頭的一般項目
            r'(\d+)\.\s*([^\n]*?)\n(.*?)(?=\d+\.|$)'
        ]
        
        responses = []
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            if matches:
                for match in matches:
                    if len(match) == 3:  # (number, title, content)
                        title = match[1].strip()
                        body = match[2].strip()
                        
                        # 清理標題和內容
                        if title:
                            full_item = f"**{title}**"
                            if body:
                                # 清理內容開頭的換行和多餘字符
                                body = re.sub(r'^\n+', '', body)
                                body = body.strip()
                                if body:
                                    full_item += f": {body}"
                            responses.append(full_item)
                break  # 如果找到匹配，就不嘗試其他模式
        
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
        
        return responses  # 限制最多10個回應
    
    def run(self):
        """執行原始模型推理"""
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
            
            # 生成回應
            response = self.generate_response(prompt, max_tokens=1000)
            
            if response:
                # 儲存對話記錄（模擬 multi-agent 格式）
                agent_name = self.model_type.capitalize()
                all_responses[item] = {
                    agent_name: [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
                }
                
                # 提取並儲存最終結果
                agent_name = self.model_type.capitalize()
                extracted = self.extract_responses(response)
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
        
        # 修改為與其他檔案一致的格式：MMDD-HHMM
        current_date = taipei_time.strftime("%m%d")
        formatted_time = taipei_time.strftime("%H%M")
        
        # 建立檔案名稱（符合評估系統格式）
        model_prefix = self.model_type.capitalize()
        base_filename = f"{self.task_type}_{self.model_type}_1_1_{model_prefix}_{model_prefix}_{self.model_type}_{current_date}-{formatted_time}_{amount_of_data}"
        
        # 評估系統期待的路徑結構：Results/{task}/Output/{agent}_agent/
        results_base_path = Path(__file__).parent.parent / "Results" / self.task_type / "Output" / f"{self.model_type}_agent"
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
    parser = argparse.ArgumentParser(description="使用原始模型進行創造性任務評估")
    parser.add_argument("-d", "--dataset", required=True, help="資料集檔案路徑")
    parser.add_argument("-t", "--type", choices=["AUT", "Scientific", "Similarities", "Instances"], 
                       required=True, help="任務類型")
    parser.add_argument("-p", "--prompt", type=int, default=1, help="提示詞編號 (1-5)")
    parser.add_argument("-m", "--model", choices=["qwen", "llama", "gemma"], default="llama", help="模型類型 (qwen 或 llama 或 gemma)")
    parser.add_argument("-e", "--eval_mode", action="store_true", default=False, help="執行評估模式")
    
    args = parser.parse_args()
    
    # 建立並執行 Vanilla 模型執行器
    runner = VanillaQwenRunner(args.dataset, args.type, args.prompt, args.model)
    vanilla_output = runner.run()
    
    if args.eval_mode and vanilla_output:
        # 整合原有的評估系統
        evaluation_root = Path(__file__).parent.parent / 'Evaluation'
        sys.path.append(str(evaluation_root))
        from auto_grade_final import auto_grade
        
        # 呼叫評估
        eval_args = SimpleNamespace(
            version="4", 
            input_file=vanilla_output,
            type="sampling", 
            sample=3, 
            task=args.type, 
            output="y"
        )
        auto_grade(eval_args)

if __name__ == "__main__":
    main()