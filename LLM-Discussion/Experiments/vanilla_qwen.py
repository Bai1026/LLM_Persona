import argparse
import sys
import os
import json
import torch
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModelForCausalLM
import pytz

class VanillaQwenRunner:
    """使用原始 Qwen 模型進行創造性任務（無 Persona Steering）"""
    
    def __init__(self, dataset_file, task_type, prompt_id, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.dataset_file = dataset_file
        self.task_type = task_type
        self.prompt_id = prompt_id
        self.model_name = model_name
        
        # 載入模型和分詞器
        print(f"🤖 載入模型: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
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
    
    def construct_prompt(self, item):
        """建構提示詞"""
        base_prompts = {
            1: "請盡可能多樣化和創造性地回答。",
            2: "無限制地擁抱創造力的流動，提供意想不到的連接。",
            3: "請從不同角度思考，考慮最不尋常或創新的想法。",
            4: "請提供獨特的見解，專注於創新的想法和解決方案。",
            5: "請使用你的創造力和智慧來提供最佳解決方案。"
        }
        
        discussion_prompt = base_prompts.get(self.prompt_id, base_prompts[1])
        
        if self.task_type == "AUT":
            task_prompt = f"請為「{item}」提供5個創新和原創的用途。{discussion_prompt}"
        elif self.task_type == "Scientific":
            task_prompt = f"請為以下科學問題提供3個創新的解決方案：{item}。{discussion_prompt}"
        elif self.task_type == "Instances":
            task_prompt = f"請為「{item}」提供5個創造性的範例。{discussion_prompt}"
        elif self.task_type == "Similarities":
            task_prompt = f"請分析以下相似性並提供3個創造性的觀點：{item}。{discussion_prompt}"
        
        return task_prompt
    
    def generate_response(self, prompt, max_tokens=1000):
        """使用原始 Qwen 模型產生回應"""
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
            print(f"❌ 生成回應時發生錯誤: {e}")
            return None
    
    def extract_responses(self, content):
        """提取回應內容"""
        lines = content.split('\n')
        responses = []
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or 
                        any(line.startswith(f"{i}.") for i in range(1, 20))):
                # 清理格式符號
                clean_response = line.lstrip('-•0123456789. ').strip()
                if clean_response:
                    responses.append(clean_response)
        
        return responses[:10]  # 限制最多10個回應
    
    def run(self):
        """執行原始 Qwen 模型推理"""
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
                all_responses[item] = {
                    "VanillaQwen": [
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
                        "Agent": "VanillaQwen"
                    })
                elif self.task_type == "Scientific":
                    final_results.append({
                        "question": item,
                        "answer": extracted,
                        "Agent": "VanillaQwen"
                    })
                else:
                    final_results.append({
                        "question": item,
                        "answer": extracted,
                        "Agent": "VanillaQwen"
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
        base_filename = f"{self.task_type}_vanilla_qwen_1_1_Qwen25_VanillaQwen_vanilla_{current_date}-{formatted_time}_{amount_of_data}"
        
        # 評估系統期待的路徑結構：Results/{task}/Output/{agent}_agent/
        results_base_path = Path(__file__).parent.parent / "Results" / self.task_type / "Output" / "vanilla_agent"
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
    parser = argparse.ArgumentParser(description="使用原始 Qwen 模型進行創造性任務評估")
    parser.add_argument("-d", "--dataset", required=True, help="資料集檔案路徑")
    parser.add_argument("-t", "--type", choices=["AUT", "Scientific", "Similarities", "Instances"], 
                       required=True, help="任務類型")
    parser.add_argument("-p", "--prompt", type=int, default=1, help="提示詞編號 (1-5)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Qwen 模型名稱")
    parser.add_argument("-e", "--eval_mode", action="store_true", default=False, help="執行評估模式")
    
    args = parser.parse_args()
    
    # 建立並執行 Vanilla Qwen 執行器
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
