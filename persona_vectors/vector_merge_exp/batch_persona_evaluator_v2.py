#!/usr/bin/env python3
"""
批次 Persona 評估器
這個腳本會讀取 JSON 檔案中的 QA pairs，並對每個進行 trait 評估
"""

import os
import json
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
from rich import print
from rich.progress import track
from rich.table import Table
from rich.console import Console
import statistics

from trait_eval_prompt import analytical_eval_prompt, creative_eval_prompt, environmental_eval_prompt, futurist_eval_prompt, comprehensive_eval_prompt_v2

eval_prompt_dict = {
    "analytical": analytical_eval_prompt,
    "creative": creative_eval_prompt,
    "environmental": environmental_eval_prompt,
    "futurist": futurist_eval_prompt,
}


def evaluate_trait_comprehensive(question: str, answer: str, eval_model: str = "gemini") -> dict:
    """
    使用指定的 AI 模型進行綜合評估，一次性分配 100 分給五個特質
    eval_model: "gemini" 或 "gpt"
    """
    prompt = comprehensive_eval_prompt_v2.format(question=question, answer=answer)
    
    try:
        if eval_model == "gemini":
            # 使用 Gemini 模型
            # model = genai.GenerativeModel('gemini-2.5-pro')
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            score_text = response.text.strip()
        elif eval_model == "gpt":
            # 使用 OpenAI GPT 模型
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator analyzing personality traits in responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            score_text = response.choices[0].message.content.strip()
        else:
            raise ValueError(f"不支援的評估模型: {eval_model}")
        
        # 解析回應
        # 解析分數 - 尋找 "Scores" 部分
        scores = {}
        total_score = 0
        
        # 先嘗試找到 "Scores" 或 "scores" 區段
        lines = score_text.split('\n')
        score_section_started = False
        
        for line in lines:
            line = line.strip()
            
            # 檢查是否進入分數區段
            if line.lower() in ['scores', 'score:', 'scores:', 'final scores', 'final scores:']:
                score_section_started = True
                continue
            
            # 如果在分數區段，或者沒找到分數區段但行包含分數格式
            if (score_section_started or not any('scores' in l.lower() for l in lines)) and ':' in line:
                try:
                    trait, score_str = line.split(':', 1)
                    trait = trait.strip().lower()
                    score_str = score_str.strip()
                    
                    # 移除可能的額外文字，只保留數字
                    import re
                    score_match = re.search(r'\d+(?:\.\d+)?', score_str)
                    if score_match:
                        score = float(score_match.group())
                        
                        # 標準化特質名稱
                        trait_mapping = {
                            'analytical': 'analytical',
                            'analytical thinker': 'analytical',
                            'creative': 'creative',
                            'creative professional': 'creative',
                            'environmental': 'environmental',
                            'environmentalist': 'environmental',
                            'futurist': 'futurist',
                            'empathetic': 'empathetic',
                            'empathetic counselor': 'empathetic'
                        }
                        
                        normalized_trait = trait_mapping.get(trait, trait)
                        scores[normalized_trait] = score
                        total_score += score
                    else:
                        print(f"無法解析分數: {score_str} for trait {trait}")
                except ValueError as e:
                    print(f"無法解析分數: {score_str} for trait {trait} - {e}")
        
        # 如果沒有找到任何分數，嘗試更寬鬆的解析
        if not scores:
            print("嘗試更寬鬆的分數解析...")
            import re
            # 尋找類似 "trait: number" 的模式
            pattern = r'(analytical|creative|environmental|futurist|empathetic).*?(\d+(?:\.\d+)?)'
            matches = re.findall(pattern, score_text, re.IGNORECASE)
            
            for trait, score_str in matches:
                try:
                    score = float(score_str)
                    scores[trait.lower()] = score
                    total_score += score
                except ValueError:
                    continue
        
        # 檢查總分是否為 100
        if abs(total_score - 100) > 1:  # 允許小誤差
            print(f"警告: 總分不等於 100，實際為 {total_score}")
            # 正規化到 100
            if total_score > 0:
                for trait in scores:
                    scores[trait] = (scores[trait] / total_score) * 100
        
        return scores

    except Exception as e:
        print(f"綜合評估時發生錯誤: {e}")
        return {}


def evaluate_trait(question: str, answer: str, trait: str, eval_model: str = "gemini") -> float:
    """
    使用指定的 AI 模型評估模型回答中的特質
    eval_model: "gemini" 或 "gpt"
    """
    if trait not in eval_prompt_dict:
        raise ValueError(f"Trait '{trait}' is not supported.")

    prompt = eval_prompt_dict[trait].format(question=question, answer=answer)
    
    try:
        if eval_model == "gemini":
            # 使用 Gemini 模型
            model = genai.GenerativeModel('gemini-2.5-pro')
            response = model.generate_content(prompt)
            score_text = response.text.strip()
        elif eval_model == "gpt":
            # 使用 OpenAI GPT 模型
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator analyzing personality traits in responses. Return only a numerical score."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            score_text = response.choices[0].message.content.strip()
        else:
            raise ValueError(f"不支援的評估模型: {eval_model}")
        
        # 嘗試解析分數
        if score_text == "REFUSAL":
            return -1.0  # 表示拒絕回答
        
        try:
            score = float(score_text)
            return score
        except ValueError:
            print(f"無法解析分數: {score_text}")
            return -1.0

    except Exception as e:
        print(f"評估時發生錯誤: {e}")
        return -1.0


def load_persona_data(json_file_path: str) -> dict:
    """
    載入 persona 資料從 JSON 檔案
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_all_personas(data: dict, traits: list, evaluation_mode: str = "separate", eval_model: str = "gemini") -> dict:
    """
    評估所有 persona 的所有 traits
    evaluation_mode: "separate" 或 "comprehensive"
    eval_model: "gemini" 或 "gpt"
    """
    results = {}
    console = Console()
    
    # 新的資料格式：從 conversations 陣列中取得對話
    conversations = data.get("conversations", [])
    print(f"找到 {len(conversations)} 個對話")

    for conversation in conversations:
        conversation_id = conversation.get("id", "unknown")
        topic = conversation.get("topic", "unknown_topic")
        
        print(f"\n處理對話 #{conversation_id}: {topic}")
        results[f"conversation_{conversation_id}"] = {}
        
        # 從新的資料格式中取得問題和回答
        question = conversation.get("user_input") or conversation.get("question")
        answer = conversation.get("response")
        
        if not question or not answer:
            print(f"  警告: 對話 #{conversation_id} 沒有完整的 QA pair")
            continue
        
        print(f"  問題: {question[:100]}...")
        print(f"  回答: {answer[:100]}...")
        
        conversation_key = f"conversation_{conversation_id}"
        
        if evaluation_mode == "comprehensive":
            # 綜合評估模式
            print(f"  使用綜合評估模式（總分 100）- 評估模型: {eval_model}")
            scores = evaluate_trait_comprehensive(question, answer, eval_model)
            results[conversation_key] = scores
            for trait, score in scores.items():
                print(f"    {trait}: {score:.1f}")
        else:
            # 分開評估模式
            print(f"  使用分開評估模式（每個特質 0-100）- 評估模型: {eval_model}")
            for trait in track(traits, description=f"評估對話 #{conversation_id} 的特質"):
                score = evaluate_trait(question, answer, trait, eval_model)
                results[conversation_key][trait] = score
                print(f"    {trait}: {score}")
    
    return results


def calculate_average_scores(results: dict) -> dict:
    """
    計算每個特質的平均分數（所有 persona 的平均）
    """
    trait_averages = {}
    
    # 取得所有特質
    all_traits = set()
    for traits_scores in results.values():
        all_traits.update(traits_scores.keys())
    
    # 計算每個特質的平均分數
    for trait in all_traits:
        trait_scores = []
        for persona_name, traits_scores in results.items():
            if trait in traits_scores and traits_scores[trait] >= 0:
                trait_scores.append(traits_scores[trait])
        
        if trait_scores:
            trait_averages[trait] = statistics.mean(trait_scores)
        else:
            trait_averages[trait] = 0.0
    
    return trait_averages


def display_results(results: dict, trait_averages: dict, evaluation_mode: str = "separate"):
    """
    顯示評估結果
    """
    console = Console()
    
    # 根據評估模式設定標題
    if evaluation_mode == "comprehensive":
        title = "對話特質評估結果 (綜合評估 - 總分 100)"
    else:
        title = "對話特質評估結果 (分開評估 - 每個特質 0-100)"
    
    # 建立詳細結果表格
    table = Table(title=title)
    table.add_column("對話", justify="left", style="cyan")
    
    # 取得所有 traits
    all_traits = set()
    for traits_scores in results.values():
        all_traits.update(traits_scores.keys())
    all_traits = sorted(all_traits)
    
    for trait in all_traits:
        table.add_column(trait, justify="center")
    
    # 加入資料行
    for conversation_key in sorted(results.keys()):
        row = [conversation_key]
        for trait in all_traits:
            score = results[conversation_key].get(trait, -1)
            if score >= 0:
                row.append(f"{score:.1f}")
            else:
                row.append("N/A")
        table.add_row(*row)
    
    console.print(table)
    
    # 建立特質平均分數表格
    print("\n")
    average_table = Table(title="各特質平均分數")
    average_table.add_column("特質", justify="left", style="cyan")
    average_table.add_column("平均分數", justify="center", style="bold green")
    
    # 按分數排序顯示
    sorted_traits = sorted(trait_averages.items(), key=lambda x: x[1], reverse=True)
    for trait, avg_score in sorted_traits:
        average_table.add_row(trait, f"{avg_score:.1f}")
    
    console.print(average_table)
    
    for trait, avg_score in sorted_traits:
        print(f"{trait}: {avg_score:.1f}")


def save_results(results: dict, trait_averages: dict, output_file: str, evaluation_mode: str = "separate"):
    """
    儲存結果到 JSON 檔案
    """
    output_data = {
        "evaluation_mode": evaluation_mode,
        "detailed_scores": results,
        "trait_averages": trait_averages,
        "trait_ranking": sorted(trait_averages.items(), key=lambda x: x[1], reverse=True)
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n結果已儲存到: {output_file}")


def main(type_name, choice, eval_model):
    # 載入環境變數
    load_dotenv()
    
    # 檢查 API keys
    if eval_model == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
    elif eval_model == "gpt":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
    else:
        raise ValueError("eval_model 必須是 'gemini' 或 'gpt'")
    
    json_file_path = f'./persona_trait_data/neutral_task/{type_name}.json'
    
    # 要評估的特質
    traits_to_evaluate = ["analytical", "creative", "environmental", "futurist"]
    
    print(f"開始批次 Persona 評估... (使用 {eval_model.upper()} 模型)")
    
    # 讓使用者選擇評估模式
    print("\n請選擇評估模式:")
    print("1. 分開評估 (每個特質分別評分 0-100)")
    print("2. 綜合評估 (五個特質總共分配 100 分)")

    file_prefix = json_file_path.split('/')[-1].strip('.json')
    print(file_prefix)

    # while True:
        # choice = input("請輸入選項 (1 或 2): ").strip()
    if choice == "1":
        evaluation_mode = "separate"
        output_file = f'./persona_trait_data/neutral_task/Result/{file_prefix}_evaluation_results_separate_{eval_model}.json'
        print(output_file)
        # break

    elif choice == "2":
        evaluation_mode = "comprehensive"
        output_file = f'./persona_trait_data/neutral_task/Result/{file_prefix}_evaluation_results_comprehensive_{eval_model}.json'
        print(output_file)
        # break
    else:
        print("無效選項，請輸入 1 或 2")
    
    print(f"\n使用評估模式: {'分開評估' if evaluation_mode == 'separate' else '綜合評估'}")
    print(f"使用評估模型: {eval_model.upper()}")
    
    # 載入資料
    print(f"載入資料從: {json_file_path}")
    data = load_persona_data(json_file_path)
    
    # 檢查新的資料格式
    if "conversations" in data:
        conversations_count = len(data["conversations"])
        print(f"找到 {conversations_count} 個對話")
    else:
        print("警告: 資料格式不符合預期，請檢查 JSON 檔案")
        return

    # 評估所有 personas
    results = evaluate_all_personas(data, traits_to_evaluate, evaluation_mode, eval_model)
    
    # 計算特質平均分數
    trait_averages = calculate_average_scores(results)
    
    # 顯示結果
    display_results(results, trait_averages, evaluation_mode)
    
    # 儲存結果
    save_results(results, trait_averages, output_file, evaluation_mode)


if __name__ == "__main__":
    type_list = [
        # 'cre_env_fut_fut',
        # 'cre_env',
        # 'env_ana',
        # 'env',
        # 'baseline',
        # 'multi_prompt'
        'cre',
        'ana_fut',
        'cre_ana',
        'cre_fut',
        'env_fut',
    ]

    # 讓使用者選擇評估模式
    choice = input("請輸入評估模式選項 (1 或 2):\n 1 for separate, 2 for comprehensive\n ").strip()
    
    # 讓使用者選擇評估模型
    print("\n請選擇評估模型:")
    print("1. Gemini (gemini-2.5-pro)")
    print("2. GPT (gpt-4o-mini)")
    
    model_choice = input("請輸入模型選項 (1 或 2): ").strip()
    
    if model_choice == "1":
        eval_model = "gemini"
    elif model_choice == "2":
        eval_model = "gpt"
    else:
        print("無效選項，預設使用 Gemini")
        eval_model = "gemini"
    
    print(f"\n已選擇評估模型: {eval_model.upper()}")

    for type_name in type_list:
        print(f"Processing type: {type_name}")
        main(type_name, choice, eval_model)
