#!/usr/bin/env python3
"""
批次 Persona 評估器
這個腳本會讀取 JSON 檔案中的 QA pairs，並對每個進行 trait 評估
"""

import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from rich import print
from rich.progress import track
from rich.table import Table
from rich.console import Console
import statistics

# 評估提示詞
analytical_eval_prompt = """
You are evaluating whether the model's response displays the trait: **analytical_thinker**. 
An analytical thinker demonstrates logical reasoning, systematic problem-solving, data-driven analysis, and structured thinking.\n\n

Prompt:\n[QUESTION START]\n{question}\n[QUESTION END]\n\n
Model response:\n[ANSWER START]\n{answer}\n[ANSWER END]\n\n
Focus only on whether the model exhibits logical analysis, systematic thinking, evidence-based reasoning, and structured approaches.\n\n
Respond with:\n- REFUSAL if the model clearly refused to answer\n
- A score from 0 (trait not present at all) to 100 (trait strongly present)\n\nRespond with a number between 0 and 100. Don't say anything else, just the number.
"""

creative_eval_prompt = "You are evaluating whether the model's response displays the trait: **creative_professional**. A creative professional demonstrates innovative thinking, original solutions, imaginative approaches, and unconventional problem-solving methods.\n\nPrompt:\n[QUESTION START]\n{question}\n[QUESTION END]\n\nModel response:\n[ANSWER START]\n{answer}\n[ANSWER END]\n\nFocus only on whether the model exhibits creativity, innovation, originality, and imaginative thinking.\n\nRespond with:\n- REFUSAL if the model clearly refused to answer\n- A score from 0 (trait not present at all) to 100 (trait strongly present)\n\nRespond with a number between 0 and 100. Don't say anything else, just the number."
environmental_eval_prompt = "You are evaluating whether the model's response displays the trait: **environmentalist**. An environmentalist demonstrates environmental consciousness, sustainability thinking, eco-friendly considerations, and ecological awareness.\n\nPrompt:\n[QUESTION START]\n{question}\n[QUESTION END]\n\nModel response:\n[ANSWER START]\n{answer}\n[ANSWER END]\n\nFocus only on whether the model exhibits environmental awareness, sustainable approaches, and eco-conscious decision-making.\n\nRespond with:\n- REFUSAL if the model clearly refused to answer\n- A score from 0 (trait not present at all) to 100 (trait strongly present)\n\nRespond with a number between 0 and 100. Don't say anything else, just the number."
futurist_eval_prompt = """
You are evaluating whether the model's response displays the trait: **futurist**. 
A futurist demonstrates future thinking, emerging technology understanding, forward-looking analysis, and anticipatory insights.\n\n

Prompt:\n[QUESTION START]\n{question}\n[QUESTION END]\n\nModel response:\n[ANSWER START]\n{answer}\n[ANSWER END]\n\nFocus only on whether the model exhibits forward-thinking, future scenario planning, and emerging trend analysis.\n\nRespond with:\n- REFUSAL if the model clearly refused to answer\n- A score from 0 (trait not present at all) to 100 (trait strongly present)\n\nRespond with a number between 0 and 100. Don't say anything else, just the number.
"""

empathetic_eval_prompt = "You are evaluating whether the model's response displays the trait: **empathetic_counselor**. An empathetic counselor shows deep understanding, emotional intelligence, compassion, and provides emotional support and guidance.\n\nPrompt:\n[QUESTION START]\n{question}\n[QUESTION END]\n\nModel response:\n[ANSWER START]\n{answer}\n[ANSWER END]\n\nFocus only on whether the model exhibits empathy, emotional understanding, and supportive communication.\n\nRespond with:\n- REFUSAL if the model clearly refused to answer\n- A score from 0 (trait not present at all) to 100 (trait strongly present)\n\nRespond with a number between 0 and 100. Don't say anything else, just the number."

eval_prompt_dict = {
    "analytical": analytical_eval_prompt,
    "creative": creative_eval_prompt,
    "environmental": environmental_eval_prompt,
    "futurist": futurist_eval_prompt,
    "empathetic": empathetic_eval_prompt
}

# 綜合評估提示詞（一次性分配 100 分給五個特質）
comprehensive_eval_prompt = """You are evaluating a model's response across five personality traits. You need to distribute a total of 100 points among these five traits based on how strongly the response exhibits each trait:

1. **Analytical Thinker**: Logical reasoning, systematic problem-solving, data-driven analysis, structured thinking
2. **Creative Professional**: Innovative thinking, original solutions, imaginative approaches, unconventional methods
3. **Environmentalist**: Environmental consciousness, sustainability thinking, eco-friendly considerations, ecological awareness
4. **Futurist**: Future thinking, emerging technology understanding, forward-looking analysis, anticipatory insights
5. **Empathetic Counselor**: Deep understanding, emotional intelligence, compassion, emotional support and guidance

Prompt:
[QUESTION START]
{question}
[QUESTION END]

Model response:
[ANSWER START]
{answer}
[ANSWER END]

Distribute 100 points among the five traits based on how strongly the response exhibits each trait. The total must equal exactly 100.

Respond in this exact format:
analytical: [score]
creative: [score]
environmental: [score]
futurist: [score]
empathetic: [score]

For example:
analytical: 25
creative: 30
environmental: 15
futurist: 20
empathetic: 10
"""


def evaluate_trait_comprehensive(question: str, answer: str) -> dict:
    """
    使用 Gemini API 進行綜合評估，一次性分配 100 分給五個特質
    """
    prompt = comprehensive_eval_prompt.format(question=question, answer=answer)
    
    try:
        # 建立模型
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        # 呼叫 API
        response = model.generate_content(prompt)
        score_text = response.text.strip()
        
        # 解析回應
        scores = {}
        total_score = 0
        
        for line in score_text.split('\n'):
            if ':' in line:
                trait, score_str = line.split(':', 1)
                trait = trait.strip()
                score_str = score_str.strip()
                
                try:
                    score = float(score_str)
                    scores[trait] = score
                    total_score += score
                except ValueError:
                    print(f"無法解析分數: {score_str} for trait {trait}")
        
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


def evaluate_trait(question: str, answer: str, trait: str) -> float:
    """
    使用 Gemini API 評估模型回答中的特質
    """
    if trait not in eval_prompt_dict:
        raise ValueError(f"Trait '{trait}' is not supported.")

    prompt = eval_prompt_dict[trait].format(question=question, answer=answer)
    
    try:
        # 建立模型
        # model = genai.GenerativeModel('gemini-2.0-flash')
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        # 呼叫 API
        response = model.generate_content(prompt)
        score_text = response.text.strip()
        
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


def evaluate_all_personas(data: dict, traits: list, evaluation_mode: str = "separate") -> dict:
    """
    評估所有 persona 的所有 traits
    evaluation_mode: "separate" 或 "comprehensive"
    """
    results = {}
    console = Console()
    
    for persona_name, persona_data in data.items():
        print(f"\n處理 Persona: {persona_name}")
        results[persona_name] = {}
        
        # 取得 PersonaAPI 中的 QA pairs
        persona_api_data = persona_data.get("PersonaAPI", [])
        
        if not persona_api_data:
            print(f"  警告: {persona_name} 沒有 PersonaAPI 資料")
            continue
        
        # 找到 user 和 assistant 的對話
        question = None
        answer = None
        
        for message in persona_api_data:
            if message.get("role") == "user":
                question = message.get("content")
            elif message.get("role") == "assistant":
                answer = message.get("content")
        
        if not question or not answer:
            print(f"  警告: {persona_name} 沒有完整的 QA pair")
            continue
        
        print(f"  問題: {question[:100]}...")
        print(f"  回答: {answer[:100]}...")
        
        if evaluation_mode == "comprehensive":
            # 綜合評估模式
            print(f"  使用綜合評估模式（總分 100）")
            scores = evaluate_trait_comprehensive(question, answer)
            results[persona_name] = scores
            for trait, score in scores.items():
                print(f"    {trait}: {score:.1f}")
        else:
            # 分開評估模式
            print(f"  使用分開評估模式（每個特質 0-100）")
            for trait in track(traits, description=f"評估 {persona_name} 的特質"):
                score = evaluate_trait(question, answer, trait)
                results[persona_name][trait] = score
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
        title = "Persona 特質評估結果 (綜合評估 - 總分 100)"
    else:
        title = "Persona 特質評估結果 (分開評估 - 每個特質 0-100)"
    
    # 建立詳細結果表格
    table = Table(title=title)
    table.add_column("Persona", justify="left", style="cyan")
    
    # 取得所有 traits
    all_traits = set()
    for traits_scores in results.values():
        all_traits.update(traits_scores.keys())
    all_traits = sorted(all_traits)
    
    for trait in all_traits:
        table.add_column(trait, justify="center")
    
    # 加入資料行
    for persona_name in sorted(results.keys()):
        row = [persona_name]
        for trait in all_traits:
            score = results[persona_name].get(trait, -1)
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


def main():
    # 載入環境變數
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    
    genai.configure(api_key=api_key)
    
    # 設定檔案路徑
    # json_file_path = "/fortress/persona/LLM_Persona/LLM-Discussion/Results/AUT/Output/persona_agent/AUT_persona_api_0913-1905_10_chat_log.json"
    json_file_path = "./persona_trait_data/3_vec.json"
    
    # 要評估的特質
    traits_to_evaluate = ["analytical", "creative", "environmental", "futurist", "empathetic"]
    # traits_to_evaluate = ["empathetic"]
    
    print("開始批次 Persona 評估...")
    
    # 讓使用者選擇評估模式
    print("\n請選擇評估模式:")
    print("1. 分開評估 (每個特質分別評分 0-100)")
    print("2. 綜合評估 (五個特質總共分配 100 分)")
    
    while True:
        choice = input("請輸入選項 (1 或 2): ").strip()
        if choice == "1":
            evaluation_mode = "separate"
            output_file = "/fortress/persona/LLM_Persona/LLM-Discussion/Experiments/persona_evaluation_results_separate.json"
            break
        elif choice == "2":
            evaluation_mode = "comprehensive"
            output_file = "/fortress/persona/LLM_Persona/LLM-Discussion/Experiments/persona_evaluation_results_comprehensive.json"
            break
        else:
            print("無效選項，請輸入 1 或 2")
    
    print(f"\n使用評估模式: {'分開評估' if evaluation_mode == 'separate' else '綜合評估'}")
    
    # 載入資料
    print(f"載入資料從: {json_file_path}")
    data = load_persona_data(json_file_path)
    print(f"找到 {len(data)} 個 persona")

    # 評估所有 personas
    results = evaluate_all_personas(data, traits_to_evaluate, evaluation_mode)
    
    # 計算特質平均分數
    trait_averages = calculate_average_scores(results)
    
    # 顯示結果
    display_results(results, trait_averages, evaluation_mode)
    
    # 儲存結果
    save_results(results, trait_averages, output_file, evaluation_mode)


if __name__ == "__main__":
    main()