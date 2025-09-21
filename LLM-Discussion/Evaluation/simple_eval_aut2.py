import json
import os
import re
import argparse
from pathlib import Path
from utils.openai_model import OpenAIModel
from eval_functions.eval_prompts import aut_prompts, scientific_prompts, wkct_prompts
from dotenv import load_dotenv, find_dotenv
from rich import print

load_dotenv(find_dotenv())

def parse_score_from_response(response_text):
    """
    從模型回應中提取分數
    支援多種格式：[[X]], (X), X
    """
    # 先嘗試匹配 [[X]] 格式
    pattern_double = r'\[\[(\d+)\]\]'
    match = re.search(pattern_double, response_text)
    if match:
        return int(match.group(1))
    
    # 再嘗試匹配 (X) 格式
    pattern_single = r'\((\d+)\)'
    matches = re.findall(pattern_single, response_text)
    if matches:
        return int(matches[-1])  # 取最後一個匹配
    
    # 最後嘗試匹配單純的數字
    pattern_number = r'\b([1-5])\b'
    matches = re.findall(pattern_number, response_text)
    if matches:
        return int(matches[-1])
    
    return 0  # 如果無法解析，回傳0

def extract_assistant_content(data):
    """
    從JSON資料中提取所有assistant的content
    """
    results = []
    
    for question, models in data.items():
        for model_name, conversation in models.items():
            if isinstance(conversation, list):
                for message in conversation:
                    if message.get('role') == 'assistant':
                        results.append({
                            'question': question,
                            'model': model_name,
                            'content': message.get('content', '')
                        })
    
    return results

def evaluate_response(model, content, criterion, num_samples=3):
    """
    評估單一回應的originality或elaboration
    重複評估num_samples次並計算平均分數
    """
    # 根據標準選擇prompt
    if criterion == 'originality':
        prompt = aut_prompts['originality']['sampling']
    elif criterion == 'elaboration':
        prompt = aut_prompts['elaboration']['sampling']
    else:
        raise ValueError(f"不支援的評估標準: {criterion}")
    
    # 建構完整的prompt
    full_prompt = prompt + f"\n\nThe response to evaluate is:\n{content}"
    full_prompt += """
    Please focus on the answers that is specify. 
    For instance: I would love to share these answer with you guys. **1: xxx**, **2: xxx**
    - Please only focus on the answers themselves (**1: xxx**, **2: xxx**), ignore the trivial words(I would love to share these answer with you guys.)
    - Could be more demanding.
    """
    
    messages = [{"role": "user", "content": full_prompt}]
    
    scores = []
    responses = []
    
    # 進行num_samples次評估
    for i in range(num_samples):
        try:
            # 使用不同的seed確保每次評估的變異性
            response = model.generate_response(messages=messages, seed=42+i)
            score = parse_score_from_response(response)
            scores.append(score)
            responses.append(response)
            print(f"    第 {i+1} 次評估 {criterion}: {score}")
        except Exception as e:
            print(f"第 {i+1} 次評估時發生錯誤: {e}")
            scores.append(0)
            responses.append(f"錯誤: {str(e)}")
    
    # 計算平均分數
    avg_score = sum(scores) / len(scores) if scores else 0
    
    return {
        'criterion': criterion,
        'score': avg_score,  # 平均分數
        'individual_scores': scores,  # 個別分數
        'responses': responses,  # 所有回應
        'content': content,
        'num_samples': num_samples
    }

def calculate_averages(evaluation_results):
    """
    計算originality和elaboration的平均分數
    """
    originality_scores = []
    elaboration_scores = []
    originality_individual_scores = []
    elaboration_individual_scores = []
    
    for result in evaluation_results:
        if result['criterion'] == 'originality':
            originality_scores.append(result['score'])
            if 'individual_scores' in result:
                originality_individual_scores.extend(result['individual_scores'])
        elif result['criterion'] == 'elaboration':
            elaboration_scores.append(result['score'])
            if 'individual_scores' in result:
                elaboration_individual_scores.extend(result['individual_scores'])
    
    avg_originality = sum(originality_scores) / len(originality_scores) if originality_scores else 0
    avg_elaboration = sum(elaboration_scores) / len(elaboration_scores) if elaboration_scores else 0
    
    return {
        'average_originality': avg_originality,
        'average_elaboration': avg_elaboration,
        'total_responses': len(evaluation_results) // 2,  # 每個回應評估兩個標準
        'originality_scores': originality_scores,  # 平均分數列表
        'elaboration_scores': elaboration_scores,  # 平均分數列表
        'originality_individual_scores': originality_individual_scores,  # 所有個別分數
        'elaboration_individual_scores': elaboration_individual_scores,  # 所有個別分數
        'total_evaluations': len(originality_individual_scores) + len(elaboration_individual_scores)  # 總評估次數
    }

def simple_eval(input_file):
    """
    主要評估函式
    """
    print(f"開始評估: {input_file}")
    
    # 設置OpenAI API
    # api_key = os.getenv("OPENAI_API_KEY")
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
    api_key = os.getenv("OPENAI_API_KEY")

    version = "gpt-4o-mini"
    cache_file_name = "cache_simple_eval.pickle"
    model = OpenAIModel(cache_file_name, version, api_key)
    
    # 讀取JSON檔案
    if not os.path.exists(input_file):
        print(f"找不到檔案: {input_file}")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取assistant內容
    assistant_contents = extract_assistant_content(data)
    print(f"找到 {len(assistant_contents)} 個assistant回應")
    
    # 評估每個回應
    all_results = []
    
    for i, item in enumerate(assistant_contents, 1):
        print(f"\n正在評估回應 {i}/{len(assistant_contents)}")
        print(f"問題: {item['question'][:50]}...")
        print(f"模型: {item['model']}")
        
        # 評估originality (3次取樣)
        print(f"  評估 Originality (3次取樣):")
        originality_result = evaluate_response(model, item['content'], 'originality', num_samples=3)
        print(f"  Originality 平均分數: {originality_result['score']:.3f} (個別分數: {originality_result['individual_scores']})")
        
        # 評估elaboration (3次取樣)
        print(f"  評估 Elaboration (3次取樣):")
        elaboration_result = evaluate_response(model, item['content'], 'elaboration', num_samples=3)
        print(f"  Elaboration 平均分數: {elaboration_result['score']:.3f} (個別分數: {elaboration_result['individual_scores']})")
        
        # 添加metadata
        originality_result.update({
            'question': item['question'],
            'model': item['model'],
            'response_index': i
        })
        elaboration_result.update({
            'question': item['question'],
            'model': item['model'],
            'response_index': i
        })
        
        all_results.extend([originality_result, elaboration_result])
        
        # 儲存快取
        model.save_cache()
    
    # 計算平均分數
    averages = calculate_averages(all_results)
    
    # 輸出結果
    print("\n" + "="*50)
    print("評估結果摘要")
    print("="*50)
    print(f"總共評估回應數: {averages['total_responses']}")
    print(f"每個回應評估次數: 3次")
    print(f"總評估次數: {averages['total_evaluations']}")
    print(f"Originality 回應平均分數: {[f'{score:.3f}' for score in averages['originality_scores']]}")
    print(f"Elaboration 回應平均分數: {[f'{score:.3f}' for score in averages['elaboration_scores']]}")
    print(f"Originality 所有個別分數: {averages['originality_individual_scores']}")
    print(f"Elaboration 所有個別分數: {averages['elaboration_individual_scores']}")
    
    # 計算變異數統計
    import statistics
    if len(averages['originality_individual_scores']) > 1:
        orig_std = statistics.stdev(averages['originality_individual_scores'])
        print(f"平均 Originality 分數: {averages['average_originality']:.3f}")
        print(f"Originality 標準差: {orig_std:.3f}")
    
    if len(averages['elaboration_individual_scores']) > 1:
        elab_std = statistics.stdev(averages['elaboration_individual_scores'])
        print(f"平均 Elaboration 分數: {averages['average_elaboration']:.3f}")
        print(f"Elaboration 標準差: {elab_std:.3f}")
    print("="*50)
    
    # 儲存詳細結果
    output_file = input_file.replace('.json', '_simple_eval_results.json')
    output_data = {
        'summary': averages,
        'detailed_results': all_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    print(f"\n詳細結果已儲存至: {output_file}")
    
    return averages, all_results

if __name__ == "__main__":
    # 測試用檔案
    # input_file = 'test_3_samples.json'
    input_file = '../Results/AUT/Output/qwen_agent/AUT_qwen_1_1_Qwen_Qwen_qwen_0912-1641_100_chat_log.json'

    if os.path.exists(input_file):
        simple_eval(input_file)
    else:
        print(f"測試檔案 {input_file} 不存在")
        # 可以改用其他檔案
        
        # simple_eval(input_file)