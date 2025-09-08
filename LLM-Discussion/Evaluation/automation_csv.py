from pathlib import Path
import numpy as np
import csv


def calculate_mean_std(total_results):
    # Extracting scores for each criterion from the total results
    fluency_scores = [item["fluency"][-1]["average_fluency"] for item in total_results]
    flexibility_scores = [item["flexibility"][-1]["average_flexibility"] for item in total_results]
    originality_scores = [item["originality"][-1]["average_originality"] for item in total_results]
    elaboration_scores = [item["elaboration"][-1]["average_elaboration"] for item in total_results]

    # Calculating mean and standard deviation for each criterion
    results = {
        "mean_fluency": round(np.mean(fluency_scores), 3),
        "std_fluency": round(np.std(fluency_scores), 3),
        "mean_flexibility": round(np.mean(flexibility_scores), 3),
        "std_flexibility": round(np.std(flexibility_scores), 3),
        "mean_originality": round(np.mean(originality_scores), 3),
        "std_originality": round(np.std(originality_scores), 3),
        "mean_elaboration": round(np.mean(elaboration_scores), 3),
        "std_elaboration": round(np.std(elaboration_scores), 3),
    }
    return results

def write_results_to_csv(input_file_name, mean_std_results, csv_file_path, version):
    
    headers = ['Timestamp', 'Task', 'Type', 'Mode', 'Agent', 'Round','Model Name', 'Role Name', 'Data Num', 'Mean Fluency', 'STD Fluency', 'Mean Flexibility', 'STD Flexibility', 'Mean Originality', 'STD Originality', 'Mean Elaboration', 'STD Elaboration', 'File Name']
    csv_data = []
    parts = input_file_name.split('_')
    
    print(f"🔍 解析檔案: {input_file_name}")
    print(f"📝 分割結果: {parts}")
    
    # Check if this is the new evaluation format: evaluation_AUT_persona_api_0908-0548_10_sampling_4.json
    if parts[0] == "evaluation":
        Task = parts[1]  # AUT
        Type = parts[2]  # persona
        Mode = parts[3]  # api
        Data_Num = parts[5]  # 10
        timestamp_str = parts[4]  # 0908-0548
    else:
        # Original format
        Task = parts[0] # AUT, Scientific, Similarities, Instances
        Type = parts[2] # debate, conversational
        Data_Num = parts[-1].split('-')[0]
        # 找到時間戳記部分 (通常是倒數第二個帶有 - 的部分)
        for i in range(len(parts)-1, -1, -1):
            if '-' in parts[i] and parts[i].replace('-', '').isdigit():
                timestamp_str = parts[i]
                break
        else:
            timestamp_str = "0000-0000"  # 預設值
    
    print(f"⏰ 時間戳記字串: {timestamp_str}")
    
    # 統一時間戳記格式為 MMDD-HHMM
    if len(timestamp_str) == 9 and timestamp_str.count('-') == 1:  # 0908-0548 格式
        Timestamp = timestamp_str
    elif len(timestamp_str) > 10:  # 20250903-144738 格式
        timestamp_parts = timestamp_str.split('-')
        if len(timestamp_parts) == 2:
            date_part = timestamp_parts[0][-4:]  # 取最後4位數當作 MMDD
            time_part = timestamp_parts[1][:4]   # 取前4位數當作 HHMM
            Timestamp = f"{date_part}-{time_part}"
        else:
            Timestamp = timestamp_str
    else:
        Timestamp = timestamp_str
    
    print(f"📅 最終時間戳記: {Timestamp}")

    Mode, Agent, Rounds, Model_Name, Role_Name = None, None, None, None, None  # Initialize to None

    # Handle the new evaluation format first
    if parts[0] == "evaluation":
        # evaluation_AUT_persona_api_0908-0645_10_sampling_4
        if len(parts) >= 4 and parts[2] == "persona" and parts[3] == "api":
            Type = "api"
            Mode = "persona"
            Agent = "PersonaAPI"
            Rounds = "1"  # Default for persona API
            Model_Name = "PersonaAPI"
            Role_Name = "PersonaAPI"
        else:
            # Generic evaluation format handling
            Type = parts[2] if len(parts) > 2 else "unknown"
            Mode = parts[3] if len(parts) > 3 else "unknown"
            Agent = "Evaluation"
            Rounds = "1"
            Model_Name = "Unknown"
            Role_Name = "Unknown"
    elif parts[1] == "single":
        Agent = parts[4] if len(parts) > 4 else "Unknown"
        Rounds = parts[5] if len(parts) > 5 else "1"
        Model_Name = parts[6] if len(parts) > 6 else "Unknown"
        Mode = parts[3] if len(parts) > 3 else "single"
        Role_Name = parts[7] if len(parts) > 7 else "Unknown"
    elif parts[1] == 'multi':
        Mode = parts[3] if len(parts) > 3 else "multi"
        Agent = parts[4] if len(parts) > 4 else "Unknown"
        Rounds = parts[5] if len(parts) > 5 else "1"
        Model_Name = parts[6] if len(parts) > 6 else "Unknown"
        Role_Name = parts[7] if len(parts) > 7 else "Unknown"
    elif parts[1] == "vanilla":
        # Handle vanilla Qwen format: AUT_vanilla_qwen_1_1_Qwen25_VanillaQwen_vanilla_20250903-144738_10
        Type = "vanilla"
        Mode = "vanilla"
        Agent = "VanillaQwen"
        Rounds = parts[3] if len(parts) > 3 else "1"  # "1"
        Model_Name = parts[5] if len(parts) > 5 else "Qwen25"  # "Qwen25"
        Role_Name = parts[6] if len(parts) > 6 else "VanillaQwen"  # "VanillaQwen"
    elif parts[1] == "persona":
        # Handle persona API format: AUT_persona_api_0908-0548_10
        Type = "api"
        Mode = "persona"
        Agent = "PersonaAPI"
        Rounds = "1"  # PersonaAPI 總是單輪
        Model_Name = "PersonaAPI"
        Role_Name = "PersonaAPI"
    elif parts[1] == "openai":
        # Handle OpenAI baseline format: AUT_openai_baseline_1_1_gpt_4_OpenAI_baseline_20250903-144738_10
        Type = "baseline"
        Mode = "baseline"
        Agent = "OpenAI"
        Rounds = parts[3] if len(parts) > 3 else "1"  # "1"
        Model_Name = parts[5] if len(parts) > 5 else "gpt_4"  # "gpt_4"
        Role_Name = parts[6] if len(parts) > 6 else "OpenAI"  # "OpenAI"
    else:
        print(f'❌ 未知格式: {parts[1]}')
        print(f'📄 完整檔名: {input_file_name}')
        print(f'🔧 分割結果: {parts}')
        # 設定預設值
        Type = "unknown"
        Mode = "unknown"
        Agent = "Unknown"
        Rounds = "1"
        Model_Name = "Unknown"
        Role_Name = "Unknown"
    
    print(f"📊 解析結果 - Type: {Type}, Mode: {Mode}, Agent: {Agent}, Rounds: {Rounds}")

    row = [Timestamp, Task, Type, Mode, Agent, Rounds, Model_Name, Role_Name, Data_Num]
    row.extend([
        mean_std_results['mean_fluency'], mean_std_results['std_fluency'],
        mean_std_results['mean_flexibility'], mean_std_results['std_flexibility'],
        mean_std_results['mean_originality'], mean_std_results['std_originality'],
        mean_std_results['mean_elaboration'], mean_std_results['std_elaboration'],
        input_file_name 
    ])
    csv_data.append(row)
    file_path = Path(csv_file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with file_path.open(mode='a+', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if file.tell() == 0:  # If file is empty, write headers
                writer.writerow(headers)
            writer.writerows(csv_data)
        
        # Now sort the data if needed, by reading, sorting, and rewriting the CSV file
        with file_path.open(mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip header
            sorted_data = sorted(reader, key=lambda x: (x[0], x[8]))  # Sort by Timestamp and Data Num

        with file_path.open(mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)  # Write headers
            writer.writerows(sorted_data)

        print(f'Data sorted by Timestamp and Data and saved to {csv_file_path}')
    except Exception as e:
        print(f'ERROR: Failed to write data to CSV due to {e}')

    print(f'Data sorted by Timestamp and Data and saved to {csv_file_path}')

# Example usage
if __name__ == "__main__":
    intput_file_name = "AUT_persona_api_0908-0548_10_sampling_4.json"
