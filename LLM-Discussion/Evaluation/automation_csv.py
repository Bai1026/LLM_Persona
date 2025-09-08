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

    Task = parts[0] # AUT, Scientific, Similarities, Instances
    Type = parts[2] # debate, conversational
    Data_Num = parts[-1].split('-')[0] if '-' in parts[-1] else parts[-1]
    
    # Find timestamp part
    timestamp_part = None
    for part in parts:
        if '-' in part and len(part) > 5:  # Look for timestamp format
            timestamp_part = part
            break
    
    if timestamp_part:
        Raw_Timestamp = timestamp_part.split('-')
    else:
        Raw_Timestamp = parts[-2].split('-') if len(parts) > 2 else ["", ""]
    
    print("Raw_Timestamp: ", Raw_Timestamp)
    
    # 修正時間戳記解析邏輯
    if len(Raw_Timestamp) == 2:
        # 格式：20250903-144738
        date_part = Raw_Timestamp[0]  # 20250903
        time_part = Raw_Timestamp[1]  # 144738
        
        # 解析日期：20250903 -> 2025-09-03
        year = date_part[:4]
        month = date_part[4:6]
        day = date_part[6:8]
        date = f'{year}-{month}-{day}'
        
        # 解析時間：144738 -> 14:47:38
        hour = time_part[:2]
        minute = time_part[2:4]
        second = time_part[4:6]
        time = f'{hour}:{minute}:{second}'
        
        Timestamp = f'{date} {time}'
    else:
        # 舊格式相容性
        date = '-'.join(Raw_Timestamp[:3])
        time = ':'.join(Raw_Timestamp[3:6]) if len(Raw_Timestamp) >= 6 else ""
        Timestamp = f'{date} {time}'
    
    print("date: ", date)
    print("time: ", time)
    print("Timestamp: ", Timestamp)

    Mode, Agent, Rounds, Model_Name, Role_Name = None, None, None, None, None  # Initialize to None

    if parts[1] == "single":
        Agent = parts[4]
        Rounds = parts[5]
        Model_Name = parts[6]
        Mode = parts[3]
        Role_Name = parts[7]
    elif parts[1] == 'multi':
        Mode = parts[3]
        Agent = parts[4]
        Rounds = parts[5]
        Model_Name = parts[6]
        Role_Name = parts[7]
    elif parts[1] == "vanilla":
        # Handle vanilla Qwen format: AUT_vanilla_qwen_1_1_Qwen25_VanillaQwen_vanilla_20250903-144738_10
        Type = "vanilla"
        Mode = "vanilla"
        Agent = "VanillaQwen"
        Rounds = parts[3]  # "1"
        Model_Name = parts[5]  # "Qwen25"
        Role_Name = parts[6]  # "VanillaQwen"
    elif parts[1] == "persona":
        # Handle persona API format: AUT_persona_api_1_1_PersonaAPI_PersonaAPI_persona_api_20250903-144738_10
        # OR shorter format: AUT_persona_api_0907-1546_10
        Type = "api"
        Mode = "persona"
        Agent = "PersonaAPI"
        
        if len(parts) >= 7:
            # Long format: AUT_persona_api_1_1_PersonaAPI_PersonaAPI_persona_api_20250903-144738_10
            Rounds = parts[3]  # "1"
            Model_Name = parts[5]  # "PersonaAPI"
            Role_Name = parts[6]  # "PersonaAPI"
        else:
            # Short format: AUT_persona_api_0907-1546_10
            Rounds = "1"  # Default value
            Model_Name = "PersonaAPI"  # Default value
            Role_Name = "PersonaAPI"  # Default value
    elif parts[1] == "openai":
        # Handle OpenAI baseline format: AUT_openai_baseline_1_1_gpt_4_OpenAI_baseline_20250903-144738_10
        Type = "baseline"
        Mode = "baseline"
        Agent = "OpenAI"
        Rounds = parts[3]  # "1"
        Model_Name = parts[5]  # "gpt_4"
        Role_Name = parts[6]  # "OpenAI"
    else:
        print(f'ERROR AGENT!! Unknown format: {parts[1]}')
        print(f'Full filename: {input_file_name}')
        print(f'Parts: {parts}')
    

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
