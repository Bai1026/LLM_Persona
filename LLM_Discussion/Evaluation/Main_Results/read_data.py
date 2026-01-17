from rich import print
import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# mother_path = './AUT/'
finding_list = [
    # "./AUT/AUT_google_topline_1_1_gemini_2_5_pro_MultiRole_0920-1527_100_chat_log",
    # "./AUT/AUT_google_topline_1_1_gemini_2_5_pro_SingleRole_0920-0015_100_chat_log",
    # "./Instances/Instances_google_topline_1_1_gemini_2_5_pro_MultiRole_0922-0244_100_chat_log",
    # "./Instances/Instances_google_topline_1_1_gemini_2_5_pro_SingleRole_0922-0138_100_chat_log",
    "./AUT/AUT_multi_debate_roleplay_4_5_llama-3-1-8b-instruct_Environmentalist-CreativeProfessional-Futurist-Futurist_chat_log_2025-09-13-11-59-42_100",
    "./Instances/Instances_multi_debate_roleplay_4_5_llama-3-1-8b-instruct_Environmentalist-CreativeProfessional-Futurist-Futurist_chat_log_2025-09-16-21-20-04_100",
    "./Scientific/Scientific_multi_debate_roleplay_4_5_llama-3-1-8b-instruct_Environmentalist-CreativeProfessional-Futurist-Futurist_chat_log_2025-09-14-18-15-16_100"
    "./Similarities/Similarities_multi_debate_roleplay_4_5_llama-3-1-8b-instruct_Environmentalist-CreativeProfessional-Futurist-Futurist_chat_log_2025-09-17-04-53-34_100"
]

for finding in finding_list:
    # file_path = mother_path + finding + '_simple_eval_results.json'
    file_path = finding + '_simple_eval_results.json'
    data = load_json(file_path)

    print('='*100)
    # print(file_path)
    print(finding)
    print(data.keys())

    print(f"Ori: {data['summary']['average_originality']}")
    # print(data['detailed_results'][-2:-1])
    print(f"Ela: {data['summary']['average_elaboration']}")
    # print(data['detailed_results'][-1:])
