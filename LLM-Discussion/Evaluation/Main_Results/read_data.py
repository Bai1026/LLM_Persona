from rich import print
import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

mother_path = './Ins/'
finding_list = [
    'llama_vector',
    'llama_single',
    'llama_multi',
    'qwen_vector',
    'qwen_single',
    'qwen_multi',
]

for finding in finding_list:
    file_path = mother_path + finding + '_simple_eval_results.json'
    data = load_json(file_path)

    print('='*100)
    # print(file_path)
    print(finding)
    print(data.keys())

    print(f"Ori: {data['summary']['average_originality']}")
    print(data['detailed_results'][-2:-1])
    print(f"Ela: {data['summary']['average_elaboration']}")
    print(data['detailed_results'][-1:])
