import argparse
import json
import os
import random
from string import Template

import jsonlines
from log import colored_print
from tqdm.auto import tqdm
from utils import chat_completion, chat_completion_for_judger, extract_and_parse_json, make_config

# ==================== define the maximum number of messages per character ====================
MAX_MESSAGES_PER_CHAR = 5
RPBENCH_PATH = "data/rpbench_character.jsonl"

# RPBENCH_PATH = "data/test_1.jsonl"
# RPBENCH_PATH = "data/test_2.jsonl"
# RPBENCH_PATH = "data/test_6.jsonl"
# RPBENCH_PATH = "data/test_40.jsonl"

TEMPLATE = Template("""$background

# NPC Profile:
## Name
$name_text

## Title
$title

## Description
$description

## Definition
$definition_text

## Long Definition
$long_definition_text
""")

JUDGER_TEMPLATE = Template("""# NPC Profile:
## Name
$name_text

## Title
$title

## Description
$description

## Definition
$definition_text

## Long Definition
$long_definition_text

You are a judge for an AI NPC system. You need to simulate a user and interact with 2 AI NPC. For each round (except the first round), you should pick a better response from the 2 AI NPC and come up with your reply. It will be in a JSON format: {"winner": "model_a" or "model_b", "next_round_user_speaks": "YOUR RESPONSE AS THE SIMULATED USER", "decision_reason": "REASON FOR PICK THE WINNER"}. For the first round, use "winner": null
""")


def chat_completion_judger(model, messages):
    while True:
        response = chat_completion_for_judger(model, messages)
        try:
            parsed_response = extract_and_parse_json(response)
            if ("winner" in parsed_response and "next_round_user_speaks" in parsed_response):
                return response
        except:
            pass


from datetime import datetime


def create_unique_folder():
    date_str = datetime.now().strftime("%Y-%m-%d")
    base_root = "../In_context_self_play/experiments_log/multi_agent"

    version = 0
    folder_name = f"{date_str}_v{version}"
    folder_path = os.path.join(base_root, folder_name)
    while os.path.exists(folder_path):
        version += 1
        folder_name = f"{date_str}_v{version}"
        folder_path = os.path.join(base_root, folder_name)

    os.makedirs(folder_path)
    return folder_path


def write_json_file(conversation_history, folder_path):
    base_filename = f"{folder_path}/v0.json"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    version = 0
    while os.path.exists(base_filename):
        version += 1
        base_filename = f"{folder_path}/v{version}.json"

    with open(base_filename, "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, ensure_ascii=False, indent=4)

    print(f"Conversation saved to {base_filename}")


def eval_models_pairwise(model_1, model_2, eval_rounds, rounds=1):
    model_1_win_count = 0
    model_2_win_count = 0
    eval_data = []
    win_lose_pairs = []
    eval_results = []

    prompt_file = f'./round_prompts/round_{eval_rounds}_prompts.json'
    print(f"使用提示詞檔案: {prompt_file}")

    character_prompts = {}
    if prompt_file:
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)
                # 建立角色編號到提示詞的對應表
                for item in prompt_data:
                    character_prompts[item['character']] = item['generator_prompt']
            print(f"已載入 {len(character_prompts)} 個角色的提示詞，從 {prompt_file}")
        except Exception as e:
            print(f"載入提示詞檔案時發生錯誤: {e}")
            return
        
    with jsonlines.open(RPBENCH_PATH) as reader:
        for obj in reader:
            eval_data.append(obj)
    print(f"已載入 {len(eval_data)} 個範例，從 {RPBENCH_PATH}")

    folder_path = create_unique_folder()

    judger_config = make_config("config/judger_config.yaml")
    assert len(judger_config) == 1, "判斷模型設定檔中應僅包含一個模型"
    judger_model_name = list(judger_config.keys())[0]
    judger_model = judger_config[judger_model_name]
    print(f"判斷模型: `{judger_model_name}`")

    candidate_config = make_config("config/api_config.yaml")
    assert model_1 in candidate_config, f"{model_1} 在候選模型設定檔中找不到"
    assert model_2 in candidate_config, f"{model_2} 在候選模型設定檔中找不到"
    print(f"正在比較 `{model_1}` 和 `{model_2}`")

    # 對每個角色重複執行指定輪數
    for round_num in range(rounds):
        print(f"\n開始第 {round_num+1}/{rounds} 輪評估")

        # for d in (pbar := tqdm(eval_data)):
        for idx, d in enumerate((pbar := tqdm(eval_data))):
            npc_profile = d["npc_profile"]
            conversation = d["conversation"]
            background = d["background"]
            greeting = "\n".join(conversation[0]["sentences"])

            # 取得角色對應的提示詞 (如果有)
            character_id = str(idx)  # 角色編號 (從 0 開始計數)
            character_prompt = character_prompts.get(character_id)

            candidate_messages = [
                {
                    "role": "system",
                    "content": TEMPLATE.substitute(background=background, **npc_profile),
                },
                {
                    "role": "assistant",
                    "content": greeting
                },
            ]

            judger_messages = [
                {
                    "role": "system",
                    "content": JUDGER_TEMPLATE.substitute(npc_profile)
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "model_a": greeting,
                        "model_b": greeting
                    }),
                },
            ]

            judger_response = chat_completion_judger(judger_model, judger_messages)
            parsed_judger_response = extract_and_parse_json(judger_response)
            judger_messages.append({"role": "assistant", "content": judger_response})

            for _ in range(MAX_MESSAGES_PER_CHAR):
                # 隨機分配 model_a 和 model_b 給 model_1 和 model_2
                model_a = model_1 if bool(random.getrandbits(1)) else model_2
                model_b = model_2 if model_a == model_1 else model_1
                assignment = {"model_a": model_a, "model_b": model_b}

                print()
                print('=' * 50)
                colored_print("blue", f"分配: {assignment}")

                user_input = parsed_judger_response["next_round_user_speaks"]
                candidate_messages.append({"role": "user", "content": user_input})

                # candidate_messages = model 的 system prompt
                discussion_log = []
                if model_a == 'gan_model':
                    model_a_response, discussion_log = chat_completion(candidate_config[model_a], candidate_messages, character_prompt)
                    model_b_response = chat_completion(candidate_config[model_b], candidate_messages)

                elif model_b == 'gan_model':
                    model_a_response = chat_completion(candidate_config[model_a], candidate_messages)
                    model_b_response, discussion_log = chat_completion(candidate_config[model_b], candidate_messages, character_prompt)

                else:
                    model_a_response = chat_completion(candidate_config[model_a], candidate_messages)
                    model_b_response = chat_completion(candidate_config[model_b], candidate_messages)

                judger_message_content = json.dumps({"model_a": model_a_response, "model_b": model_b_response})
                judger_messages.append({"role": "user", "content": judger_message_content})
                judger_response = chat_completion_judger(judger_model, judger_messages)
                parsed_judger_response = extract_and_parse_json(judger_response)
                colored_print('red', f"判斷回應: {parsed_judger_response}")

                eval_result = {
                    "round": round_num + 1,
                    "candidate_messages": candidate_messages,
                    "assignment": assignment,
                    "judger_messages": judger_messages,
                    "judger_response": judger_response,
                }

                eval_results.append(eval_result)
                winner = parsed_judger_response["winner"]
                if winner:
                    winner_model = None
                    if winner == "model_a":
                        winner_model = model_a
                        win_lose_pairs.append((model_a, model_b))
                    elif winner == "model_b":
                        winner_model = model_b
                        win_lose_pairs.append((model_b, model_a))
                    if winner_model == model_1:
                        model_1_win_count += 1
                    elif winner_model == model_2:
                        model_2_win_count += 1

                total_matches = model_1_win_count + model_2_win_count
                win_rate = model_1_win_count / total_matches if total_matches > 0 else 0
                pbar.set_postfix({
                    "model_1_win_rate": win_rate,
                    "model_1_win_count": model_1_win_count,
                    "model_2_win_count": model_2_win_count,
                    "round": f"{round_num+1}/{rounds}",
                })

                judger_messages.append({"role": "assistant", "content": judger_response})
                candidate_messages.append({
                    "role": "assistant",
                    "content": model_a_response if winner == "model_a" else model_b_response,
                })

                if model_a == 'gan_model' or model_b == 'gan_model':
                    discussion_log.append({
                        "round": round_num + 1,
                        "model_1": model_1,
                        "model_2": model_2,
                        "model_a": model_a,
                        "model_b": model_b,
                        "candidate_messages": candidate_messages,
                        "judger_messages": judger_messages,
                        "model_a_response": model_a_response,
                        "model_b_response": model_b_response,
                        "winner": winner,
                        "judger_response": judger_response,
                        "model_1_win_rate": win_rate,
                        "model_1_win_count": model_1_win_count,
                        "model_2_win_count": model_2_win_count,
                        "eval_result": eval_result,
                    })
                    # num of the files = number of rounds * number of characters * MAX_MESSAGES_PER_CHAR
                    write_json_file(discussion_log, folder_path)

    print(f"\n評估完成！總共進行 {rounds} 輪評估")
    print(f"模型 {model_1} 勝場: {model_1_win_count}")
    print(f"模型 {model_2} 勝場: {model_2_win_count}")

    if not os.path.exists("results/character"):
        os.makedirs("results/character")
    with jsonlines.open(f"results/character/eval_{model_1}_vs_{model_2}_rounds{rounds}.jsonl", "w") as writer:
        writer.write_all(eval_results)

    return win_lose_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m1", "--model_1", type=str, required=True)
    # parser.add_argument("-m2", "--model_2", type=str, default="gpt-4o-mini")
    parser.add_argument("-m2", "--model_2", type=str, default="gpt-4o")
    parser.add_argument("-r", "--rounds", type=int, default=1, help="每個角色執行的評估輪數")
    parser.add_argument("-er", "--eval_rounds", type=int, default=1, help="用哪一輪的訓練 prompt")

    args = parser.parse_args()
    eval_models_pairwise(args.model_1, args.model_2, args.eval_rounds, args.rounds)
