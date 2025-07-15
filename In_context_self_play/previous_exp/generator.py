# Remember to source bashrc before running this script
import json
import os
import sys

from calculate_token import num_tokens_from_string
from dotenv import load_dotenv
from openai import OpenAI
from openai_token import openai_token
# from prompts.all_prompt import all_chat, chat_log_4, chat_log_7, chat_log_8
from prompts.all_prompt_grouping import all_chat, chat_log_8
from prompts.discriminator_advice import advice_1, advice_2, advice_3, advice_4
from prompts.prompts import (v1_2, v1_3, v1_4_1, v1_4_2, v1_5, v1_6, v1_7, v1_8, v2_1, v2_2_1,
                             v2_2_2, v2_3_1, v2_3_2, v3_4, v3_5)
from prompts.user_prompts import user_v1, user_v2, user_v3

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
llm_client = OpenAI(api_key=api_key)


# load the json file
def load_json_file(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        print(f"Loaded {len(raw_data)} lines from {input_file}")
        print(raw_data)
        print(type(raw_data[0]['input']))
        return raw_data


# json file name would be the yy/mm/dd, and version0,1,2 ... etc 9if the file exists, then version+1
from datetime import datetime


def write_json_file(conversation_history):
    date_str = datetime.now().strftime("%Y-%m-%d")
    base_filename = f"./experiments_log/original/{date_str}_v0.json"

    version = 0
    while os.path.exists(base_filename):
        version += 1
        base_filename = f"./experiments_log/original/{date_str}_v{version}.json"

    with open(base_filename, "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, ensure_ascii=False, indent=4)

    print(f"Conversation saved to {base_filename}")


def generator(raw_data, prompt, MODEL="gpt-4o"):
    prompt_token_usage = num_tokens_from_string(prompt, "cl100k_base")
    print(f"Prompt token usage: {prompt_token_usage}")
    print('===============================')

    conversation_history = []
    for data in raw_data:
        user_input = data['input']
        if MODEL != 'o1-mini' and MODEL!= 'o1-preview':
            messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user_input + user_v3}]
            response = llm_client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.5,
            )
        else:
            print(f'using {MODEL}')
            messages = [{"role": "user", "content": prompt + '\n' + user_input + user_v3}]
            response = llm_client.chat.completions.create(
                model=MODEL,
                messages=messages,
            )


        agent_response = response.choices[0].message.content
        print(f"Input: {user_input}")
        print(f"Agent: {agent_response}")
        print()

        conversation_history.append({"Input": user_input, "Agent": agent_response})

    write_json_file(conversation_history)

    return conversation_history


INPUT_FILE = "./input_data/processed_data_50.json"
MODEL = 'o1-mini'
# MODEL = 'gpt-4o'

if __name__ == "__main__":
    raw_data = load_json_file(INPUT_FILE)
    advice = advice_2 + advice_3 + advice_4
    prompt = v3_5.format(chat_log_8=chat_log_8, discriminator_advice=advice)
    # prompt = v3_4.format(all_chat=all_chat, discriminator_advice=advice)
    generator(raw_data, prompt, MODEL)
