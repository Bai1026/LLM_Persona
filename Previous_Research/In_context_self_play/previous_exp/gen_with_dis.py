import json
import os
from datetime import datetime

from discriminator import discriminator
from generator import generator, load_json_file
# from prompts.all_prompt import all_chat, chat_log_4, chat_log_7, chat_log_8
from prompts.all_prompt_grouping import chat_log_8
from prompts.discriminator_prompt import (discriminator_1, discriminator_2, discriminator_3,
                                          discriminator_4, discriminator_5)
from prompts.prompts import v3_5_autogen
from prompts.user_prompts import user_v1, user_v2, user_v3


def write_json_file(conversation_history):
    date_str = datetime.now().strftime("%Y-%m-%d")
    base_filename = f"./experiments_log/gen_with_dis/{date_str}_v0.json"

    version = 0
    while os.path.exists(base_filename):
        version += 1
        base_filename = f"./experiments_log/gen_with_dis/{date_str}_v{version}.json"

    with open(base_filename, "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, ensure_ascii=False, indent=4)

    print(f"Conversation saved to {base_filename}")


# for discriminator not to see the reason
def parse_data(data):
    parsed_data = []
    for entry in data:
        input_text = entry['Input']
        agent_text = entry['Agent']

        if '[reason:' in agent_text:
            agent_text = agent_text.split('[reason:')[0].strip()

        parsed_data.append({'Input': input_text, 'Other': agent_text})
    return parsed_data


def discuss(data, round):
    advice = ''
    discussion_log = []
    user_agent_chat = ''
    generator_model = 'gpt-4o-mini'
    discriminator_model = 'gpt-4o-mini'

    discussion_log.append({"generator_model": generator_model, "discriminator_model": discriminator_model})
    for _ in range(round):
        generator_prompt = v3_5_autogen.format(chat_log_8=chat_log_8, discriminator_advice=advice, generator_response=user_agent_chat)
        print(f"Generator prompt: {generator_prompt}")
        user_agent_chat = generator(data, generator_prompt, generator_model)
        parsed_user_agent_chat = parse_data(user_agent_chat)

        discriminator_prompt = discriminator_5.format(chat_log=chat_log_8, user_agent_chat=parsed_user_agent_chat)
        print(f"Discriminator prompt: {discriminator_prompt}")
        advice = discriminator(user_agent_chat, discriminator_prompt, discriminator_model)

        discussion_log.append({"user_agent_chat": user_agent_chat, "advice": advice})
    return discussion_log



import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and discuss')
    parser.add_argument("-r", "--round", type=int, default=5, help='Number of rounds to discuss')
    args = parser.parse_args()
    round = args.round

    INPUT_FILE = "./input_data/processed_data_50.json"
    raw_data = load_json_file(INPUT_FILE)

    discussion_log = discuss(raw_data, round)

    write_json_file(discussion_log)

    print(discussion_log)
