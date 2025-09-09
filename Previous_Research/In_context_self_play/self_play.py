import json
import os
from datetime import datetime

from discriminator import discriminator
from generator import generate_response
from prompts.discriminator_prompt import self_play_discriminator_1, self_play_discriminator_2
from utils.log import colored_print


def write_json_file(conversation_history):
    date_str = datetime.now().strftime("%Y-%m-%d")
    base_filename = f"./experiments_log/self_play/{date_str}_v0.json"
    
    if not os.path.exists("./experiments_log/self_play"):
        os.makedirs("./experiments_log/self_play")

    version = 0
    while os.path.exists(base_filename):
        version += 1
        base_filename = f"./experiments_log/self_play/{date_str}_v{version}.json"

    with open(base_filename, "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, ensure_ascii=False, indent=4)

    print(f"Conversation saved to {base_filename}")



def extract_conversation(messages):
    system_message = None
    conversation_list = []
    last_user_message = None

    last_user_index = None
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            last_user_index = i

    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")
        
        if role == "system":
            system_message = content
        
        elif role in ("user", "assistant"):
            # pass the last user message
            if role == "user" and i == last_user_index:
                last_user_message = content
                continue
            conversation_list.append(msg)
    
    return system_message, conversation_list, last_user_message

def discuss(messages):
    discriminator_advice = ''
    discussion_log = []
    user_agent_chat = ''
    generator_model = 'gpt-4o-mini'
    discriminator_model = 'gpt-4o-mini'
    round = 2

    # Extract the messages into system_message, conversation_list, and last_user_message
    # system messages: role play background
    # conversation_list: list of user and assistant messages
    # last_user_message: last user message that needs to be responded to

    system_message, conversation_list, last_user_message = extract_conversation(messages)

    discussion_log.append({"generator_model": generator_model, "discriminator_model": discriminator_model})
    for i in range(round):
        messages.insert(1, {"role": "system", "content": discriminator_advice})
        colored_print("blue", f"Generator prompt: {messages}")

        # need to get the discriminator_advices here
        generator_output = generate_response(generator_model, messages)

        discriminator_prompt = self_play_discriminator_2.format(system_message=system_message, conversation_list=conversation_list, last_user_message=last_user_message, generator_output=generator_output)
        colored_print("yellow", f"Discriminator prompt: {discriminator_prompt}")
        discriminator_advice = discriminator(discriminator_prompt, discriminator_model)

        discussion_log.append({"round": i, "messages": messages, "generator": generator_output, "discriminator_advice": discriminator_advice})
        colored_print("green", f"Round {i}: {discussion_log}")

        if i == round - 1:
            colored_print('red', f"Final round")
            messages.insert(1, {"role": "system", "content": discriminator_advice})
            generator_output = generate_response(generator_model, messages)

    # save the conversation history
    # write_json_file(discussion_log)
    return generator_output, discussion_log
