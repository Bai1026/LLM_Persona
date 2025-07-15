import json
import os
from datetime import datetime

from discriminator import discriminator
from generator import generate_response
from prompts.discriminator_prompt import (gan_discriminator_1, gan_discriminator_1_en, gan_discriminator_2, gan_discriminator_2_en,
                                          gan_discriminator_3, gan_discriminator_4, gan_discriminator_4_en, gan_discriminator_5,
                                          gan_discriminator_5_en)
from prompts.generator_prompt import gan_generator_1, gan_generator_1_en, gan_generator_2, gan_generator_2_en, gan_generator_3_en
from rich import print
from utils.config_loader import get_model_settings, load_config_2
from utils.log import colored_print
from utils.memory import update_memory


def write_json_file(conversation_history):
    date_str = datetime.now().strftime("%Y-%m-%d")
    base_filename = f"./experiments_log/gan/{date_str}_v0.json"

    if not os.path.exists("./experiments_log/gan"):
        os.makedirs("./experiments_log/gan")

    version = 0
    while os.path.exists(base_filename):
        version += 1
        base_filename = f"./experiments_log/gan/{date_str}_v{version}.json"

    with open(base_filename, "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, ensure_ascii=False, indent=4)

    print(f"Conversation saved to {base_filename}")


def extract_conversation(messages):
    system_message = None
    scene_summary = []
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
            scene_summary.append(msg)

    return system_message, scene_summary, last_user_message


def discuss(messages):
    """
    Extract the messages into system_message, scene_summary, and last_user_message
    system messages: role play background
    scene_summary: list of user and assistant messages
    last_user_message: last user message that needs to be responded to
    """

    generator_output = ''
    discriminator_advice = ''
    discriminator_advices = []
    last_memory = {"traits": [], "tone": [], "inner_conflict": [], "interaction_dynamics": [], "improvement_suggestions": []}

    discussion_log = []
    user_agent_chat = ''

    config = load_config_2()
    generator_model = config['generator_model']
    discriminator_model = config['discriminator_model']
    round = config['discussion_round']
    method = config['gan_method']
    colored_print('red', f'Using {method}')
    colored_print('red', f'original messages: {messages}')

    system_message, scene_summary, last_user_message = extract_conversation(messages)
    print(f"System message: {system_message}")
    print(f"Conversation list: {scene_summary}")
    print(f"Last user message: {last_user_message}")
    print()

    discussion_log.append({"generator_model": generator_model, "discriminator_model": discriminator_model})
    for i in range(round):
        new_messages = [messages[0]]
        for msg in messages:
            if msg["role"] != "system":
                new_messages.append(msg)

        messages = new_messages
        print(f"Messages: {messages}")
        if discriminator_advice:
            messages.insert(1, {"role": "system", "content": discriminator_advice})

        # TODO: need to get the discriminator_advices here
        # gan_generator_prompt = gan_generator_2_en.format(system_message=system_message,
        #                                                  scene_summary=scene_summary,
        #                                                  last_user_message=last_user_message,
        #                                                  discriminator_advice=discriminator_advice)
        gan_generator_prompt = gan_generator_3_en.format(
            system_message=system_message,
            scene_summary=scene_summary,
            last_user_message=last_user_message,
            discriminator_advice=discriminator_advice,
            previous_generator_response=generator_output,
        )
        # # gan_generator_prompt = messages

        # colored_print("red", f"Generator prompt: {gan_generator_prompt}")
        colored_print("blue", f"Generator prompt: {gan_generator_prompt}")
        generator_output = generate_response(generator_model, gan_generator_prompt)

        if method == 'independent':
            # discriminator_prompt = gan_discriminator_1_en.format(system_message=system_message, scene_summary=scene_summary, last_user_message=last_user_message, generator_output=generator_output, discriminator_advice=discriminator_advice)
            # discriminator_prompt = gan_discriminator_2_en.format(system_message=system_message, scene_summary=scene_summary, last_user_message=last_user_message, generator_output=generator_output, discriminator_advice=discriminator_advice)
            discriminator_prompt = gan_discriminator_5_en.format(system_message=system_message,
                                                                 scene_summary=scene_summary,
                                                                 last_user_message=last_user_message,
                                                                 generator_output=generator_output,
                                                                 discriminator_advice=discriminator_advice)
            # colored_print("yellow", f"Discriminator prompt: {discriminator_prompt}")
            discriminator_advice = discriminator(discriminator_prompt, discriminator_model)

        elif method == 'advice_list':
            discriminator_prompt = gan_discriminator_5_en.format(system_message=system_message,
                                                                 scene_summary=scene_summary,
                                                                 last_user_message=last_user_message,
                                                                 generator_output=generator_output,
                                                                 discriminator_advice=discriminator_advices)
            # colored_print("yellow", f"Discriminator prompt: {discriminator_prompt}")
            discriminator_advice = discriminator(discriminator_prompt, discriminator_model)
            discriminator_advices.append(discriminator_advice)

        elif method == 'summary':
            discriminator_prompt = gan_discriminator_5_en.format(system_message=system_message,
                                                                 scene_summary=scene_summary,
                                                                 last_user_message=last_user_message,
                                                                 generator_output=generator_output,
                                                                 discriminator_advice=discriminator_advice)
            # colored_print("yellow", f"Discriminator prompt: {discriminator_prompt}")
            discriminator_advice = discriminator(discriminator_prompt, discriminator_model)

            # This is the method of update the last memory by summary for every advice
            discriminator_advice = update_memory(last_memory, discriminator_advice)
            last_memory = discriminator_advice
            discriminator_advice = "\n".join(
                [f"{key.replace('_', ' ').capitalize()}: {', '.join(value)}" for key, value in discriminator_advice.items()])

        else:
            colored_print('red', 'METHOD IS NOT IN THE METHOD_LIST!!!!!!')
            break

        discussion_log.append({
            "round": i,
            "generator_prompt": gan_generator_prompt,
            "generator": generator_output,
            "discriminator_advice": discriminator_advice
        })
        # colored_print("green", f"Round {i}: {discussion_log}")

        if i == round - 1:
            colored_print('red', f"Final round")
            # messages.insert(1, {"role": "system", "content": discriminator_advice})
            # generator_output = generate_response(generator_model, messages)
            # gan_generator_prompt = gan_generator_1_en.format(system_message=system_message, scene_summary=scene_summary, last_user_message=last_user_message, discriminator_advice=discriminator_advice)
            gan_generator_prompt = gan_generator_3_en.format(
                system_message=system_message,
                scene_summary=scene_summary,
                last_user_message=last_user_message,
                discriminator_advice=discriminator_advice,
                previous_generator_response=generator_output,
            )
            # gan_generator_prompt = messages
            generator_output = generate_response(generator_model, gan_generator_prompt)
            discussion_log.append({
                "round": f'final_round: {i}',
                "generator": generator_output,
            })

    return generator_output, discussion_log
