import json
import os
from datetime import datetime
import dspy

from discriminator import discriminator
from generator import generate_response
from prompts.discriminator_prompt import multi_discriminator_1, multi_discriminator_1_en
from prompts.generator_prompt import multi_generator_1, multi_generator_1_en
from prompts.evaluator_prompt_dspy import EvaluatorPromptChinese, EvaluatorPrompt, EvaluatorPrompt_2
from rich import print
from utils.config_loader import get_model_settings, load_config
from utils.log import colored_print
from utils.memory import update_memory
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

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

    config = load_config()
    generator_model = config['generator_model']
    discriminator_model = config['discriminator_model']
    evaluator_model = config['evaluator_model']

    if evaluator_model == 'gpt-4o-mini':
        dspy_model = "openai/gpt-4o-mini"
    if evaluator_model == 'gpt-4o-2024-11-20':
        dspy_model = "openai/gpt-4o-2024-11-20"
    elif evaluator_model == 'gemini-2.0-flash':
        dspy_model = "gemini/gemini-2.0-flash"
    else:
        raise ValueError(f"Unsupported evaluator model: {evaluator_model}")

    dspy_cache = True

    dspy.configure(lm=dspy.LM(model=dspy_model, temperature=0.3, cache=dspy_cache))
    evaluator = dspy.ChainOfThought(EvaluatorPrompt_2)

    round = config['discussion_round']
    method = config['gan_method']
    colored_print('red', f'Using {method}')
    colored_print('red', f'original messages: {messages}')

    system_message, scene_summary, last_user_message = extract_conversation(messages)
    print(f"System message: {system_message}")
    print(f"Conversation list: {scene_summary}")
    print(f"Last user message: {last_user_message}")
    print()

    fixed_prompt = f"""
    <input>
    **Persona Background** (Background of the person u need to imitate): {system_message}  
    **Latest User Message** (The query you need to reply with proper persona): {last_user_message}
    """
    # Not using the dialogue summary here
    # **Dialogue Summary** (The dialogue of the user and assistant, and you are assistant): {scene_summary}  

    new_generator_prompt = multi_generator_1_en
    new_discriminator_prompt = multi_discriminator_1_en
    discussion_log.append({
        "generator_model": generator_model,
        "discriminator_model": discriminator_model,})
    
    # discussion_log.append({ 
    #     "generator_model": generator_model, 
    #     "discriminator_model": discriminator_model,
    #     "evaluator_output": {
    #         "new_generator_prompt": new_generator_prompt,
    #         "new_discriminator_prompt": new_discriminator_prompt,}
    #     })
    
    for i in range(round):
        if i == round - 1:
            colored_print('red', f"Final round")
            tmp_gen = f"""
            **Discriminator Feedback**: {discriminator_advice}
            **Your Previous Response**: {generator_output}
            </input>
            """

            generator_prompt = new_generator_prompt + '\n' + fixed_prompt + tmp_gen
            colored_print("blue", f"Generator prompt: {generator_prompt}")
            generator_output = generate_response(generator_model, generator_prompt)

            discussion_log.append({
                "round": f'final_round: {i}',
                "generator_response": generator_output,
                "evaluator_output": {
                    "new_generator_prompt": new_generator_prompt,
                    "new_discriminator_prompt": new_discriminator_prompt,
                }
            })
            return generator_output, discussion_log
        
        # ============================= Generator Part ============================= 
        tmp_gen = f"""
        **Discriminator Feedback** (Discriminator's advice about the response you gave last round): {discriminator_advice}
        **Your Previous Response** (Your previous response about the same condition): {generator_output}
        </input>
        """

        generator_prompt = new_generator_prompt + '\n' + fixed_prompt + tmp_gen
        colored_print("blue", f"Generator prompt: {generator_prompt}")
        generator_output = generate_response(generator_model, generator_prompt)

        # ============================= Discriminator Part ============================= 
        tmp_dis = f"""
        **Generator Response** (Generator's response about the same condition that need your advice): {generator_output}
        **Your Previous Advice** (Your previous advice about the same condition): {discriminator_advice}
        </input>
        """
        discriminator_prompt = new_discriminator_prompt + '\n' + fixed_prompt + tmp_dis        
        colored_print("yellow", f"Discriminator prompt: {discriminator_prompt}")
        discriminator_advice = discriminator(discriminator_prompt, discriminator_model)

        # ============================= Evaluator Part =============================
        evaluator_output = evaluator(system_message=system_message,
                        #    scene_summary=scene_summary,
                        #    last_user_message=last_user_message,
                           original_generator_prompt=new_generator_prompt,
                           original_discriminator_prompt=new_discriminator_prompt,
                           original_generator_response=generator_output,
                           original_discriminator_advice=discriminator_advice)

        # update both prompt here and pass them through
        new_generator_prompt = evaluator_output.new_generator_prompt
        new_discriminator_prompt = evaluator_output.new_discriminator_prompt

        discussion_log.append({
            "round": i,
            "generator_response": generator_output,
            "discriminator_advice": discriminator_advice,
            "evaluator_output": {
                "new_generator_prompt": new_generator_prompt,
                "new_discriminator_prompt": new_discriminator_prompt,
            }
        })

    return generator_output, discussion_log
