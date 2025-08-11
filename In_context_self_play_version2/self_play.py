import copy
import json
import os
import re
from datetime import datetime

from discriminator import discriminator
from generator import generate_response
from prompts.discriminator_prompt import self_play_discriminator_1, self_play_discriminator_2
from prompts.generator_prompt import multi_generator_1_en
from utils.log import colored_print
from utils.config_loader import load_config


def write_json_file(log_data, log_dir):
    """將對話記錄寫入 JSON 檔案，並自動處理版本號。"""
    os.makedirs(log_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    version = 0
    
    while True:
        log_filename = os.path.join(log_dir, f"{date_str}_v{version}.json")
        if not os.path.exists(log_filename):
            break
        version += 1

    with open(log_filename, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=4)

    print(f"\nConversation log saved to {log_filename}")


def extract_conversation_info(messages):
    """
    從 CoSER 格式的 messages 列表中提取結構化資訊，專為 Discriminator 使用。
    """
    if not messages or not isinstance(messages, list):
        raise ValueError("Input 'messages' must be a non-empty list of dictionaries.")

    system_message = messages[0].get('content', '') if messages[0].get('role') == 'system' else ''
    
    last_user_message = ""
    last_user_index = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get('role') == 'user':
            last_user_message = messages[i].get('content', '')
            last_user_index = i
            break
            
    if last_user_index == -1:
        raise ValueError("No 'user' role message found to respond to.")

    conversation_list = messages[1:last_user_index]
    
    return system_message, conversation_list, last_user_message


def format_conversation_history_for_prompt(conversation_list):
    """將對話歷史列表格式化為易於閱讀的字串。"""
    if not conversation_list:
        return "No conversation history yet."
    
    history_str = ""
    for msg in conversation_list:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        history_str += f"{role}: {content}\n"
    return history_str.strip()


def discuss(messages):
    """
    基於 GAN 概念的對話生成與精煉迴圈 (重構版)。
    Generator 直接使用 messages 列表，Discriminator 的建議會被附加進去。
    """
    # 載入設定
    config = load_config()
    generator_model = config.get('generator_model')
    discriminator_model = config.get('discriminator_model')
    round_count = config.get('discussion_round')
    log_dir = config.get('log_dir', './experiments_log/self_play')

    colored_print('red', f'--- Starting Refactored GAN-based Discussion ---')
    colored_print('red', f'Generator: {generator_model} | Discriminator: {discriminator_model} | Rounds: {round_count}')

    # 提取結構化資訊，這部分僅供 Discriminator 使用
    system_message, conversation_list, last_user_message = extract_conversation_info(messages)
    formatted_history = format_conversation_history_for_prompt(conversation_list)
    
    # 這是傳遞給 Generator 的對話歷史，它會隨著迭代而增長
    messages_for_generator = copy.deepcopy(messages)
    
    discussion_log = []
    final_generator_output = ""

    for i in range(round_count):
        colored_print('blue', f"\n{'='*20} Round {i+1} {'='*20}")
        
        # 1. 生成器 (演員) 產生回應
        # Generator 直接使用完整的 messages 列表
        colored_print("blue", f"Generator Input (Messages Count: {len(messages_for_generator)}):{messages_for_generator}")
        # for msg in messages_for_generator:
        #     colored_print("gray", f"  - Role: {msg['role']}, Content: {msg['content'][:150]}...")

        generator_output = generate_response(generator_model, messages_for_generator)
        colored_print("cyan", f"Generator Output (Round {i+1}):\n{generator_output}")
        final_generator_output = generator_output # 永遠保留最新的生成結果

        # 記錄本輪的生成部分
        round_log = {
            "round": i + 1,
            "generator_input_messages": messages_for_generator,
            "generator_output": generator_output,
        }

        # 如果是最後一輪，則不再需要判別器提供建議
        if i == round_count - 1:
            round_log["discriminator_prompt"] = "N/A (Final Round)"
            round_log["discriminator_advice"] = "N/A (Final Round)"
            discussion_log.append(round_log)
            colored_print('red', "--- Final Round Concluded ---")
            break

        # 2. 建構判別器提示
        # Discriminator 使用結構化的、清晰的上下文來進行評判
        discriminator_prompt = self_play_discriminator_2.format(
            system_message=system_message,
            conversation_list=formatted_history, # 使用格式化後的歷史
            last_user_message=last_user_message,
            generator_output=generator_output
        )
        
        # 3. 判別器 (評論家) 提供回饋
        discriminator_advice = discriminator(discriminator_prompt, discriminator_model)
        colored_print("magenta", f"Discriminator Advice (for next round):\n{discriminator_advice}")

        # 4. **核心改動**：將判別器的建議作為一個新的 user message 附加到對話歷史中
        # 這模擬了在對話中直接給予指導的過程
        feedback_message = {
            "role": "user",
            "content": f"""--- CRITIC'S FEEDBACK ---\n{discriminator_advice}\n--- END FEEDBACK ---\nPlease regenerate your previous response based on this feedback."""
        }
        # 首先，將剛才的生成結果加入歷史，以維持對話的連貫性
        messages_for_generator.append({"role": "assistant", "content": generator_output})
        # 然後，加入評論家的指導作為新的 user 指令
        messages_for_generator.append(feedback_message)

        # 完整記錄本輪的所有資訊
        round_log["discriminator_prompt"] = discriminator_prompt
        round_log["discriminator_advice"] = discriminator_advice
        discussion_log.append(round_log)

    # 儲存完整的對話記錄
    write_json_file(discussion_log, log_dir)
    
    return final_generator_output, discussion_log
