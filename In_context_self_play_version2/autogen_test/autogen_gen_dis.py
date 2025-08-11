import os

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from autogen.cache import Cache
from autogen.coding import DockerCommandLineCodeExecutor, LocalCommandLineCodeExecutor

from dotenv import load_dotenv

from prompts.all_prompt import all_chat, chat_log_4, chat_log_7, chat_log_8
from prompts.discriminator_prompt import discriminator_1, discriminator_2
from user_agent_chat import v2_1_u1, v3_4, v3_5, v3_6

from prompts.discriminator_advice import advice_1, advice_2, advice_3, advice_4
from prompts.prompts import (v1_2, v1_3, v1_4_1, v1_4_2, v1_5, v1_6, v1_7, v1_8, v2_1, v2_2_1, v2_2_2, v2_3_1, v2_3_2, v3_5_autogen)
from prompts.user_prompts import user_v1, user_v2, user_v3

load_dotenv()
config_list = [{"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}]
# You can also use the following method to load the config list from a file or environment variable.
# config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

# os.makedirs("coding", exist_ok=True)
# Use DockerCommandLineCodeExecutor if docker is available to run the generated code.
# Using docker is safer than running the generated code directly.
# code_executor = DockerCommandLineCodeExecutor(work_dir="coding")
# code_executor = LocalCommandLineCodeExecutor(work_dir="coding")

user_proxy = UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    # code_execution_config={"executor": code_executor},
)

generator = AssistantAgent(
    name="generator",
    system_message=v3_5_autogen.format(chat_log_8=chat_log_8) + "請依照 discriminator 的建議 對所有的 Input 再做一次回答。",
    llm_config={
        "config_list": config_list,
        "cache_seed": None
    },
)

discriminator = AssistantAgent(
    name="discriminator",
    system_message=discriminator_2.format(chat_log=chat_log_8, user_agent_chat=v3_6),
    llm_config={
        "config_list": config_list,
        "cache_seed": None
    },
)


def reflection_message(recipient, messages, sender, config):
    print("Reflecting...")
    # return f"Reflect and provide critique on the following writing. \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}"
    return f"請提供關於這些對話紀錄中，Other 的回覆像 <<LLM>> 的地方在哪裡，以及像 <<白>> 的地方又在哪。\n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}"


nested_chat_queue = [
    {
        "recipient": discriminator,
        "message": reflection_message,
        "max_turns": 1,
    },
]
user_proxy.register_nested_chats(
    nested_chat_queue,
    trigger=generator,
    # position=4,
)

# Use Cache.disk to cache the generated responses.
# This is useful when the same request to the LLM is made multiple times.
with Cache.disk(cache_seed=42) as cache:
    user_proxy.initiate_chat(
        generator,
        message=v3_5_autogen.format(chat_log_8=chat_log_8),
        max_turns=4,
        cache=cache,
    )
