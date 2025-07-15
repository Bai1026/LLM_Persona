import os

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from autogen.cache import Cache
from autogen.coding import DockerCommandLineCodeExecutor, LocalCommandLineCodeExecutor
from dotenv import load_dotenv

load_dotenv()
config_list = [
    {
        "model": "gpt-4o-mini",
        "api_key": os.environ["OPENAI_API_KEY"]
    },
    {
        "model": "gpt-40-mini",
        "api_key": os.environ["OPENAI_API_KEY"]
    },
]
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

writing_assistant = AssistantAgent(
    name="writing_assistant",
    system_message="你是一個作家 請寫一篇關於最近人工智慧的更新的博客文章。這篇博客文章應該引人入勝，並且對一般觀眾易懂。",
    llm_config={
        "config_list": config_list,
        "cache_seed": None
    },
)

reflection_assistant = AssistantAgent(
    name="reflection_assistant",
    system_message="你是一個評論者，請提供關於這篇文章的評論。包括長度、深度、風格等方面的建議。",
    llm_config={
        "config_list": config_list,
        "cache_seed": None
    },
)


def reflection_message(recipient, messages, sender, config):
    print("Reflecting...")
    # return f"Reflect and provide critique on the following writing. \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}"
    return f"請提供關於這篇文章的評論。包括長度、深度、風格等方面的建議。\n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}"


nested_chat_queue = [
    {
        "recipient": reflection_assistant,
        "message": reflection_message,
        "max_turns": 1,
    },
]
user_proxy.register_nested_chats(
    nested_chat_queue,
    trigger=writing_assistant,
    # position=4,
)

# Use Cache.disk to cache the generated responses.
# This is useful when the same request to the LLM is made multiple times.
with Cache.disk(cache_seed=42) as cache:
    user_proxy.initiate_chat(
        writing_assistant,
        message="請寫一篇關於最近人工智慧的更新的博客文章。這篇博客文章應該引人入勝，並且對一般觀眾易懂。"
        "這篇博客文章應該引人入勝，並且對一般觀眾易懂。"
        "必須有超過3段但不超過1000字。",
        max_turns=2,
        cache=cache,
    )
