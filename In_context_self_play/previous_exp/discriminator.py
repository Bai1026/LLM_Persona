import os

from calculate_token import num_tokens_from_string
from dotenv import load_dotenv
from openai import OpenAI
from prompts.all_prompt import all_chat, chat_log_4, chat_log_7, chat_log_8
from prompts.discriminator_prompt import discriminator_1, discriminator_2
from user_agent_chat import v2_1_u1, v3_4, v3_5, v3_6

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
llm_client = OpenAI(api_key=api_key)


def discriminator(user_agent_chat, prompt, model='gpt-4o'):
    print('===============================')
    print("Discriminator:")
    print(user_agent_chat)

    prompt_token_usage = num_tokens_from_string(prompt, "cl100k_base")
    print(f"Prompt token usage: {prompt_token_usage}")
    print('===============================')

    messages = [{"role": "user", "content": prompt}]

    if model != 'o1-mini' and model != 'o1-preview':
        response = llm_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
        )
    else:
        response = llm_client.chat.completions.create(
            model=model,
            messages=messages,
        )

    agent_response = response.choices[0].message.content
    print(f"Agent: {agent_response}")
    print()

    return agent_response


if __name__ == '__main__':
    prompt = discriminator_2.format(chat_log=chat_log_8, user_agent_chat=v3_6)
    discriminator(prompt)
