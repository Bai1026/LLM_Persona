import os
import time

import openai
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
# from openai import OpenAI
from utils.openai_token import openai_token

load_dotenv(find_dotenv())
# llm_client = OpenAI()
# api_key = os.getenv('OPENAI_API_KEY')
# llm_client = OpenAI(api_key=api_key)

API_MAX_RETRY = 5
API_RETRY_SLEEP = 5
API_ERROR_OUTPUT = "API ERROR"


def chat_completion_deepseek(model, messages, temperature, max_tokens, top_p=0.7, stream=False):
    client = OpenAI(
        base_url='https://integrate.api.nvidia.com/v1',
        api_key="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",
    )

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            completion = client.chat.completions.create(
                model=model,
                # messages=messages,
                messages=[{
                    "role": "user",
                    "content": "hiii"
                }],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=stream)
            if isinstance(completion, str):
                output = completion  # 如果已經是字串，直接使用
            else:
                # 正常情況下從物件中取得回應內容
                output = completion.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(messages)
            print(type(e), e)
        except TypeError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except KeyError:
            print(type(e), e)
            break

    return output


def discriminator(prompt, model='gpt-4o-mini'):
    if type(prompt) == str and 'gemini' not in model:
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt

    if 'gemini' in model:
        from google import genai
        client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        # print(messages)
        # print(type(messages))

        response = client.models.generate_content(model=model, contents=[messages])
        # print(response.text)
        return response.text

    elif model != 'o1-mini' and model != 'o1-preview':
        llm_client = openai.OpenAI()
        response = llm_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
        )

    elif model == 'deepseek-r1':
        response = chat_completion_deepseek(model='deepseek-ai/deepseek-r1',
                                            messages=messages,
                                            temperature=0.6,
                                            max_tokens=4096,
                                            top_p=0.7,
                                            stream=False)

    else:
        response = llm_client.chat.completions.create(
            model=model,
            messages=messages,
        )

    agent_response = response.choices[0].message.content
    print(f"Discriminator token usage: {openai_token(response)}")

    return agent_response
