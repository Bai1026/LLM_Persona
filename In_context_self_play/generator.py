from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

import os


def generate_response(model, prompt):
    if type(prompt) == str and 'gemini' not in model:
        messages = [{"role": "user", "content": prompt}]
    elif type(prompt) == str and 'gemini' in model:
        messages = prompt
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

    else:
        from openai import OpenAI
        llm_client = OpenAI()
        response = llm_client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content


# from dotenv import find_dotenv, load_dotenv

# load_dotenv(find_dotenv())

# import os

# from openai import OpenAI

# llm_client = OpenAI()

# def generate_response(model, prompt):
#     if type(prompt) == str:
#         messages = [{"role": "user", "content": prompt}]
#     else:
#         messages = prompt

#     if 'gemini' in model:
#         from google import genai
#         client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
#         response = client.models.generate_content(model=model, contents=messages)
#         return response.text
#     else:
#         response = llm_client.chat.completions.create(
#             model=model,
#             messages=messages,
#         )
#         return response.choices[0].message.content
