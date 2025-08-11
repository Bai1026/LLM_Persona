from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

import os


# generator.py (Recommended Refactor)
import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

def generate_response(model, messages):
    """
    Generates a response from a specified LLM.

    Args:
        model (str): The model identifier (e.g., 'gpt-4o-mini', 'gemini-1.5-pro-latest').
        messages (list or str): The prompt, either as a string or a list of message dicts.

    Returns:
        str: The generated text response.
    """
    # Standardize input to the required format for each API
    if isinstance(messages, str):
        # Gemini's genai.Client().models.generate_content expects a string
        # while OpenAI expects a list of dicts.
        if 'gemini' not in model:
            messages = [{"role": "user", "content": messages}]
    
    if 'gemini' in model:
        from google import genai
        # Ensure genai is configured, preferably once at the start of your app
        if not genai.API_KEY:
             genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        
        # The new client takes the model in generate_content
        client = genai.GenerativeModel(model) 
        response = client.generate_content(messages)
        return response.text

    else:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
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
