# discriminator.py (Recommended Refactor)
import os
import time
import openai
from dotenv import find_dotenv, load_dotenv

# --- Client Initialization (Do this once) ---
load_dotenv(find_dotenv())

openai_client = openai.OpenAI()
gemini_client = None # Lazy load
nvidia_client = openai.OpenAI(
    base_url='https://integrate.api.nvidia.com/v1',
    api_key=os.getenv("NVIDIA_API_KEY"), # Move key to .env
)

API_MAX_RETRY = 3
API_RETRY_SLEEP = 5

def _call_nvidia_api(model, messages, temperature, max_tokens):
    """Helper function for NVIDIA API calls with retry logic."""
    for _ in range(API_MAX_RETRY):
        try:
            completion = nvidia_client.chat.completions.create(
                model=model,
                messages=messages, # Fixed: Use the actual messages
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content
        except openai.RateLimitError as e:
            print(f"NVIDIA API Rate Limit Error: {e}. Retrying in {API_RETRY_SLEEP}s...")
            time.sleep(API_RETRY_SLEEP)
        except Exception as e:
            print(f"An unexpected error occurred with NVIDIA API: {e}")
            return "API_ERROR"
    return "API_ERROR_MAX_RETRIES"


def discriminator(prompt, model='gpt-4o-mini'):
    """
    Acts as a critic, providing feedback on a generated response.

    Args:
        prompt (str or list): The prompt for the discriminator.
        model (str): The model to use for discrimination.

    Returns:
        str: The discriminator's feedback.
    """
    messages = prompt
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]

    if 'gemini' in model:
        global gemini_client
        if gemini_client is None:
            from google import genai
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            gemini_client = genai.GenerativeModel(model)
        
        response = gemini_client.generate_content(messages)
        return response.text

    elif 'deepseek' in model:
        # Assuming deepseek is hosted on NVIDIA's API endpoint
        return _call_nvidia_api(
            model='deepseek-ai/deepseek-r1',
            messages=messages,
            temperature=0.6,
            max_tokens=4096
        )
    
    else: # Default to OpenAI compatible APIs
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
        )
        return response.choices[0].message.content