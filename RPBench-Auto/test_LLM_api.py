import google.generativeai as genai
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

import openai
# genai.configure()
# model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
# response = model.generate_content("台中最好吃的牛排店?")
# print(response.text)
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    # api_key="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",
    api_key="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",
)
try:
    completion = client.chat.completions.create(
    model="deepseek-ai/deepseek-r1",
    messages=[{"role":"user","content":"hiii"}],
    temperature=0.6,
    top_p=0.7,
    max_tokens=4096,
    stream=False
    )
    output = completion.choices[0].message.content
    print(output)

    
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

print(output)