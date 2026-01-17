import json
import os
import logging
import subprocess
import requests
import time
from datetime import timedelta
from openai import OpenAI
from gemmaModel import ModelRunner
# import google.generativeai as genai
# from google import genai
# from google.genai import types
from dotenv import load_dotenv
load_dotenv()

def generate_response_llama2_torchrun(
    message,
    ckpt_dir: str = "/tmp2/llama-2-7b-chat",
    tokenizer_path: str = "/home/chenlawrance/repo/LLM-Creativity/model/tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_batch_size: int = 4):
    message_json = json.dumps(message)  # Serialize the message to a JSON string
    command = [
        "torchrun", "--nproc_per_node=1", "/home/chenlawrance/repo/LLM-Creativity/llama_model/llama_chat_completion.py",
        "--ckpt_dir", ckpt_dir,
        "--tokenizer_path", tokenizer_path,
        "--max_seq_len", str(max_seq_len),
        "--max_batch_size", str(max_batch_size),
        "--temperature", str(temperature),
        "--top_p", str(top_p),
        "--message", message_json
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout.strip()

        # Find the beginning of the generated response
        assistant_prefix = "> Assistant:"
        start_idx = output.find(assistant_prefix)
        if start_idx != -1:
            # Calculate the starting index of the actual response
            start_of_response = start_idx + len(assistant_prefix)
            # Extract and return the generated response part
            generated_response = output[start_of_response:].strip()
            return generated_response
        else:
            return "No response generated or unable to extract response."
    except subprocess.CalledProcessError as e:
        print(f"Error executing torchrun command: {e.stderr}")
        return "Unable to generate response due to an error."

class Agent:
    def generate_answer(self, answer_context):
        raise NotImplementedError("This method should be implemented by subclasses.")
    def construct_assistant_message(self, prompt):
        raise NotImplementedError("This method should be implemented by subclasses.")
    def construct_user_message(self, prompt):
        raise NotImplementedError("This method should be implemented by subclasses.")

class OpenAIAgent(Agent):
    def __init__(self, model_name, agent_name, agent_role, agent_speciality, agent_role_prompt, speaking_rate, missing_history = []):
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.agent_speciality = agent_speciality
        self.agent_role_prompt = agent_role_prompt
        self.speaking_rate = speaking_rate
        self.missing_history = missing_history
        
    def generate_answer(self, answer_context, temperature=1):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=answer_context,
                n=1)
            result = completion.choices[0].message.content
            # for pure text -> return completion.choices[0].message.content
            return result
        except Exception as e:
            print(f"Error with model {self.model_name}: {e}")
            time.sleep(10)
            return self.generate_answer(answer_context)

    def construct_assistant_message(self, content):
        return {"role": "assistant", "content": content}
    
    def construct_user_message(self, content):
        return {"role": "user", "content": content}

TOTAL_TOKEN_USED = {"Input": 0, "Thinking": 0, "Output": 0}
class GeminiAgent(Agent):
    def __init__(self, model_name, agent_name, agent_role, agent_speciality, agent_role_prompt, speaking_rate):
        self.model_name = model_name
        self.client = genai.Client()
        self.model_name = model_name
        # self.model = genai.GenerativeModel(self.model_name)
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.agent_speciality = agent_speciality
        self.agent_role_prompt = agent_role_prompt
        self.speaking_rate = speaking_rate
        print(f"Using 🔑: {os.getenv('GOOGLE_API_KEY')}")

    def token_calculate(self, response, time_delta):
        if not response:
            return 
        input_token = response.usage_metadata.prompt_token_count
        output_token = response.usage_metadata.candidates_token_count
        thinking_token = response.usage_metadata.thoughts_token_count
        TOTAL_TOKEN_USED["Input"] += input_token
        TOTAL_TOKEN_USED["Output"] += output_token
        TOTAL_TOKEN_USED["Thinking"] += thinking_token
        current_input = TOTAL_TOKEN_USED["Input"]
        current_out = TOTAL_TOKEN_USED["Output"]+TOTAL_TOKEN_USED["Thinking"]
        current_fee = TOTAL_TOKEN_USED["Input"]*(1.25/1000000) + (TOTAL_TOKEN_USED["Output"]+TOTAL_TOKEN_USED["Thinking"])*(10/1000000)

        record = f"\n---\nTime Usage: {time_delta}\n"
        record += f"Input token: {input_token}, Thinking token: {thinking_token}, Output token: {output_token}\n"
        record += f"    Current Input: {current_input}, Current Ouput: {current_out}, Current Fee: {current_fee}"

        with open("token_record.txt", "a", encoding="utf-8") as f:
            f.write(record)

    def generate_answer(self, answer_context,temperature= 1.0):
        try: 
            # print(f"\n\n{answer_context}\n\n")
            start_time = time.time()
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=answer_context[0]['content'],
                config=types.GenerateContentConfig(
                    # max_output_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    # thinking_config=types.ThinkingConfig(thinking_budget=128), # Disables thinking
                ),
            )
            # for pure text -> return response.text
            # return response.candidates[0].content
            end_time = time.time()
            total_time = end_time - start_time
            time_delta = timedelta(seconds=total_time)
            self.token_calculate(response, time_delta)
            print("Sleep Time 😴...")
            time.sleep(10)
            return response.text
        except Exception as e:
            logging.exception("Exception occurred during response generation: " + str(e))
            time.sleep(30)
            return self.generate_answer(answer_context)
        
    def construct_assistant_message(self, content):
        response = {"role": "assistant", "content": [content]}
        return response
    
    def construct_user_message(self, content):
        response = {"role": "user", "content": [content]}
        return response
        
class Llama2Agent(Agent):
    def __init__(self, ckpt_dir, tokenizer_path, agent_name):
        self.ckpt_dir = ckpt_dir
        self.tokenizer_path = tokenizer_path
        self.agent_name = agent_name

    def generate_answer(self, answer_context, temperature=0.6, top_p=0.9, max_seq_len=100000, max_batch_size=4): # return pure text
        return generate_response_llama2_torchrun(
            message=answer_context,
            ckpt_dir=self.ckpt_dir,
            tokenizer_path=self.tokenizer_path,
            temperature=temperature,
            top_p=top_p,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size
        )
    
    def construct_assistant_message(self, content):
        return {"role": "assistant", "content": content}
    
    def construct_user_message(self, content):
        return {"role": "user", "content": content}
    

class QwenAgent(Agent):
    def __init__(self, model_name, agent_name, agent_role, agent_speciality, agent_role_prompt, speaking_rate, missing_history = []):
        self.model_name = model_name
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.agent_speciality = agent_speciality
        self.agent_role_prompt = agent_role_prompt
        self.speaking_rate = speaking_rate
        self.missing_history = missing_history
        
    def generate_answer(self, answer_context, api_idx=0):
        print(f"     ---> Calling {self.model_name}")
        try:
            ## NVIDIA API
            # print(f"\nAnswer Context:\n{answer_context}")
            # client = OpenAI(
            #     base_url = "https://integrate.api.nvidia.com/v1",
            #     api_key = os.getenv(f"NVIDIA_API_KEY_{api_idx}")
            # )
            # completion = client.chat.completions.create(
            #     model="qwen/"+self.model_name,
            #     messages=answer_context,
            #     n=1
            # )
            # result = completion.choices[0].message.content
            # print(f"\nResult:\n{result}")
            # return result
            # print(f"\n\nanswer_context: {answer_context}\n\n")
            # print(f'answer_context[0][content]: {answer_context[0]["content"]}\n\n')

            ## Local Model
            url = "http://127.0.0.1:5003/chat"
            # payload = {"message": answer_context[0]['content']}
            payload = {"message": answer_context}
            response = requests.post(url, json=payload)
            result = response.json()["response"]
            return result

        except Exception as e:
            print(f"Error with model {self.model_name}: {e}\nsleep 60s, api idx: {api_idx}")
            time.sleep(60)
            api_idx = (api_idx+1)%4
            return self.generate_answer(answer_context, api_idx)

    def construct_assistant_message(self, content):
        return {"role": "assistant", "content": content}
    
    def construct_user_message(self, content):
        return {"role": "user", "content": content}
    

class Llama3Agent(Agent):
    def __init__(self, model_name, agent_name, agent_role, agent_speciality, agent_role_prompt, speaking_rate, missing_history = []):
        self.model_name = model_name
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.agent_speciality = agent_speciality
        self.agent_role_prompt = agent_role_prompt
        self.speaking_rate = speaking_rate
        self.missing_history = missing_history
        
    def generate_answer(self, answer_context, api_idx=0):
        print(f"     ---> Calling {self.model_name}")
        try:
            # client = OpenAI(
            #     base_url = "https://integrate.api.nvidia.com/v1",
            #     api_key = os.getenv(f"NVIDIA_API_KEY_{api_idx}")
            # )
            # completion = client.chat.completions.create(
            #     model="meta/"+self.model_name,
            #     messages=answer_context,
            #     n=1
            # )
            # result = completion.choices[0].message.content
            # return result
            
            ## Local Model
            url = "http://127.0.0.1:5001/chat"
            # payload = {"message": answer_context[0]['content']}
            payload = {"message": answer_context}
            response = requests.post(url, json=payload)
            result = response.json()["response"]
            return result

        except Exception as e:
            print(f"Error with model {self.model_name}: {e}\nsleep 60s, api idx: {api_idx}")
            time.sleep(60)
            api_idx = (api_idx+1)%4
            return self.generate_answer(answer_context, api_idx)

    def construct_assistant_message(self, content):
        return {"role": "assistant", "content": content}
    
    def construct_user_message(self, content):
        return {"role": "user", "content": content}

    
class GemmaAgent(Agent):
    def __init__(self, model_name, agent_name, agent_role, agent_speciality, agent_role_prompt, speaking_rate, missing_history = []):
        self.model_name = model_name
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.agent_speciality = agent_speciality
        self.agent_role_prompt = agent_role_prompt
        self.speaking_rate = speaking_rate
        self.missing_history = missing_history
        self.gemma = ModelRunner(model_name="gemma")
        
    def generate_answer(self, answer_context):
        print(f"     ---> Calling {self.model_name}")
        # print(f"\n\n📣 answer_context:\n{answer_context}")
        # contents = []
        # for msg in answer_context:
        #     formatted_msg = {"role": msg["role"], "content":[{"type":"text", "text": msg["content"]}]}
        #     contents.append(formatted_msg)
        msg_str = ""
        for m in answer_context:
            msg_str += f"{m['role']}: {m['content']}\n\n"

        try:
            ## Local Model
            # url = "http://127.0.0.1:5000/chat"
            # payload = {"user_input": msg_str}
            # response = requests.post(url, json=payload)
            # print(f"\n\n📣 Input:\n{msg_str}")
            # result = response.json()["response"]
            # print(f"\n\n🤖 Result:\n{result}")
            # return result
            response = self.gemma.generate_response(msg_str)
            return response

        except Exception as e:
            print(f"Error with model {self.model_name}: {e}")
            return 
            # return self.generate_answer(answer_context)

    def construct_assistant_message(self, content):
        return {"role": "assistant", "content": content}
    
    def construct_user_message(self, content):
        return {"role": "user", "content": content}