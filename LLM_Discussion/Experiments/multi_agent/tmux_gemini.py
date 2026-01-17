import subprocess
import time
from datetime import timedelta

start_time = time.time()

## ✅
# command = f"python auto_eval_persona.py -d /workspace/LLM-discussion/LLM-discussion-reproduce/Datasets/Instances/instances_100.json -t Instances -p 1 -v 4 --topline --gemini_model gemini-2.5-pro --no_eval"
# print(f"\n🌟 Command: {command}\n")
# process = subprocess.Popen(command, shell=True)
# process.wait()

## ✅
# command = f"python auto_eval_persona.py -d /workspace/LLM-discussion/LLM-discussion-reproduce/Datasets/Instances/instances_100.json -t Instances -p 1 -v 4 --topline --gemini_model gemini-2.5-pro --no_eval -a"
# print(f"\n🌟 Command: {command}\n")
# process = subprocess.Popen(command, shell=True)
# process.wait()


## ✅
# command = f"python auto_eval_persona.py -d /workspace/LLM-discussion/LLM-discussion-reproduce/Datasets/Scientific/scientific_100.json -t Scientific -p 1 -v 4 --topline --gemini_model gemini-2.5-pro --no_eval"
# print(f"\n🌟 Command: {command}\n")
# process = subprocess.Popen(command, shell=True)
# process.wait()
# time.sleep(300)

## ✅
# command = f"python auto_eval_persona.py -d /workspace/LLM-discussion/LLM-discussion-reproduce/Datasets/Scientific/scientific_100.json -t Scientific -p 1 -v 4 --topline --gemini_model gemini-2.5-pro --no_eval -a"
# print(f"\n🌟 Command: {command}\n")
# process = subprocess.Popen(command, shell=True)
# process.wait()
# time.sleep(300)

## ✅
# command = f"python auto_eval_persona.py -d /workspace/LLM-discussion/LLM-discussion-reproduce/Datasets/Similarities/similarities_100.json -t Similarities -p 1 -v 4 --topline --gemini_model gemini-2.5-pro --no_eval"
# print(f"\n🌟 Command: {command}\n")
# process = subprocess.Popen(command, shell=True)
# process.wait()
# time.sleep(300)

## ✅
# command = f"python auto_eval_persona.py -d /workspace/LLM-discussion/LLM-discussion-reproduce/Datasets/Similarities/similarities_100.json -t Similarities -p 1 -v 4 --topline --gemini_model gemini-2.5-pro --no_eval -a"
# print(f"\n🌟 Command: {command}\n")
# process = subprocess.Popen(command, shell=True)
# process.wait()
# time.sleep(300)



command = f"python llm_discussion.py -c config_role_gemini.json -d /workspace/LLM-discussion/LLM-discussion-reproduce/Datasets/Instances/instances_100.json -r 5 -t Instances"
print(f"\n🌟 Command: {command}\n")
process = subprocess.Popen(command, shell=True)
process.wait()

