import subprocess
import time
from datetime import timedelta

start_time = time.time()
print(f"Time Zone: {time.strftime('%Z', time.localtime())}")
print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")

command = f"python auto_eval_persona.py -d /fortress/persona/LLM-discussion/LLM-discussion-reproduce/Datasets/AUT/aut_100.json -t AUT -p 1 -v 4 -m llama -tp 1.0 --no_eval"
print(f"Command: {command}")
process = subprocess.Popen(command, shell=True)
process.wait()

end_time_1 = time.time()

print(f"Time Zone: {time.strftime('%Z', time.localtime())}\n")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")

total_time_1 = end_time_1 - start_time
time_delta_1 = timedelta(seconds=total_time_1)
print(f"First Experiment time period: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))} ~ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_1))}\n")
print(f"First Experiment total time: {time_delta_1}\n")

# ==============================

command = f"python auto_eval_persona.py -d /fortress/persona/LLM-discussion/LLM-discussion-reproduce/Datasets/Instances/instances_100.json -t Instances -p 1 -v 4 -m llama -tp 1.0 --no_eval"
print(f"Command: {command}")
process = subprocess.Popen(command, shell=True)
process.wait()

end_time_2 = time.time()

total_time_2 = end_time_2 - end_time_1
time_delta_2 = timedelta(seconds=total_time_2)
print(f"Second Experiment time period: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_1))} ~ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_2))}\n")
print(f"Second Experiment total time: {time_delta_2}\n")


# ==============================

command = f"python auto_eval_persona.py -d /fortress/persona/LLM-discussion/LLM-discussion-reproduce/Datasets/Scientific/scientific_100.json -t Scientific -p 1 -v 4 -m llama -tp 1.0 --no_eval"
print(f"Command: {command}")
process = subprocess.Popen(command, shell=True)
process.wait()

end_time_3 = time.time()

total_time_3 = end_time_3 - end_time_2
time_delta_3 = timedelta(seconds=total_time_3)
print(f"Third Experiment time period: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_2))} ~ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_3))}\n")
print(f"Third Experiment total time: {time_delta_3}\n")

# ==============================

command = f"python auto_eval_persona.py -d /fortress/persona/LLM-discussion/LLM-discussion-reproduce/Datasets/Similarities/similarities_100.json -t Similarities -p 1 -v 4 -m llama -tp 1.0 --no_eval"
print(f"Command: {command}")
process = subprocess.Popen(command, shell=True)
process.wait()

end_time_4 = time.time()

print(f"Time Zone: {time.strftime('%Z', time.localtime())}\n")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")

total_time_1 = end_time_1 - start_time
time_delta_1 = timedelta(seconds=total_time_1)
print(f"---- \nFirst Experiment time period: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))} ~ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_1))}")
print(f"First Experiment total time: {time_delta_1}\n")

total_time_2 = end_time_2 - end_time_1
time_delta_2 = timedelta(seconds=total_time_2)
print(f"---- \nSecond Experiment time period: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_1))} ~ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_2))}")
print(f"Second Experiment total time: {time_delta_2}\n")

total_time_3 = end_time_3 - end_time_2
time_delta_3 = timedelta(seconds=total_time_3)
print(f"---- \nThird Experiment time period: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_2))} ~ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_3))}")
print(f"Third Experiment total time: {time_delta_3}\n")

total_time_4 = end_time_4 - end_time_3
time_delta_4 = timedelta(seconds=total_time_4)
print(f"---- \nForth Experiment time period: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_3))} ~ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_4))}")
print(f"Forth Experiment total time: {time_delta_4}\n")

# ------------------------------------------------------------

# import requests


# url = "http://127.0.0.1:5000/chat"
# payload = {"message": "who are you"}
# response = requests.post(url, json=payload)

# if response.status_code == 200:
#     print("Result:", response.json()['response'])
# else:
#     print("Error:", response.status_code, response.text)
