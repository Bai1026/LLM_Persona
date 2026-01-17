import subprocess
import time
from datetime import timedelta

start_time = time.time()
print(f"Time Zone: {time.strftime('%Z', time.localtime())}")
print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")

# command = f"python llm_discussion.py -c config_role_qwen.json -d /workspace/LLM-discussion/LLM-discussion-reproduce/Datasets/Instances/instances_100.json -r 5 -t Instances"
# print(f"\n🌟 Command: {command}\n")
# process = subprocess.Popen(command, shell=True)
# process.wait()
# output = process.stdout.read().decode()
# print(output)

end_time_1 = time.time()

command = f"python llm_discussion.py -c config_role_qwen.json -d /workspace/LLM-discussion/LLM-discussion-reproduce/Datasets/Similarities/similarities_100.json -r 5 -t Similarities"
print(f"\n🌟 Command: {command}\n")
process = subprocess.Popen(command, shell=True)
process.wait()
# output = process.stdout.read().decode()
# print(output)

end_time_2 = time.time()

command = f"python llm_discussion.py -c config_role_qwen.json -d /workspace/LLM-discussion/LLM-discussion-reproduce/Datasets/Scientific/scientific_100.json -r 5 -t Scientific"
print(f"\n🌟 Command: {command}\n")
process = subprocess.Popen(command, shell=True)
process.wait()
# output = process.stdout.read().decode()
# print(output)

end_time_3 = time.time()

print(f"Time Zone: {time.strftime('%Z', time.localtime())}\n")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")

total_time_1 = end_time_1 - start_time
time_delta_1 = timedelta(seconds=total_time_1)
print(f"First Experiment time period: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))} ~ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_1))}\n")
print(f"First Experiment total time: {time_delta_1}\n")

total_time_2 = end_time_2 - end_time_1
time_delta_2 = timedelta(seconds=total_time_2)
print(f"Second Experiment time period: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_1))} ~ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_2))}\n")
print(f"Second Experiment total time: {time_delta_2}\n")

total_time_3 = end_time_3 - end_time_2
time_delta_3 = timedelta(seconds=total_time_3)
print(f"Third Experiment time period: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_2))} ~ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_3))}\n")
print(f"Third Experiment total time: {time_delta_3}\n")


# ------------------------------------------------------------

# import requests


# url = "http://127.0.0.1:5000/chat"
# payload = {"message": "who are you"}
# response = requests.post(url, json=payload)

# if response.status_code == 200:
#     print("Result:", response.json()['response'])
# else:
#     print("Error:", response.status_code, response.text)
