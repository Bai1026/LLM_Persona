import subprocess
import time
from datetime import timedelta

start_time = time.time()
print(f"Time Zone: {time.strftime('%Z', time.localtime())}")
print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")

command = f"python llm_discussion.py -c config_role.json -d /home/u9655801/LLM-discussion2/LLM-discussion-reproduce/Datasets/AUT/aut_10.json -r 2 -t AUT"
print(f"\n🌟 Command: {command}\n")
process = subprocess.Popen(command, shell=True)
process.wait()
# output = process.stdout.read().decode()
# print(output)

end_time_1 = time.time()

print(f"Time Zone: {time.strftime('%Z', time.localtime())}\n")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")

total_time_1 = end_time_1 - start_time
time_delta_1 = timedelta(seconds=total_time_1)
print(f"First Experiment time period: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))} ~ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_1))}\n")
print(f"First Experiment total time: {time_delta_1}\n")
