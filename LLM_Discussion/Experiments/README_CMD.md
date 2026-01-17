

Gemini-2.5-pro (Topline):

```bash
python llm_discussion.py -c config_role_gemini.json -d /workspace/LLM-discussion/LLM-discussion-reproduce/Datasets/AUT/aut_10.json -r 5 -t AUT -e

python auto_eval_persona.py -d /workspace/LLM-discussion/LLM-discussion-reproduce/Datasets/AUT/aut_100.json -t AUT -p 1 -v 4 --topline --gemini_model gemini-2.5-pro --no_eval
```

---

Gemma:
```bash
python llm_discussion.py -c config_role_gemma.json -d /workspace/LLM-discussion/LLM-discussion-reproduce/Datasets/AUT/aut_100.json -r 5 -t AUT
```

---

Qwen:
```bash
python llm_discussion.py -c config_role_qwen.json -d /Users/reiiwang/Research/LLM-discussion/LLM-Discussion/Datasets/AUT/aut_10.json -r 5 -t AUT -e
python llm_discussion.py -c config_role_qwen.json -d /workspace/LLM-discussion/LLM-discussion-reproduce/Datasets/AUT/aut_100.json -r 5 -t AUT -e
```

Baseline Version (single agent), temperature = 1.0
```bash
python auto_eval_persona.py -d /fortress/persona/LLM-discussion/LLM-discussion-reproduce/Datasets/AUT/aut_100.json -t AUT -p 1 -v 4 -m qwen -tp 1.0
python auto_eval_persona.py -d /workspace/LLM-discussion/LLM-discussion-reproduce/Datasets/AUT/aut_100.json -t AUT -p 1 -v 4 -m gemma -tp 1.0
```

Baseline Version (single agent)
```bash
python auto_eval_persona.py -d /fortress/persona/LLM-discussion/LLM-discussion-reproduce/Datasets/Similarities/similarities_3.json -t Similarities -p 1 -v 4 -m qwen -a
```
Line 124: `MULTI_ROLE_PLAY = False`

Baseline Version (MULTI_ROLE_PLAY)
```bash
python auto_eval_persona.py \
  -d ../Datasets/AUT/aut_100.json \
  -t AUT \
  -p 1 \
  -v 4 \
  -m qwen
```
Line 124: `MULTI_ROLE_PLAY = True`


---

Llama3.1:
```bash
python llm_discussion.py -c config_role_llama.json -d /Users/reiiwang/Research/LLM-discussion/LLM-Discussion/Datasets/AUT/aut_10.json -r 5 -t AUT -e
python llm_discussion.py -c config_role_llama.json -d /workspace/LLM-discussion-reproduce/Datasets/Instances/instances_100.json -r 5 -t Instances
```

Baseline Version (single agent)
```bash
python auto_eval_persona.py -d /workspace/LLM-discussion/LLM-discussion-reproduce/Datasets/Instances/instances_3.json -t Instances -p 1 -v 4 -m llama 
```

Baseline Version (MULTI_ROLE_PLAY)
```bash
python auto_eval_persona.py -d /fortress/persona/LLM-discussion/LLM-discussion-reproduce/Datasets/Similarities/similarities_3.json -t Similarities -p 1 -v 4 -m llama -a
```
Line 124: `MULTI_ROLE_PLAY = True`



- dynamo 沒改 input, output很低 -> 一直重複
(gemmaModel copy.py)
LLM-discussion-reproduce/Results/AUT/chat_log/AUT_multi_debate_roleplay_4_5_gemma-3-4b-it_Environmentalist-CreativeProfessional-Futurist-Futurist_chat_log_2025-09-28-15-19-23_2.json

- 修改 dynamo  input: 6146, output: 2048 -> 還不錯 偶爾亂說話
LLM-discussion-reproduce/Results/AUT/chat_log/AUT_multi_debate_roleplay_4_5_gemma-3-4b-it_Environmentalist-CreativeProfessional-Futurist-Futurist_chat_log_2025-09-28-16-32-56_2.json

LLM-discussion-reproduce/Results/AUT/chat_log/AUT_multi_debate_roleplay_4_5_gemma-3-4b-it_Environmentalist-CreativeProfessional-Futurist-Futurist_chat_log_2025-09-28-16-44-00_2.json