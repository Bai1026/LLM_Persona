# README of Bai Persona Vectors

## What did I add?

```bash
📦persona_vectors
 ┣ ...
 ┣ 📜.env
 ┣ 📜README_BAI.md
 ┣ 📜create_multi_role_dataset.py
 ┣ 📜generate_multi_role_activations.py
 ┣ 📜generate_multi_role_vectors.py
 ┣ 📜interactive_chat.py
 ┣ 📜...
```

---
## Before getting the character persona vector
### Generate dataset
```bash
python create_multi_role_dataset.py
```

### Generate activations using positive and negative system prompts:

```bash
cd persona_vectors
python Experiment/generate_multi_role_activations.py
```

- the eval_prompt should be like:
```bash
"eval_prompt": "You are evaluating whether the model's response displays the trait: **creative_professional**. This involves assessing if the response demonstrates creative thinking, aesthetic considerations, narrative ability, emotional resonance, and artistic sensibility. Creative professional responses infuse solutions with beauty and depth, think expressively, ensure resonance on human level, use design thinking approaches, and balance functional and emotional considerations.\n\nPrompt:\n[QUESTION START]\n{question}\n[QUESTION END]\n\nModel response:\n[ANSWER START]\n{answer}\n[ANSWER END]\n\nFocus only on whether the model exhibits this trait.\n\nRespond with:\n- REFUSAL if the model clearly refused to answer\n- A score from 0 (trait not present at all) to 100 (trait strongly present)\n\nRespond with a number between 0 and 100. Don't say anything else, just the number."
```

### Generate persona vector using mean difference between positive and negative activations:

```bash
python generate_multi_role_vectors.py
```

---
## After getting the character persona vector

### Run the interactive chat

```bash
python interactive_chat.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --vector_path "persona_vectors/Qwen2.5-7B-Instruct/multi_role/empathetic_counselor_response_avg_diff.pt" \
    --layer 20 \
    --coef 2.0
```

### Run the persona api for evaluation
```bash
python persona_api.py --vector_path "persona_vectors/Qwen2.5-7B-Instruct/multi_role/empathetic_counselor_response_avg_diff.pt" --layer 20 --coef 2.0
```

### Run the persona api with mulitple characters
```bash
python Experiment/persona_api.py --vector_paths environmentalist_response_avg_diff.pt creative_professional_response_avg_diff.pt futurist_response_avg_diff.pt futurist_response_avg_diff.pt --fusion_method weighted_average --layer 20 --coef 2.0
```
- this merge 4 characters' vector then evaluate them.
- default is the config of LLM-DIscussion role-playing setting.

---
## Evaluation using LLM-Discussion Benchmark

- First: `cd Experiments`

### Auto evaluate after generating dataset
```bash
python auto_eval_persona.py \
  -d ../Datasets/AUT/aut_2.json \
  -t AUT \
  -p 1 \
  -v 4
```
Baseline Version (gpt-4o-mini)
```bash
python auto_eval_persona.py \
  -d ../Datasets/AUT/aut_2.json \
  -t AUT \
  -p 1 \
  -v 4 \
  --baseline \
  --model gpt-4
```

### Manual evaluate after generating dataset
```bash
python auto_grade_final.py \
  -i "AUT_persona_api_1_1_PersonaAPI_PersonaAPI_persona_api_20250901-025003_2" \
  -t sampling \
  -d AUT \
  -v 4 \
  -o y
```
