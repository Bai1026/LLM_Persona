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
python generate_multi_role_activations.py
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
python persona_api.py --vector_path "persona_vectors/Qwen2.5-7B-Instruct/multi_role/academic_researcher_response_avg_diff.pt" --layer 20 --coef 2.0
```

### Run the persona api with mulitple characters

```bash
python Experiment/persona_api.py \
--vector_paths creative_professional_response_avg_diff.pt environmentalist_response_avg_diff.pt futurist_response_avg_diff.pt futurist_response_avg_diff.pt \
--fusion_method weighted_average \
--layer 20 \
--coef 2.0
```

- this merge 4 characters' vector then evaluate them.
- This is also the LLM-Dicussion benchmark evaluation setting.

- if wanna run Llama
  - also need to change the folder path in `multi_persona_handler.py` line 101

```bash
python Experiment/persona_api.py --model meta-llama/Llama-3.1-8B-Instruct --vector_paths analytical_thinker_response_avg_diff.pt creative_professional_response_avg_diff.pt environmentalist_response_avg_diff.pt futurist_response_avg_diff.pt futurist_response_avg_diff.pt --fusion_method weighted_average --layer 20 --coef 2.0
```

---

## Evaluation using LLM-Discussion Benchmark

- First: `cd Experiments`

### Auto evaluate after generating dataset

- AUT

```bash
python auto_eval_persona.py \
  -d ../Datasets/AUT/aut_10.json \
  -t AUT \
  -p 1 \
  -v 4
```

- Scientific

```bash
ython auto_eval_persona.py   -d ../Datasets/Scientific/scientific_3.json   -t Scientific   -p 1   -v 4
```

- Similarities

```bash
python auto_eval_persona.py \
  -d ../Datasets/Similarities/similarities_10.json \
  -t Similarities \
  -p 1 \
  -v 4
```

- Instances

```bash
python auto_eval_persona.py \
  -d ../Datasets/Instances/instances_10.json \
  -t Instances \
  -p 1 \
  -v 4
```

- Baseline Version (gpt-4o-mini)

```bash
python auto_eval_persona.py -d ../Datasets/AUT/aut_10.json -t AUT --baseline --openai_model gpt-3.5-turbo-0125
```

- Baseline Version (Qwen2.5)

```bash
python auto_eval_persona.py \
  -d ../Datasets/AUT/aut_100.json \
  -t AUT \
  -p 1 \
  -v 4 \
  -m qwen
```

- Baseline Version (Llama3)

````bash
python auto_eval_persona.py \
  -d ../Datasets/AUT/aut_100.json \
  -t AUT \
  -p 1 \
  -v 4 \
  -m llama
```
### Manual evaluate after generating dataset

```bash
python auto_grade_final.py \
  -i "AUT_persona_api_0908-0548_10" \
  -t sampling \
  -d AUT \
  -v 4 \
  -o y
````
