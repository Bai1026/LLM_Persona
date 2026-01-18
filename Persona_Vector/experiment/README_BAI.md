# Persona Vector API Guide

This guide explains how to create persona vectors and launch API services using **Qwen**, **Llama**, and **Gemma** models.

## 📋 Table of Contents

- [Creating Persona Vectors](#creating-persona-vectors)
  - [1. Generate Dataset](#1-generate-dataset)
  - [2. Generate Activations](#2-generate-activations)
  - [3. Generate Persona Vectors](#3-generate-persona-vectors)
- [Launch API Service](#launch-api-service)
  - [Single Persona API](#single-persona-api)
  - [Multi-Persona Fusion API](#multi-persona-fusion-api)
- [Supported Models](#supported-models)

---

## Creating Persona Vectors

### 1. Generate Dataset

First, create a persona trait dataset that defines positive instructions and evaluation questions for each persona.

```bash
cd /workspace/LLM_Persona/Persona_Vector
python experiment/create_multi_role_dataset.py
```

**Description**: This script generates persona trait files (JSON format) in the `data_generation/trait_data_extract/` directory, containing:
- `positive_instructions`: Persona trait instructions
- `questions`: Evaluation questions
- `eval_prompt`: Evaluation prompts

### 2. Generate Activations

Generate model activation data using positive and negative system prompts.

```bash
python generate_multi_role_activations.py
```

**Model Selection**: Edit lines 8-15 in `generate_multi_role_activations.py` to uncomment and select the target model:

```python
# Choose one of the models
def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):       # Qwen model
# def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):  # Llama model
# def __init__(self, model_name: str = "google/gemma-3-4b-it"):        # Gemma model
```

**Output**: Generated in `eval_persona_extract/{model_name}/` directory:
- `{role_name}_pos_instruct.csv` - Positive trait activation data
- `{role_name}_neutral_instruct.csv` - Neutral baseline activation data

### 3. Generate Persona Vectors

Calculate the mean difference between positive and negative activations to create persona vectors.

```bash
python generate_multi_role_vectors.py
```

**Output**: Generated in `experiment/persona_vectors/` directory:
- `{role_name}_response_avg_diff.pt` - Persona vector file

---

## Launch API Service

### Single Persona API

Launch API service using a single persona vector:

```bash
cd experiment

# Example: Using academic researcher persona
python persona_api.py \
  --vector_paths persona_vectors/academic_researcher_response_avg_diff.pt \
  --layer 20 \
  --coef 2.0
```

### Multi-Persona Fusion API

Combine multiple persona vectors using fusion strategies to create hybrid personalities:

#### 🔹 Qwen Model (Default)

```bash
python persona_api.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --vector_paths creative_professional_response_avg_diff.pt \
                 environmentalist_response_avg_diff.pt \
                 futurist_response_avg_diff.pt \
  --fusion_method weighted_average \
  --layer 20 \
  --coef 2.0 \
  --port 5000
```

#### 🔹 Llama Model

```bash
python persona_api.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --vector_paths creative_professional_response_avg_diff.pt \
                 environmentalist_response_avg_diff.pt \
                 futurist_response_avg_diff.pt \
  --fusion_method weighted_average \
  --layer 20 \
  --coef 2.0 \
  --port 5001
```

#### 🔹 Gemma Model

```bash
python persona_api.py \
  --model google/gemma-3-4b-it \
  --vector_paths creative_professional_response_avg_diff.pt \
                 environmentalist_response_avg_diff.pt \
                 futurist_response_avg_diff.pt \
  --fusion_method weighted_average \
  --layer 20 \
  --coef 2.0 \
  --port 5002
```

### API Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model name | `meta-llama/Llama-3.1-8B-Instruct` |
| `--vector_paths` | Persona vector file paths (multiple) | Required |
| `--layer` | Target layer index | `20` |
| `--coef` | Steering coefficient (control strength) | `2.0` |
| `--fusion_method` | Vector fusion method | `weighted_average` |
| `--host` | API host address | `127.0.0.1` |
| `--port` | API port | `5000` |

#### Running as Single Agent (No Steering)

To run the model as a **single agent without persona steering**, simply set `--coef 0`:

```bash
python persona_api.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --vector_paths persona_vectors/creative_professional_response_avg_diff.pt \
  --layer 20 \
  --coef 0 \
  --port 5000
```

This disables the steering mechanism, making the model behave as a standard single agent without persona influence.

#### Multi-Role Prompt (MRP) Mode

To enable **Multi-Role Prompt (MRP)** mode, which adds role-playing instructions to the system prompt:

1. Open [persona_api.py](persona_api.py) or [persona_api_diff_layer.py](persona_api_diff_layer.py)
2. Find the `ROLE_PROMPT` setting (around line 32):

```python
# Role Prompt
ROLE_PROMPT = False  # Set to True to enable role-play in system prompt
```

3. Set `ROLE_PROMPT = True` to include role-playing instructions in the system prompt

This enables **SA with Multi-Role Prompt (MRP)**, where the model receives explicit role-playing instructions alongside the persona vectors.

### Fusion Method Options

- `weighted_average`: Weighted average fusion
- `concatenate`: Concatenation fusion
- `attention`: Attention mechanism fusion
- `dynamic`: Dynamic fusion

### API Endpoints

After launching, the following endpoints are available:

- `POST /chat` - Send messages
- `POST /set_persona_weights` - Set persona weights
- `POST /set_persona_mode` - Set mode
- `POST /reset` - Reset conversation
- `GET /status` - Get status

**Example Request**:

```bash
curl -X POST http://127.0.0.1:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_input": "Tell me about renewable energy.", "max_tokens": 1000}'
```

---

## Supported Models

| Model | Model Name | Recommended Layer |
|-------|------------|-------------------|
| **Qwen 2.5** | `Qwen/Qwen2.5-7B-Instruct` | 20 |
| **Llama 3.1** | `meta-llama/Llama-3.1-8B-Instruct` | 20 |
| **Gemma 3** | `google/gemma-3-4b-it` | 20 |

---

## Integration with LLM_Discussion

### Generate Dataset with Persona API

After launching the Persona API, you can use `auto_eval_persona.py` in the LLM_Discussion project for automatic evaluation:

#### AUT (Alternative Uses Task)

```bash
cd ../../LLM_Discussion/Experiments

# Persona model evaluation
python auto_eval_persona.py \
  -d ../Datasets/AUT/aut_100.json \
  -t AUT \
  -p 1 \
  -v 4
```

#### Scientific (Scientific Questions)

```bash
python auto_eval_persona.py \
  -d ../Datasets/Scientific/scientific_100.json \
  -t Scientific \
  -p 1 \
  -v 4
```

#### Similarities (Similarity Tasks)

```bash
python auto_eval_persona.py \
  -d ../Datasets/Similarities/similarities_100.json \
  -t Similarities \
  -p 1 \
  -v 4
```

#### Instances (Instance Tasks)

```bash
python auto_eval_persona.py \
  -d ../Datasets/Instances/instances_100.json \
  -t Instances \
  -p 1 \
  -v 4
```

### Baseline Model Evaluation

#### Qwen Baseline

```bash
python auto_eval_persona.py \
  -d ../Datasets/Scientific/scientific_100.json \
  -t Scientific \
  -p 1 \
  -v 4 \
  -m qwen
```

#### Llama Baseline

```bash
python auto_eval_persona.py \
  -d ../Datasets/AUT/aut_100.json \
  -t AUT \
  -p 1 \
  -v 4 \
  -m llama
```

#### Gemma Baseline

```bash
python auto_eval_persona.py \
  -d ../Datasets/AUT/aut_100.json \
  -t AUT \
  -p 1 \
  -v 4 \
  -m gemma
```

#### OpenAI Baseline

```bash
python auto_eval_persona.py \
  -d ../Datasets/AUT/aut_10.json \
  -t AUT \
  --baseline \
  --openai_model gpt-4o-mini
```

### Parameter Descriptions

| Parameter | Description |
|-----------|-------------|
| `-d` | Dataset path |
| `-t` | Task type (AUT/Scientific/Similarities/Instances) |
| `-p` | Persona mode (1: enabled) |
| `-v` | Version number |
| `-m` | Model type (qwen/llama/gemma) |
| `--baseline` | Use baseline mode |
| `--no_eval` | Skip evaluation, only generate data |

### Manual Evaluation

After generating the dataset, you can use the following command for manual evaluation:

```bash
python auto_grade_final.py \
  -i "Similarities_persona_api_0920-2032_100" \
  -t sampling \
  -d Similarities \
  -v 4 \
  -o y
```

For more details, please refer to the [LLM_Discussion documentation](../../LLM_Discussion/README.md).

---

## Advanced Features

### Multi-Layer Steering

Use `persona_api_diff_layer.py` to apply different steering coefficients to different layers:

```bash
python persona_api_diff_layer.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --vector_paths persona1.pt persona2.pt \
  --layers 10 20 \
  --coefs 1.5 2.0
```

### Interactive Chat

Use `interactive_chat.py` for interactive testing:

```bash
python interactive_chat.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --vector_path persona_vectors/creative_professional_response_avg_diff.pt \
  --layer 20 \
  --coef 2.0
```

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `max_tokens` parameter
   - Use smaller models (e.g., gemma-3-4b-it)
   - Reduce the number of simultaneously loaded personas

2. **Vector Dimension Mismatch**
   - Ensure vectors are generated using the same model
   - Check if model versions are consistent

3. **API Connection Failed**
   - Verify port is not occupied
   - Check firewall settings
   - Use `--host 0.0.0.0` to allow external connections

### Check Logs

When the API starts, it will display:
```
🚀 正在初始化多重 Persona API 服務...
📁 載入 persona 向量: [...]
✅ API 服務啟動於 http://127.0.0.1:5000
```

---

## File Structure

```
experiment/
├── README_BAI.md                          # This document
├── create_multi_role_dataset.py           # Create persona dataset
├── persona_api.py                         # Persona API main program
├── persona_api_diff_layer.py              # Multi-layer steering API
├── multi_persona_handler.py               # Multi-persona handler
├── interactive_chat.py                    # Interactive chat
├── persona_vectors/                       # Persona vector storage directory
│   ├── creative_professional_response_avg_diff.pt
│   ├── environmentalist_response_avg_diff.pt
│   └── ...
└── eval_persona_extract/                  # Activation data storage directory
    ├── Qwen2.5-7B-Instruct/
    ├── Llama-3.1-8B-Instruct/
    └── gemma-3-4b-it/
```
