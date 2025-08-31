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

## Run the interactive chat

```bash
python interactive_chat.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --vector_path "persona_vectors/Qwen2.5-7B-Instruct/multi_role/empathetic_counselor_response_avg_diff.pt" \
    --layer 20 \
    --coef 2.0
```