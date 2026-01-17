gpu=${1:-0}

# --file_path eval_persona_eval/Qwen2.5-7B-Instruct/creative_professional.csv \
# --vector_path persona_vectors/Llama-3.1-8B-Instruct/multi_role/creative_professional_response_avg_diff.pt \

CUDA_VISIBLE_DEVICES=$gpu python -m eval.cal_projection \
    --file_path eval_persona_eval/Data/Llama-3.1-8B-Instruct/cre_env.json \
    --vector_path persona_vectors/Llama-3.1-8B-Instruct/multi_role/creative_professional_response_avg_diff.pt \
    --layer 32 \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --projection_type proj