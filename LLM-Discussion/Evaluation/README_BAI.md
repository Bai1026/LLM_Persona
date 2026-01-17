# README for new eval

- This new evaluation is based on QA pair in `chat_log.json`
- This new evaluation only use `Originality` and `Elaboration`

## First get the chat log json file

- same method including: `Vector Merge`, `LLM Discussion`, `Single Agent with Multi-Role Prompt`, and `Single Agent`

## Put into the folder you wanna evaluate`Main_Result` folder with corresponding task name

```bash
python simple_eval_{task_name}.py
```

- Different task needs to use different eval script since the prompts are different
- Results would be saved in the same folder with postfix `_simple_eval_results,json`
