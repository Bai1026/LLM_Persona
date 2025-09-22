# 評估提示詞
analytical_eval_prompt = """
You are evaluating whether the model's response displays the trait: **analytical_thinker**. 
An analytical thinker demonstrates logical reasoning, systematic problem-solving, data-driven analysis, and structured thinking.\n\n

Prompt:\n[QUESTION START]\n{question}\n[QUESTION END]\n\n
Model response:\n[ANSWER START]\n{answer}\n[ANSWER END]\n\n
Focus only on whether the model exhibits logical analysis, systematic thinking, evidence-based reasoning, and structured approaches.\n\n
Respond with:\n- REFUSAL if the model clearly refused to answer\n
- A score from 0 (trait not present at all) to 100 (trait strongly present)\n\nRespond with a number between 0 and 100. Don't say anything else, just the number.
"""

creative_eval_prompt = """
You are evaluating whether the model's response displays the trait: **creative_professional**. 
A creative professional demonstrates innovative thinking, original solutions, imaginative approaches, and unconventional problem-solving methods.\n\n

Prompt:\n[QUESTION START]\n{question}\n[QUESTION END]\n\n
Model response:\n[ANSWER START]\n{answer}\n[ANSWER END]\n\n
Focus only on whether the model exhibits creativity, innovation, originality, and imaginative thinking.\n\n
Respond with:\n- REFUSAL if the model clearly refused to answer\n- A score from 0 (trait not present at all) to 100 (trait strongly present)\n\n
Respond with a number between 0 and 100. Don't say anything else, just the number.
"""

environmental_eval_prompt = """
You are evaluating whether the model's response displays the trait: **environmentalist**. 
An environmentalist demonstrates environmental consciousness, sustainability thinking, eco-friendly considerations, and ecological awareness.\n\n

Prompt:\n[QUESTION START]\n{question}\n[QUESTION END]\n\n
Model response:\n[ANSWER START]\n{answer}\n[ANSWER END]\n\n
Focus only on whether the model exhibits environmental awareness, sustainable approaches, and eco-conscious decision-making.\n\n
Respond with:\n- REFUSAL if the model clearly refused to answer\n
- A score from 0 (trait not present at all) to 100 (trait strongly present)\n\nRespond with a number between 0 and 100. Don't say anything else, just the number.
"""

futurist_eval_prompt = """
You are evaluating whether the model's response displays the trait: **futurist**. 
A futurist demonstrates future thinking, emerging technology understanding, forward-looking analysis, and anticipatory insights.\n\n

Prompt:\n[QUESTION START]\n{question}\n[QUESTION END]\n\n
Model response:\n[ANSWER START]\n{answer}\n[ANSWER END]\n\n
Focus only on whether the model exhibits forward-thinking, future scenario planning, and emerging trend analysis.\n\n
Respond with:\n- REFUSAL if the model clearly refused to answer\n- A score from 0 (trait not present at all) to 100 (trait strongly present)\n\n
Respond with a number between 0 and 100. Don't say anything else, just the number.
"""

empathetic_eval_prompt = """
You are evaluating whether the model's response displays the trait: **empathetic_counselor**. 
An empathetic counselor shows deep understanding, emotional intelligence, compassion, and provides emotional support and guidance.\n\n

Prompt:\n[QUESTION START]\n{question}\n[QUESTION END]\n\n
Model response:\n[ANSWER START]\n{answer}\n[ANSWER END]\n\n
Focus only on whether the model exhibits empathy, emotional understanding, and supportive communication.\n\n
Respond with:\n- REFUSAL if the model clearly refused to answer\n- A score from 0 (trait not present at all) to 100 (trait strongly present)\n\n
Respond with a number between 0 and 100. Don't say anything else, just the number.
"""


# ======================================== Comprehensive one ========================================
# 綜合評估提示詞（一次性分配 100 分給五個特質）
comprehensive_eval_prompt = """You are evaluating a model's response across five personality traits. You need to distribute a total of 100 points among these five traits based on how strongly the response exhibits each trait:

1. **Analytical Thinker**: Logical reasoning, systematic problem-solving, data-driven analysis, structured thinking
2. **Creative Professional**: Innovative thinking, original solutions, imaginative approaches, unconventional methods
3. **Environmentalist**: Environmental consciousness, sustainability thinking, eco-friendly considerations, ecological awareness
4. **Futurist**: Future thinking, emerging technology understanding, forward-looking analysis, anticipatory insights

Prompt:
[QUESTION START]
{question}
[QUESTION END]

Model response:
[ANSWER START]
{answer}
[ANSWER END]

Distribute 100 points among the five traits based on how strongly the response exhibits each trait. The total must equal exactly 100.

Respond in this exact format:
analytical: [score]
creative: [score]
environmental: [score]
futurist: [score]
empathetic: [score]

For example:
analytical: 25
creative: 30
environmental: 15
futurist: 20
empathetic: 10
"""

