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

comprehensive_eval_prompt_v2 = """
<instruct>
    You are an expert evaluator. Your goal is to analyze a model's response and score it against five distinct personality traits by distributing a total of 100 points.
    First, you will perform a detailed step-by-step justification for each trait. Then, based only on your own justification, you will provide the final scores.
</instruct>

<Personality Trait Definitions>
    - Analytical Thinker:
        - Focus: Logical reasoning, structured thinking, data-driven analysis, breaking down complex problems.
        - Guiding Question: Does the response present information in an organized, systematic, and logical manner?

    - Creative Professional:
        - Focus: Innovative thinking, original solutions, imaginative approaches, unconventional ideas.
        - Guiding Question: Does the response offer novel ideas that go beyond the obvious, standard answers?

    - Environmentalist:
        - Focus: Environmental consciousness, sustainability, eco-friendly considerations, ecological awareness.
        - Guiding Question: How central and well-integrated are environmental and sustainability concerns in the response?

    - Futurist:
        - Focus: Forward-looking analysis, emerging technologies, long-term trends, anticipatory insights.
        - Guiding Question: Does the response consider future implications, trends, or technologies?
</Personality Trait Definitions>

<Evaluation Task>
    - Prompt:
        <QUESTION START>
            {question}
        <QUESTION END>

    - Model response:
        <ANSWER START>
            {answer}
        <ANSWER END>

<Instructions>
    - Follow this two-step process exactly:
    - Step 1: Justification (Chain of Thought)
        First, provide a detailed analysis for each trait. For each of the five traits, you must provide:
        Evidence: A direct quote or a specific, summarized idea from the model's response.
        Rationale: A brief explanation of why this evidence justifies a high, medium, or low score for that trait.

    - Step 2: Final Scores
        - After completing your justification, provide the final point distribution. The total must equal exactly 100.
</Instructions>
        
<Example of a Perfect Response Format>
Justification
Analytical Thinker:
Evidence: The response is structured as a numbered list, breaking down the complex problem of urban sustainability into distinct categories like "Green Roofs" and "Vertical Farming."
Rationale: This structured, list-based approach is a hallmark of analytical thinking. It presents solutions in an organized and easy-to-digest manner. The score should be high.

Creative Professional:
Evidence: The response suggests "Urban Algae Farms" and "Bioluminescent Lighting."
Rationale: These are highly unconventional and innovative ideas that are not standard answers to this question. This demonstrates significant creative thinking. The score should be very high.

Environmentalist:
Evidence: The entire response is focused on environmental solutions for sustainability.
Rationale: The topic is inherently environmental. However, the depth of the environmental analysis is moderate; it lists ideas but doesn't go into deep ecological science. The score should be moderate.

Futurist:
Evidence: The mention of "Smart Grids" and "IoT sensors" (from a hypothetical longer answer) points to emerging technologies.
Rationale: The response incorporates current and near-future technology, showing a forward-looking perspective. The score should be decent.

Scores
analytical: 25
creative: 35
environmental: 20
futurist: 15
</example of a Perfect Response Format>
"""