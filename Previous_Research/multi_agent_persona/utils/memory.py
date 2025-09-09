import os
from typing import Dict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

Memory = Dict[str, list]

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a memory extractor that analyzes character feedback and organizes it into specific categories. Based on the given feedback, identify and extract insights into the following five areas, using clear and concise natural language:

1. traits: The character's personality and defining attributes. Examples include being defensive, proud, vulnerable, possessive, empathetic, assertive, manipulative, shy, cunning, or loyal.

2. tone: Style and tone of the character's language. Examples include sarcastic, casual, formal, passive-aggressive, confrontational, cold, enthusiastic, overly emotional, or deadpan.

3. inner_conflict: Any signs of emotional struggle or layered feelings. Examples include hiding fear behind confidence, being torn between duty and desire, wanting help but refusing it out of pride, or expressing affection while pretending to be indifferent.

4. interaction_dynamics: Relationship and tension with the user or other characters. Examples include rivalry, friendship, power imbalance, mistrust, dependency, manipulation, mutual respect, or emotional detachment.

5. improvement_suggestions: Concrete recommendations for improving the next generation. Examples include enhancing emotional contrast, making tone more consistent, increasing character distinctiveness, showing more tension in dialogue, or clarifying motivations.

Return the result in the following JSON format:
{
  "traits": [...],
  "tone": [...],
  "inner_conflict": [...],
  "interaction_dynamics": [...],
  "improvement_suggestions": [...]
}
"""

def extract_memory_with_llm(discriminator_advice: str) -> Memory:
    """
    Uses an LLM to extract memory components from discriminator feedback
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Here is the feedback:\n{discriminator_advice}"}
        ],
        temperature=0.3
    )
    content = response.choices[0].message.content
    try:
        memory_dict = eval(content)
        return memory_dict
    except Exception as e:
        print("Failed to parse response format:", e)
        return {}

def merge_memory(old: Memory, new: Memory) -> Memory:
    return {
        k: list(set(old.get(k, []) + new.get(k, []))) for k in old.keys()
    }

def update_memory(prev_memory: Memory, discriminator_advice: str) -> Memory:
    new_memory = extract_memory_with_llm(discriminator_advice)
    return merge_memory(prev_memory, new_memory)



