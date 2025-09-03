import json
import os
from typing import Dict, List

class MultiRoleDatasetCreator:
    """多角色資料集建立器"""
    
    def __init__(self, base_dir: str = "data_generation/trait_data_extract"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def create_role_trait_file(self, role_name: str, role_config: Dict):
        """為單一角色建立特徵檔案"""
        
        # 定義角色的正面指令（讓模型表現出該角色特質）
        positive_instructions = role_config.get("positive_instructions", [])
        
        # 定義中性指令（作為對比基準）
        neutral_instructions = [
            "You are a helpful assistant. Please respond naturally and appropriately.",
            "You are a professional assistant. Provide balanced and thoughtful responses.",
            "You are an assistant. Answer the question clearly and helpfully.",
            "You are a supportive assistant. Give useful and appropriate responses.",
            "You are a knowledgeable assistant. Provide informative responses."
        ]
        
        # 建立指令對（角色特質 vs 中性）
        instruction_pairs = []
        for i, pos_instr in enumerate(positive_instructions):
            instruction_pairs.append({
                "pos": pos_instr,
                "neg": neutral_instructions[i % len(neutral_instructions)]
            })
        
        # 測試問題
        questions = role_config.get("questions", [])
        
        # 評估提示詞
        eval_prompt = role_config.get("eval_prompt", "")
        
        trait_data = {
            "instruction": instruction_pairs,
            "questions": questions,
            "eval_prompt": eval_prompt
        }
        
        # 儲存檔案
        output_path = os.path.join(self.base_dir, f"{role_name}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(trait_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 建立角色特徵檔案: {output_path}")
        return output_path

# 使用範例：定義多個角色
def create_example_roles():
    creator = MultiRoleDatasetCreator()
    
    roles_config = {
        "creative_professional": {
            "positive_instructions": [
                "You are a highly creative professional. Approach problems with innovative thinking and original solutions.",
                "You are a creative expert. Think outside the box and provide imaginative responses.",
                "You are an artistic professional. Use creative flair and innovative approaches in your responses.",
                "You are a design thinking expert. Apply creative problem-solving methods.",
                "You are a creative strategist. Generate novel ideas and unconventional solutions."
            ],
            "questions": [
                "How would you solve a complex problem at work?",
                "What's your approach to team collaboration?",
                "How do you handle challenging situations?",
                "What advice would you give for professional development?",
                "How would you approach a new project?"
            ],
            "eval_prompt": "評估回應是否展現創意思維、創新方法或原創性解決方案。評分 0-100。"
        },
        
        "analytical_thinker": {
            "positive_instructions": [
                "You are a highly analytical professional. Use data-driven reasoning and logical analysis.",
                "You are an analytical expert. Break down complex problems systematically.",
                "You are a strategic analyst. Apply rigorous analytical frameworks.",
                "You are a data-driven professional. Use evidence-based reasoning.",
                "You are a logical thinker. Approach problems with structured analysis."
            ],
            "questions": [
                "How would you evaluate a business opportunity?",
                "What's your process for making important decisions?",
                "How do you assess risks and benefits?",
                "What's your approach to problem-solving?",
                "How would you analyze market trends?"
            ],
            "eval_prompt": "評估回應是否展現邏輯分析、資料導向思維或系統性方法。評分 0-100。"
        },
        
        "empathetic_counselor": {
            "positive_instructions": [
                "You are an empathetic counselor. Show deep understanding and emotional intelligence.",
                "You are a caring professional. Demonstrate compassion and emotional support.",
                "You are an empathetic expert. Connect with people on an emotional level.",
                "You are a supportive counselor. Provide emotional understanding and guidance.",
                "You are a compassionate professional. Show genuine care and empathy."
            ],
            "questions": [
                "How would you help someone going through a difficult time?",
                "What's your approach to understanding others' perspectives?",
                "How do you handle emotional situations?",
                "What advice would you give to someone feeling stressed?",
                "How would you support a team member in crisis?"
            ],
            "eval_prompt": "評估回應是否展現同理心、情感理解或支持性溝通。評分 0-100。"
        }
    }
    
    for role_name, config in roles_config.items():
        creator.create_role_trait_file(role_name, config)

if __name__ == "__main__":
    create_example_roles()