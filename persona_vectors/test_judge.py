#!/usr/bin/env python3
"""測試 OpenAI Judge 是否能正確評分 creative_professional"""

import json
import asyncio
from judge import OpenAiJudge

async def test_creative_professional_judge():
    # 載入 creative_professional 的 eval_prompt
    with open("data_generation/trait_data_extract/creative_professional.json", "r") as f:
        trait_data = json.load(f)
    
    eval_prompt = trait_data["eval_prompt"]
    print("📝 使用的 eval_prompt:")
    print(eval_prompt)
    print("\n" + "="*50)
    
    # 建立 judge
    judge = OpenAiJudge("gpt-4o-mini", eval_prompt, eval_type="0_100")
    
    # 測試問題和回答
    test_question = "How would you solve a complex problem at work?"
    test_answer = """Solving a complex problem at work involves a structured approach that taps into both creative and systematic thinking. Here's a step-by-step guide to effectively address such challenges:

### 1. Define the Problem Clearly
- **Gather Information:** Collect all relevant data and details about the problem.
- **Clarify Objectives:** Understand what you are trying to achieve and the desired outcomes.
- **Identify Stakeholders:** Recognize who will be affected by the problem and the solution.

### 2. Analyze the Problem
- **Root Cause Analysis:** Use tools such as the Five Whys, Fishbone Diagram, or SWOT analysis to break down the problem into its core components.
- **Data Collection and Analysis:** Conduct research, surveys, or gather relevant data to understand the extent of the problem.
- **Stakeholder Feedback:** Engage stakeholders in discussions to gain diverse perspectives and insights.

### 3. Generate Ideas
- **Brainstorming:** Facilitate a brainstorming session with your team. Encourage wild, creative ideas.
- **Idea Mapping:** Use visual or digital tools to map out ideas and their potential impacts.
- **External Input:** Consider outside experts or consultants for fresh perspectives."""

    print(f"🤖 測試問題: {test_question}")
    print(f"🤖 測試回答: {test_answer[:200]}...")
    print("\n" + "="*50)
    
    try:
        # 呼叫 judge
        print("🔍 呼叫 OpenAI Judge...")
        score = await judge(question=test_question, answer=test_answer)
        print(f"✅ 獲得分數: {score}")
        print(f"📊 分數類型: {type(score)}")
        
        if score is None:
            print("❌ 分數是 None - 這就是 NaN 問題的原因！")
        else:
            print(f"✅ 分數正常: {score}")
            
    except Exception as e:
        print(f"❌ 測試失敗: {e}")

if __name__ == "__main__":
    asyncio.run(test_creative_professional_judge())
