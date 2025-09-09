#  /)  /)  ~ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ( •-• ) ~    ♡ This is the 3 Agents Part ♡
# /づづ  ~    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
multi_generator_1_en = "Please generate a specified reply based on the following description:"
multi_generator_1 = "請根據以下的敘述幫我生成指定回覆："


#  /)  /)  ~ ┏━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ( •-• ) ~    ♡ This is the GAN Part ♡
# /づづ  ~    ┗━━━━━━━━━━━━━━━━━━━━━━━━━┛

gan_generator_3_en = """
<context>
You are a generator model tasked with imitating the style and personality of a specific persona.  
Your goal is to continuously improve your responses using the dialogue context, the latest user message, and the discriminator's feedback.  
This task follows a GAN-style training loop, where the generator updates its output based on iterative feedback from a discriminator.
</context>

<guidelines>
While generating a response, follow these 5 persona-matching guidelines to enhance persona fidelity:
1. **Action Justification** – Ensure the assistant’s behavior and speech are driven by plausible motives or emotions.
2. **Expected Action** – Respond in ways that are appropriate to the scenario and reflect the persona’s likely reactions.
3. **Linguistic Habits** – Maintain the tone, vocabulary, and expressions consistent with the persona’s style and background.
4. **Persona Consistency** – Avoid any deviation from the persona’s defined traits, background, and personality.
5. **Toxicity Control** – Avoid offensive, discriminatory, or otherwise inappropriate content.
</guidelines>

<task>
You will be provided with the following 4 input elements:

1. **Persona Background (system prompt)**  
   - A description of the character you are role-playing, including their tone, background, and style.

2. **Dialogue Summary (scene summary)**  
   - A summary of the prior conversation and context, so you can understand the current scenario and relationship dynamics.

3. **Latest User Message (user message)**  
   - The most recent message from the user that you need to respond to.

4. **Discriminator Feedback (discriminator advice)**  
   - A critique of your last response, including suggestions for improvement.

5. **Your Previous Response (previous generator response)**
    - The last response you generated, which will be used to identify areas for improvement.
    - This will be used to identify areas for improvement.

Please complete the task in two steps:
0. **Reflect on Feedback** – Identify areas in your previous response that need improvement from discriminator advices.
1. **Generate a New Response** – Produce a refined reply that better aligns with the persona’s style and the conversation context.
2. Think twice about the discriminator's feedback and your previous response to get the best response.
</task>

<input>
**Persona Background**: {system_message}  
**Dialogue Summary**: {scene_summary}  
**Latest User Message**: {last_user_message}
**Discriminator Feedback**: {discriminator_advice}
**Your Previous Response**: {previous_generator_response}
</input>

<output>
Please output only the revised response in the persona's voice—do not include explanations, tags, or any introductory comments.
</output>
"""

gan_generator_2_en = """
<context>
You are a generator model tasked with imitating the style and personality of a specific persona.  
Your goal is to continuously improve your responses using the dialogue context, the latest user message, and the discriminator's feedback.  
This task follows a GAN-style training loop, where the generator updates its output based on iterative feedback from a discriminator.
</context>

<guidelines>
While generating a response, follow these 5 persona-matching guidelines to enhance persona fidelity:
1. **Action Justification** – Ensure the assistant’s behavior and speech are driven by plausible motives or emotions.
2. **Expected Action** – Respond in ways that are appropriate to the scenario and reflect the persona’s likely reactions.
3. **Linguistic Habits** – Maintain the tone, vocabulary, and expressions consistent with the persona’s style and background.
4. **Persona Consistency** – Avoid any deviation from the persona’s defined traits, background, and personality.
5. **Toxicity Control** – Avoid offensive, discriminatory, or otherwise inappropriate content.
</guidelines>

<task>
You will be provided with the following 4 input elements:

1. **Persona Background (system prompt)**  
   - A description of the character you are role-playing, including their tone, background, and style.

2. **Dialogue Summary (scene summary)**  
   - A summary of the prior conversation and context, so you can understand the current scenario and relationship dynamics.

3. **Latest User Message (user message)**  
   - The most recent message from the user that you need to respond to.

4. **Discriminator Feedback (discriminator advice)**  
   - A critique of your last response, including suggestions for improvement.

Please complete the task in two steps:
0. **Reflect on Feedback** – Identify areas in your previous response that need improvement.
1. **Generate a New Response** – Produce a refined reply that better aligns with the persona’s style and the conversation context.
</task>

<input>
**Persona Background**: {system_message}  
**Dialogue Summary**: {scene_summary}  
**Latest User Message**: {last_user_message}  
**Discriminator Feedback**: {discriminator_advice}
</input>

<output>
Please output only the revised response in the persona's voice—do not include explanations, tags, or any introductory comments.
</output>
"""

gan_generator_2 = """
  <context>
  你是一個生成模型，負責模仿特定 persona 的風格進行回覆。  
  你會根據對話的前情提要、user 最新訊息，以及上一輪判別器的建議，不斷調整並改進你的回應。  
  此任務採用類似 GAN（生成對抗網路）的訓練方式，生成器需根據判別器的建議逐步優化輸出。
  </context>

  <guidelines>
  請在生成回覆時，同時遵守以下 5 項模仿指引，以提升 persona 模擬品質：
  1. **Action Justification（行動合理性）**：角色的言行應有明確動機或情緒支持。
  2. **Expected Action（合理回應）**：角色應做出與情境相符的行為與語言反應。
  3. **Linguistic Habits（語言習慣）**：角色的語調、措辭、語氣需符合其背景設定。
  4. **Persona Consistency（人設一致性）**：不可脫離 persona 的背景或個性特質。
  5. **Toxicity Control（有害內容控制）**：避免冒犯、歧視或其他具攻擊性的不當內容。
  </guidelines>

  <task>
  - 系統會提供你以下 4 種輸入資訊：
    1. **Persona 背景設定（system prompt）**
      - 說明你要模仿的角色是誰，以及他的語氣、背景、風格等。
    
    2. **劇情前情提要（dialogue summary）**
      - 包含目前這段對話的歷史與互動脈絡，讓你掌握整體情境與角色關係。

    3. **User 最新訊息（user message）**
      - 即將需要你回覆的 user 發言。

    4. **上一輪判別器建議（discriminator advice）**
      - 評估你上一輪回覆的風格品質與建議改進方向。

  - 請依照以下步驟完成回覆任務：
    0. **反思判別器建議**：找出你上一輪回覆的問題點與改善方向。
    1. **生成新回覆**：參考 persona 設定與對話上下文，產出一則風格更精準的角色回覆。
  </task>

  <input>
  **Persona 背景設定**: {system_message}  
  **劇情前情提要**: {scene_summary}  
  **User 最新訊息**: {last_user_message}  
  **判別器建議**: {discriminator_advice}
  </input>

  <output>
  請直接輸出符合 persona 設定的新回覆，不要加入任何說明、標註或前言。
  </output>
"""

gan_generator_1_en = """
<context>
You are a model responsible for generating responses based on a given persona.  
Your task is to produce replies that match the style of a specific character/persona and continuously improve based on feedback from a discriminator.
Please improve your response in this round by following the advice from the previous round's discriminator (GAN style).
</context>

<task>
- I will provide you with the following information:
  1. **The persona background you should adopt (system prompt)**
  2. **A conversation history**
  3. **The latest user message in this round**
  4. **Feedback from the discriminator about your previous response**

- Please refer to:
  - The persona’s style settings
  - The conversation context
  - The discriminator's suggestions (to help you improve iteratively)

- Your tasks are:
  0. First, reflect on the differences between the discriminator’s advice and your previous response to identify areas for improvement
  1. Generate a new reply that aligns with the given persona’s style
</task>

<input>
**Persona background you should follow**: {system_message}  
**Conversation history**: {conversation_list}  
**User's latest message to respond to**: {last_user_message}  
**Discriminator's feedback**: {discriminator_advice}
</input>

<output>
Please directly output a persona-style response. Do not include any explanation, labels, or introductory text.
</output>
"""

gan_generator_1 = """
   <context>
   你是一個模型，負責根據指定的 persona 產生回覆。  
   你的任務是生成符合特定角色風格的回應，並根據判別器的意見持續改進。
   請根據上一輪判別器的建議來持續改進你這一輪的回覆 (GAN style)。
   </context>

   <task>
   - 我會提供你以下資訊：
   1. **你應該扮演的 persona 背景（system prompt）**
   2. **一段對話紀錄**
   3. **此輪對話中，user 的最新訊息**
   4. **判別器對你上一輪回覆的建議**

   - 請你參考：
   - persona 的風格設定
   - 對話歷史情境
   - 判別器的回饋建議（以便逐步改進回覆）

   - 請你完成以下任務：
   0. 先反思判別器建議與你上一輪回覆的差異，找出需要改進的地方
   1. 產生一個符合 persona 風格的新回覆
   </task>

   <input>
   **你應該扮演的 persona 背景**: {system_message}  
   **對話紀錄**: {conversation_list}  
   **你要回覆的 user 的最新訊息**: {last_user_message}
   **判別器的建議**: {discriminator_advice}
   </input>

   <output>
   請直接輸出符合 persona 風格的回覆，不要包含任何解釋、標記或前言。
   </output>
"""
