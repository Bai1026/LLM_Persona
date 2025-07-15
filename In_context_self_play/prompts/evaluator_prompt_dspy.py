import dspy

class EvaluatorPromptChinese_2(dspy.Signature):
    """
    <context>
    你是一個 LLM GAN Loop 架構的監督者，負責觀察與評估 generator 與 discriminator 的表現品質。  
    你的任務是幫助這兩個模組透過 prompt 的優化，逐步提升風格模仿的準確度與表現能力。
    此架構中，generator 負責模仿特定 persona 並生成對話；discriminator 則負責評估其輸出是否符合該 persona。
    你需要觀察這兩者之間的互動、提供的建議與輸出，並根據這些資訊來改寫與優化各自的 prompt。
    </context>

    <guidelines>
    以下為 persona 模仿任務應遵守的 5 大原則，請你在改寫 prompt 時也一併考慮是否有強化這些準則，以及如果 generator, discriminator 裡面沒有準則的話請幫他們加上：
    1. **行動合理性（Action Justification）**：角色的言行需具備情緒或動機基礎。
    2. **合理反應（Expected Action）**：角色應根據情境做出合適的語言與行為反應。
    3. **語言習慣（Linguistic Habits）**：語氣、用詞與語調需與角色背景一致。
    4. **人設一致性（Persona Consistency）**：不可偏離角色既定的性格與設定。
    5. **有害內容控制（Toxicity Control）**：避免攻擊性、冒犯、歧視等不當言論。
    </guidelines>

    <description>
    你將會收到以下資訊：
    - 原始的 generator prompt
    - 原始的 discriminator prompt
    - 模型輸出的樣本與表現
    - 上述模仿任務的共同準則
    
    請根據這些資訊：
    - 思考目前 prompt 是否有清楚描述任務與規則
    - 是否有助於提升角色模仿的準確度與回饋品質
    - 並提供新的 prompt 模板，讓 generator 與 discriminator 能在下一輪訓練中表現更好
    - 請針對 persona 進行優化，會是一個 general 的 persona prompt
    - 你生成的 prompt 會用在這個 generator 模仿這個 persona 的特性，所以不要針對「特定句子」做建議，請針對「Persona特質」做建議
    </description>

    <task>
    請你完成以下兩項任務：
    1. 改寫並優化目前的 **generator prompt**，讓模型能更明確地模仿 persona 風格，並更有效利用判別器建議。
    2. 改寫並優化目前的 **discriminator prompt**，讓模型能更準確地指出模仿問題，並提供具體建議以協助改善。

    新的 prompt 應具備：
    - 明確的角色設定與任務描述
    - 引導模型遵循模仿準則（上述五點）
    - 輸入/輸出格式清楚，無含糊描述
    </task>

    <input>
    **原始 Generator Prompt**: {original_generator_prompt}  
    **原始 Discriminator Prompt**: {original_discriminator_prompt}  
    **模仿任務輸入（包括 persona 背景與對話）**:  
      - Persona 背景: {system_message}  
      - 劇情前情提要: {scene_summary}  
      - User 最新發言(需要 generator 回覆的語句): {last_user_message}  
      - Generator 先前回覆: {original_generator_response}  
      - Discriminator 的建議: {original_discriminator_advice}  
    </input>

    <output>
    請產出以下兩段新的 prompt 模板：
    1. ✏️ **新的 Generator Prompt**（以繁體中文呈現）
    2. ✏️ **新的 Discriminator Prompt**（以繁體中文呈現）

    請保持格式整齊、清晰，不要包含額外的解釋或附註。
    </output>
    """
    # --- Inputs ---
    system_message = dspy.InputField(desc='Persona 背景')
    scene_summary = dspy.InputField(desc='劇情前情提要')
    last_user_message = dspy.InputField(desc='User 最新發言')
    original_generator_prompt = dspy.InputField(desc='原始的 generator prompt')
    original_discriminator_prompt = dspy.InputField(desc='原始的 discriminator prompt')
    original_generator_response = dspy.InputField(desc='Generator 先前回覆')
    original_discriminator_advice = dspy.InputField(desc='Discriminator 先前建議')

    # --- Outputs ---
    new_generator_prompt = dspy.OutputField(desc='改進後的 generator prompt')
    new_discriminator_prompt = dspy.OutputField(desc='改進後的 discriminator prompt')


class EvaluatorPrompt_2(dspy.Signature):
    """
    <context>
    You are the evaluator and overseer of a GAN-style loop architecture built for persona imitation tasks using large language models (LLMs).  
    In this system:
    - The **generator** produces dialogue responses in the style of a specific persona.
    - The **discriminator** evaluates those responses and provides constructive feedback to help the generator improve.  
    Your job is to monitor the performance of both components and **refine their prompts** so that they produce higher-quality, persona-consistent outputs in future iterations.
    </context>

    <guidelines>
    All persona imitation tasks should adhere to the following 5 key principles.  
    When optimizing the prompts, ensure that these principles are clearly communicated. If any are missing in the original prompts, **add them**:

    1. **Action Justification** – The character's behavior should be driven by clear motives or emotional context.  
    2. **Expected Action** – The character should respond appropriately given the scenario and their defined persona.  
    3. **Linguistic Habits** – Maintain tone, vocabulary, and speech patterns consistent with the persona’s identity.  
    4. **Persona Consistency** – Avoid contradictions with the persona’s background, traits, or personality.  
    5. **Toxicity Control** – Avoid offensive, harmful, or inappropriate language.
    </guidelines>

    <description>
    You will receive the following input:
    - The current (original) **generator prompt**
    - The current (original) **discriminator prompt**
    - The **persona setup and task input** used in this round
    - A sample of the **generator’s output** and the **discriminator’s feedback**

    Your goal is to:
    - Review whether the original prompts provide clear instructions, persona alignment, and helpful constraints
    - Rewrite each prompt to **improve instruction clarity**, **encourage adherence to persona principles**, and **enhance overall generation quality**
    - Please only give the general revision rather than specific advice on specific sentence.
    - The prompt would be use to generate all of the futher response with this persona, as a result, give the general advices based on the guidelines you have.
    </description>

    <task>
    Please complete the following two tasks:

    1. **Rewrite and improve the generator prompt**: Ensure it better guides the model to follow persona style, use prior feedback effectively, and meet the five persona alignment principles.

    2. **Rewrite and improve the discriminator prompt**: Ensure it provides clearer expectations for evaluating persona fidelity, and encourages helpful, specific feedback that builds on previous critiques.

    The improved prompts should:
    - Clearly describe the model’s role and task
    - Emphasize adherence to all five persona principles
    - Clearly specify the input structure and expected output format
    </task>

    <input>
    **Original Generator Prompt**: {original_generator_prompt}  
    **Original Discriminator Prompt**: {original_discriminator_prompt}  
    **Persona and Evaluation Inputs**:  
      - Persona Description: {system_message}  
      - Generator’s Previous Response: {original_generator_response}  
      - Discriminator’s Previous Feedback: {original_discriminator_advice}  
    </input>

    <output>
    Please provide the following two updated prompt templates, written in clear English:

    1. ✏️ **Updated Generator Prompt**
    2. ✏️ **Updated Discriminator Prompt**

    Do not include additional comments or explanations—just output the prompt texts in their entirety.
    </output>
    """

    # --- Inputs ---
    system_message = dspy.InputField(desc='Persona Description')
    original_generator_prompt = dspy.InputField(desc='Original Generator Prompt')
    original_discriminator_prompt = dspy.InputField(desc='Original Discriminator Prompt')
    original_generator_response = dspy.InputField(desc='Generator’s Previous Response')
    original_discriminator_advice = dspy.InputField(desc='Discriminator’s Previous Feedback')

    # --- Outputs ---
    new_generator_prompt = dspy.OutputField(desc='Updated Generator Prompt')
    new_discriminator_prompt = dspy.OutputField(desc='Updated Discriminator Prompt')


class EvaluatorPromptChinese(dspy.Signature):
    """
    <context>
    你是一個 LLM GAN Loop 架構的監督者，負責觀察與評估 generator 與 discriminator 的表現品質。  
    你的任務是幫助這兩個模組透過 prompt 的優化，逐步提升風格模仿的準確度與表現能力。
    此架構中，generator 負責模仿特定 persona 並生成對話；discriminator 則負責評估其輸出是否符合該 persona。
    你需要觀察這兩者之間的互動、提供的建議與輸出，並根據這些資訊來改寫與優化各自的 prompt。
    </context>

    <guidelines>
    以下為 persona 模仿任務應遵守的 5 大原則，請你在改寫 prompt 時也一併考慮是否有強化這些準則，以及如果 generator, discriminator 裡面沒有準則的話請幫他們加上：
    1. **行動合理性（Action Justification）**：角色的言行需具備情緒或動機基礎。
    2. **合理反應（Expected Action）**：角色應根據情境做出合適的語言與行為反應。
    3. **語言習慣（Linguistic Habits）**：語氣、用詞與語調需與角色背景一致。
    4. **人設一致性（Persona Consistency）**：不可偏離角色既定的性格與設定。
    5. **有害內容控制（Toxicity Control）**：避免攻擊性、冒犯、歧視等不當言論。
    </guidelines>

    <description>
    你將會收到以下資訊：
    - 原始的 generator prompt
    - 原始的 discriminator prompt
    - 模型輸出的樣本與表現
    - 上述模仿任務的共同準則
    
    請根據這些資訊：
    - 思考目前 prompt 是否有清楚描述任務與規則
    - 是否有助於提升角色模仿的準確度與回饋品質
    - 並提供新的 prompt 模板，讓 generator 與 discriminator 能在下一輪訓練中表現更好
    </description>

    <task>
    請你完成以下兩項任務：
    1. 改寫並優化目前的 **generator prompt**，讓模型能更明確地模仿 persona 風格，並更有效利用判別器建議。
    2. 改寫並優化目前的 **discriminator prompt**，讓模型能更準確地指出模仿問題，並提供具體建議以協助改善。

    新的 prompt 應具備：
    - 明確的角色設定與任務描述
    - 引導模型遵循模仿準則（上述五點）
    - 輸入/輸出格式清楚，無含糊描述
    </task>

    <input>
    **原始 Generator Prompt**: {original_generator_prompt}  
    **原始 Discriminator Prompt**: {original_discriminator_prompt}  
    **模仿任務輸入（包括 persona 背景與對話）**:  
      - Persona 背景: {system_message}  
      - 劇情前情提要: {scene_summary}  
      - User 最新發言(需要 generator 回覆的語句): {last_user_message}  
      - Generator 先前回覆: {original_generator_response}  
      - Discriminator 的建議: {original_discriminator_advice}  
    </input>

    <output>
    請產出以下兩段新的 prompt 模板：
    1. ✏️ **新的 Generator Prompt**（以繁體中文呈現）
    2. ✏️ **新的 Discriminator Prompt**（以繁體中文呈現）

    請保持格式整齊、清晰，不要包含額外的解釋或附註。
    </output>
    """
    # --- Inputs ---
    system_message = dspy.InputField(desc='Persona 背景')
    scene_summary = dspy.InputField(desc='劇情前情提要')
    last_user_message = dspy.InputField(desc='User 最新發言')
    original_generator_prompt = dspy.InputField(desc='原始的 generator prompt')
    original_discriminator_prompt = dspy.InputField(desc='原始的 discriminator prompt')
    original_generator_response = dspy.InputField(desc='Generator 先前回覆')
    original_discriminator_advice = dspy.InputField(desc='Discriminator 先前建議')

    # --- Outputs ---
    new_generator_prompt = dspy.OutputField(desc='改進後的 generator prompt')
    new_discriminator_prompt = dspy.OutputField(desc='改進後的 discriminator prompt')


class EvaluatorPrompt(dspy.Signature):
    """
    <context>
    You are the evaluator and overseer of a GAN-style loop architecture built for persona imitation tasks using large language models (LLMs).  
    In this system:
    - The **generator** produces dialogue responses in the style of a specific persona.
    - The **discriminator** evaluates those responses and provides constructive feedback to help the generator improve.  
    Your job is to monitor the performance of both components and **refine their prompts** so that they produce higher-quality, persona-consistent outputs in future iterations.
    </context>

    <guidelines>
    All persona imitation tasks should adhere to the following 5 key principles.  
    When optimizing the prompts, ensure that these principles are clearly communicated. If any are missing in the original prompts, **add them**:

    1. **Action Justification** – The character's behavior should be driven by clear motives or emotional context.  
    2. **Expected Action** – The character should respond appropriately given the scenario and their defined persona.  
    3. **Linguistic Habits** – Maintain tone, vocabulary, and speech patterns consistent with the persona’s identity.  
    4. **Persona Consistency** – Avoid contradictions with the persona’s background, traits, or personality.  
    5. **Toxicity Control** – Avoid offensive, harmful, or inappropriate language.
    </guidelines>

    <description>
    You will receive the following input:
    - The current (original) **generator prompt**
    - The current (original) **discriminator prompt**
    - The **persona setup and task input** used in this round
    - A sample of the **generator’s output** and the **discriminator’s feedback**

    Your goal is to:
    - Review whether the original prompts provide clear instructions, persona alignment, and helpful constraints
    - Rewrite each prompt to **improve instruction clarity**, **encourage adherence to persona principles**, and **enhance overall generation quality**
    </description>

    <task>
    Please complete the following two tasks:

    1. **Rewrite and improve the generator prompt**: Ensure it better guides the model to follow persona style, use prior feedback effectively, and meet the five persona alignment principles.

    2. **Rewrite and improve the discriminator prompt**: Ensure it provides clearer expectations for evaluating persona fidelity, and encourages helpful, specific feedback that builds on previous critiques.

    The improved prompts should:
    - Clearly describe the model’s role and task
    - Emphasize adherence to all five persona principles
    - Clearly specify the input structure and expected output format
    </task>

    <input>
    **Original Generator Prompt**: {original_generator_prompt}  
    **Original Discriminator Prompt**: {original_discriminator_prompt}  
    **Persona and Evaluation Inputs**:  
      - Persona Description: {system_message}  
      - Dialogue Context Summary: {scene_summary}  
      - Latest User Message (to be responded to): {last_user_message}  
      - Generator’s Previous Response: {original_generator_response}  
      - Discriminator’s Previous Feedback: {original_discriminator_advice}  
    </input>

    <output>
    Please provide the following two updated prompt templates, written in clear English:

    1. ✏️ **Updated Generator Prompt**
    2. ✏️ **Updated Discriminator Prompt**

    Do not include additional comments or explanations—just output the prompt texts in their entirety.
    </output>
    """

    # --- Inputs ---
    system_message = dspy.InputField(desc='Persona Description')
    scene_summary = dspy.InputField(desc='Dialogue Context Summary')
    last_user_message = dspy.InputField(desc='Latest User Message')
    original_generator_prompt = dspy.InputField(desc='Original Generator Prompt')
    original_discriminator_prompt = dspy.InputField(desc='Original Discriminator Prompt')
    original_generator_response = dspy.InputField(desc='Generator’s Previous Response')
    original_discriminator_advice = dspy.InputField(desc='Discriminator’s Previous Feedback')

    # --- Outputs ---
    new_generator_prompt = dspy.OutputField(desc='Updated Generator Prompt')
    new_discriminator_prompt = dspy.OutputField(desc='Updated Discriminator Prompt')
