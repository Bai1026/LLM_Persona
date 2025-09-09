好的，專注於 CoSER 的用法。這個程式碼庫的核心是從小說等虛構作品中，透過大型語言模型（LLM）自動化地提取、擴充並建構高品質的角色扮演對話資料，然後用這些資料來訓練（SFT）更擅長角色扮演的模型。

以下解析如何使用 CoSER 來提取角色資訊，以及如何將您自己的 Agent 整合進來。

1. 了解與提取角色資訊
   CoSER 的資料處理流程產出的最終資料位於 final 目錄下，每個 JSON 檔案對應一本書。這些檔案包含了非常豐富的角色資訊。

資料結構
以 [CoSER/data/final/A Feast for Crows (A Song of Ice and Fire, #4).json](CoSER/data/final/A Feast for Crows (A Song of Ice and Fire, #4).json) 為例，其結構如下：

最外層是一個以角色名稱為鍵（key）的巨大 JSON 物件。
每個角色物件底下包含以下關鍵資訊：
profile: (字串) 一段由 LLM 生成的、關於該角色的詳細介紹，描述其背景、動機與性格。這是最完整的角色設定。
plots: (列表) 該角色參與的劇情摘要。每個 plot 包含 name, description, summary。
conversations: (列表) 該角色在特定情境下的內心獨白（thought）。
utterances: (列表) 該角色說過的每一句話（message）。這些對話通常包含了用 [] 包起來的內心想法，這是 CoSER 的一個特色，旨在讓模型學會思考。
如何提取
您可以撰寫一個簡單的 Python 腳本來讀取這些 JSON 檔案，並根據角色名稱遍歷，從而提取您需要的任何資訊（如 profile、utterances 等）。

範例程式碼片段：

2. 如何建構自己的資料集
   如果您想用自己的文本（例如，您自己的小說）來建立類似的資料集，可以遵循 CoSER 的資料建構流程。

準備來源內容: 將您的書籍或文本放到一個資料夾。
執行資料建構: 主要的腳本是 main.py。這個腳本會呼叫 LLM API（如 GPT-4）來執行多個步驟：
extract: 從文本中提取關鍵劇情、角色和對話。
assemble: 整理並豐富提取出的內容，例如生成更詳細的角色動機和場景描述。
transform: 將處理好的資料轉換為可用於模型訓練的格式（例如 ShareGPT 格式）。
您需要先在 config_template.json 中設定好您的 LLM API 金鑰，然後執行 main.py。詳細步驟可以參考 README.md 中的 "Constructing Your Own Datasets" 章節。

3. 如何與 CoSER 模型互動或評估
   CoSER 提供了兩種主要方式來使用其產出的模型或資料：

A. 直接對話
chat.py 是一個與模型直接對話的範例腳本。它展示了如何載入一個已經用 CoSER 資料訓練好的模型，並與其進行角色扮演互動。

根據 README.md 的建議，為了達到最好的效果，您應該在 system prompt 中提示模型使用內心想法，格式如下：

Use [your thought] for thoughts, which others can't see. Use (your action) for actions, which others can see.

B. 整合您的 Agent 進行評估
如果您想將自己的 Agent 與 CoSER 的方法進行比較，可以使用 gca_evaluation 中的評估框架。

建立相容的 API: 您的 Agent 需要提供一個與 OpenAI API 相容的 HTTP 端點。它應該接受一個包含 messages 列表的 POST 請求，並回傳一個包含模型回應的 JSON。

修改評估代理: 打開 agent.py。這個檔案中的 Agent 類別負責與不同的模型 API 溝通。您可以在 chat 函式中新增一個分支來處理對您自訂 API 的呼叫。

            raise ValueError(f"Unsupported provider: {self.provider}")

執行評估: 修改並執行 main.py。您可以在腳本中實例化一個指向您 Agent 的 Agent 物件，並將其加入評估流程中與其他模型（如 GPT-4 或 CoSER 模型）進行比較。

總結來說，CoSER 提供了一套完整的從資料建構到模型訓練和評估的工具鏈。您可以利用其豐富的資料集，也可以遵循其流程來建立自己的角色扮演資料。
