from rich import print
import requests
import time
from typing import Dict, List
from utils import config, fix_repeation
from openai import OpenAI
import copy
import json

from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS  
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

from utils import logger, config, colored_print

ENVIRONMENT = "Environment"
NSP = "NSP"

special_characters = [ENVIRONMENT, NSP]

class Embedding(Embeddings):
    """A class that provides text embedding functionality using OpenAI library.
    
    This class implements the Embeddings interface and provides methods to:
    - Initialize an OpenAI client with proper credentials
    - Embed single pieces of text (queries)
    - Batch embed multiple texts (documents)
    - Handle errors and edge cases during the embedding process
    
    Attributes:
        client (OpenAI): OpenAI client instance for making API calls
        model (str): Name of the embedding model to use
        embedding_ctx_length (int): Maximum context length for embeddings
    """

    def __init__(self):
        # Initialize OpenAI client with credentials from config
        self.client = OpenAI(
            api_key=config['embedding_api_key'],
            base_url=config['embedding_base_url']
        )
        # Set the embedding model, default is typically "eval-BAAI-bge-m3-embedding"
        self.model = config['embedding_model']
        # Maximum tokens that can be embedded in one request
        self.embedding_ctx_length = 8192  

    def _embed(self, text: str) -> List[float]:
        """Internal method to embed a single piece of text.

        Args:
            text (str): The text to embed

        Returns:
            List[float]: The embedding vector

        Raises:
            ValueError: If input is not a string
            Exception: If embedding fails
        """
        if not isinstance(text, str):
            raise ValueError(f"Expected string input, got {type(text)}")
        
        # Replace newlines with spaces for cleaner input
        text = text.replace("\n", " ")
        try:
            embedding = self.client.embeddings.create(
                input=[text],
                model=self.model
            )
            return embedding.data[0].embedding
        except Exception as e:
            print(f"Error during embedding: {e}")
            print(f"Problematic text: {text[:100]}...")  
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a query text using the embedding model.

        Args:
            text (str): Query text to embed

        Returns:
            List[float]: Embedding vector for the query
        """
        return self._embed(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch embed multiple documents using the embedding model.

        Args:
            texts (List[str]): List of texts to embed

        Returns:
            List[List[float]]: List of embedding vectors

        Raises:
            Exception: If batch embedding fails
        """
        embeddings = []
        # Process texts in batches of 100 to avoid rate limits
        for i in range(0, len(texts), 100):  
            batch = texts[i:i+100]
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                embeddings.extend([data.embedding for data in response.data])
            except Exception as e:
                print(f"Error during batch embedding: {e}")
                raise
        return embeddings

def build_rag_corpus(character, database, target):
    """Build a retrieval-augmented generation (RAG) corpus from a character database.
    
    This function creates different types of document collections (book content, summaries, conversations)
    and builds vector stores for retrieval. It supports different retrieval targets and configurations.

    Args:
        character (str): Name of the character to build corpus for
        database (dict): Database containing character information, plots, conversations etc.
                        Should have 'detailed_plots' key with plot information.
        target (str): Type of corpus to build. Options:
            - 'raw_text': Only book content
            - 'expr1': Character and plot summaries
            - 'expr3': Summaries with k=3 retrieval
            - 'conv1': Only conversations
            - 'expr3_conv1': Mix of summaries and conversations
            - 'expr10_conv1': Expanded summaries with conversations

    Returns:
        dict: Mapping of corpus types to their respective retrievers.
              Returns None if database is None.
    """
    # Return None if no database provided
    if database is None:
        return None

    # Initialize empty corpora lists
    book_corpus = []
    experience_corpus = []
    conversation_corpus = [] 
    
    # Iterate through plots to build different corpora
    for plot in database['detailed_plots']:
        # Add full plot text to book corpus
        # Each plot is ~1000-3000 tokens, so no splitting needed
        book_corpus.append(plot['text'])

        # Build character-specific experience
        character_experience = { _['name']: _['experience'] for _ in  plot['key_characters']}.get(character, '')
        if character_experience:
            character_experience = f"{character}'s role: " + character_experience
        experience_corpus.append('PLOT: ' + plot['summary'] + '\n' + character_experience)

        # Process conversations, removing internal metadata
        conversation_info = {"summary": plot['summary'], "conversation": copy.deepcopy(plot['conversation'])}
        for conversation in conversation_info['conversation']:
            # Remove internal tracking fields from character info
            for character_info in conversation['key_characters']:
                character_info.pop("i_p", None)
                character_info.pop("i_c", None)
            
            # Remove internal tracking fields from dialogues
            for dialogue in conversation['dialogues']:
                dialogue.pop("i_p", None)
                dialogue.pop("i_c", None)
                dialogue.pop("i_u", None)

            # Convert conversation to string format with background context
            from utils import conversation_to_str
            try:
                conversation_corpus.append(conversation_to_str(
                    conversation=conversation['dialogues'], 
                    background={
                        'Plot Background': plot['summary'], 
                        'Scenario': conversation['scenario'], 
                        'topic': conversation['topic']
                    }
                ))
            except Exception as e:
                # Log any conversion errors
                from utils import setup_logger
                logger.error(f'Error in conversation_to_str: {e}')
                logger.error(f'Conversation: {conversation}')

    # Define corpus configurations for different retrieval targets
    # Format: (corpus, num_results, corpus_type)
    corpus_map = {
        'raw_text': [(book_corpus, 1, 'book')],
        'expr1': [(experience_corpus, 1, 'experience')],
        'expr3': [(experience_corpus, 3, 'experience')],
        'conv1': [(conversation_corpus, 1, 'conversation')],
        'expr3_conv1': [(experience_corpus, 3, 'experience'), (conversation_corpus, 1, 'conversation')],
        'expr10_conv1': [(experience_corpus, 10, 'experience'), (conversation_corpus, 1, 'conversation')]
    }

    corpora = corpus_map[target]
    retriever = {}

    # Process each corpus configuration
    for (corpus, k, target_type) in corpora:
        # Create document objects
        documents = [Document(page_content=doc) for doc in corpus]

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)

        # Add index metadata to documents
        for i, doc in enumerate(split_docs):
            doc.metadata['idx'] = i

        # Initialize embedding model
        custom_embed_model = Embedding()

        # Create vector store with error handling
        try:
            # Try creating vectorstore all at once
            vectorstore = FAISS.from_documents(split_docs, custom_embed_model)
        except Exception as e:
            print(f"Cannot create vectorstore at once: {e}; will try again.")
            try:
                # Fallback: Create incrementally
                vectorstore = FAISS.from_documents([split_docs[0]], custom_embed_model)
                for doc in split_docs[1:]:
                    vectorstore.add_documents([doc])
                print('Successfully created vectorstore')
            except:
                continue

        # Configure retriever based on k value
        if k != -1:
            # Standard k-nearest neighbor retriever
            retriever[target_type] = vectorstore.as_retriever(search_kwargs={"k": k})
        else:
            # Sequential retriever that returns all documents
            class SequentialRetriever:
                def __init__(self, docs):
                    self.docs = docs
                def invoke(self, query):
                    return self.docs

            # Store original documents and create retriever
            retriever[target_type] = SequentialRetriever(split_docs)

            #retriever[target_type] = vectorstore.as_retriever(search_kwargs={"k": len(split_docs)})
        

    return retriever

def rag(contexts, retriever, target_type):
    """
    Retrieves and formats relevant information from a database based on input contexts.
    
    Args:
        contexts (List[Dict]): List of message dictionaries containing conversation context
        retriever: Document retriever object with invoke() method to fetch relevant docs
        target_type (str): Type of content to retrieve - 'book', 'experience', or 'conversation'
    
    Returns:
        str: Formatted string containing retrieved information with appropriate headers
    """
    # Define headers and titles for different content types
    title_header = {
        'book': '====Book Content====\n\n',
        'experience': '====Historical Experience====\n\n', 
        'conversation': '====Historical Conversation====\n\n'
    }[target_type]
    
    title = {
        'book': 'Content',
        'experience': 'Historical Experience',
        'conversation': 'Historical Conversation'
    }[target_type]

    # Combine context messages into a single query string
    query = "\n\n".join([msg["content"] for msg in contexts])
    
    # Retrieve relevant documents using the retriever
    retrieved_docs = retriever.invoke(query)

    # Sort documents by index if metadata is available
    if retrieved_docs and 'idx' in retrieved_docs[0].metadata:
        retrieved_docs = sorted(retrieved_docs, key=lambda x: x.metadata['idx'])

    # Format the retrieved information
    if len(retrieved_docs) > 1:
        # For multiple documents, include numbered sections
        relevant_info = title_header + ''
        for i, doc in enumerate(retrieved_docs):
            relevant_info +=  f'{title} {i+1}\n' + doc.page_content + '\n\n'
    else:
        # For single document, simple concatenation
        relevant_info = title_header + "\n\n".join([doc.page_content for doc in retrieved_docs])

    return relevant_info

class Agent:
    """
    對話代理類別，管理使用語言模型的對話互動。
    
    此代理可以：
    - 參與對話同時維護上下文
    - 從知識資料庫檢索相關資訊  
    - 根據配置產生適當回應
    - 自動偵測模型名稱並切換到 API 模式

    屬性:
        model (str): 用於產生回應的語言模型
        name (str): 代理/角色名稱
        database (Dict): 用於檢索的知識資料庫
        scene (Dict): 場景上下文資訊
        system_prompt (str): 引導代理行為的初始系統提示
        retrievers (Dict): 不同內容類型的 RAG 檢索器
        system_role (str): 系統訊息的角色類型 ('user' 或 'system')
        messages (List): 對話歷史記錄，為訊息字典的列表
    """
    def __init__(self, model: str, name, database: Dict, system_prompt: str = None, scene: Dict = None, retrieval_target: str = 'conversation'):
        # 初始化基本代理屬性
        self.model = model 
        self.name = name 
        self.database = database
        self.scene = scene

        self.system_prompt = system_prompt 
        
        # 清理系統提示詞，移除尾隨換行符
        self.system_prompt = self.system_prompt.strip('\n')
        
        # 如果提供資料庫，設定 RAG 檢索器
        if retrieval_target and database:
            self.retrievers = build_rag_corpus(name, database, retrieval_target)
        else:
            self.retrievers = None

        # 為非特殊角色新增簡潔性指令
        if self.name not in special_characters:
            self.system_prompt = self.system_prompt + '\n\n像人類一樣簡潔地說話，而不是冗長。將您的回應限制在 60 個字以內。\n\n'

        # 為特定模型新增名稱前綴指令
        if self.model.startswith('llama') or self.model.startswith('step'):
            self.system_prompt = self.system_prompt + f'以 "{name}: " 開始您的回應。避免代表其他角色發言。\n\n'
        
        # 根據模型類型設定適當的系統角色
        if self.model.startswith('claude') or self.model.startswith('o1'):
            self.system_role = 'user'
        else:
            self.system_role = 'system'

        # 使用系統提示初始化對話歷史記錄
        self.messages = [{"role": self.system_role, "content": self.system_prompt}]
        colored_print('blue', f"messages: {self.messages}")

    def _is_api_model(self) -> bool:
        """
        檢查模型是否為 API 模型
        
        Returns:
            bool: 如果是 API 模型回傳 True
        """
        # 修改這裡，讓所有模型都使用 API
        api_model_triggers = [
            'custom-api',           # 完全匹配
            'api-',                # 前綴匹配
            'local-',              # 前綴匹配
        ]
        
        # 檢查是否匹配任何 API 觸發條件
        for trigger in api_model_triggers:
            if self.model == trigger or self.model.startswith(trigger):
                return True
        return False

    def _get_api_config(self) -> dict:
        """
        根據模型名稱取得 API 配置
        
        Returns:
            dict: API 配置字典
        """
        # 預設配置
        config = {
            'endpoint': 'http://localhost:6969/v1/chat/completions',
            'headers': {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            'timeout': 300,
            'retry_times': 2,
            'retry_delay': 1.0
        }

        port = 6969
        config['endpoint'] = f'http://localhost:{port}/v1/chat/completions'
        
        return config

    def _call_api(self, messages, max_tokens=512, temperature=0.7) -> str:
        """
        透過 API 呼叫產生回應
        
        Args:
            messages: 對話訊息列表
            max_tokens (int): 最大產生 token 數量
            temperature (float): 產生溫度參數
            
        Returns:
            str: 產生的回應文字，錯誤時回傳空字串
        """
        api_config = self._get_api_config()
        
        for attempt in range(api_config['retry_times']):
            try:
                # 準備 API 請求資料
                payload = {
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                    "model": self.model  # 將原始模型名稱也傳送給 API
                }
                
                # 發送 API 請求
                logger.info(f"正在呼叫 API：{api_config['endpoint']} (嘗試 {attempt + 1}/{api_config['retry_times']})")
                colored_print('blue', f"正在呼叫 API：{api_config['endpoint']} (嘗試 {attempt + 1}/{api_config['retry_times']})")
                logger.info(f"請求資料：{json.dumps(payload, ensure_ascii=False, indent=2)}")
                # colored_print('blue', f"請求資料：{json.dumps(payload, ensure_ascii=False, indent=2)}")
                
                response = requests.post(
                    api_config['endpoint'],
                    headers=api_config['headers'],
                    json=payload,
                    timeout=api_config['timeout']
                )
                
                # 記錄原始回應內容以進行偵錯
                logger.info(f"API 回應狀態程式碼：{response.status_code}")
                colored_print('blue', f"API 回應狀態程式碼：{response.status_code}")
                logger.info(f"API 原始回應內容：{response.text[:500]}...")
                colored_print('blue', f"API 原始回應內容：{response.text[:500]}...")
                
                # 檢查 HTTP 狀態程式碼
                response.raise_for_status()
                
                # 檢查回應內容是否為空
                if not response.text.strip():
                    logger.error("API 回傳空內容")
                    colored_print('red', "❌ BUG: API 回傳空內容")
                    if attempt < api_config['retry_times'] - 1:
                        time.sleep(api_config['retry_delay'] * (attempt + 1))
                        continue
                    return "抱歉，我現在無法回應。"
                
                # 解析回應
                try:
                    response_data = response.json()
                except json.JSONDecodeError as e:
                    logger.error(f"JSON 解析錯誤：{e}")
                    colored_print('red', f"❌ EXCEPTION: JSON 解析錯誤：{e}")
                    logger.error(f"原始回應內容：{response.text}")
                    colored_print('red', f"❌ 原始回應內容：{response.text}")
                    # 如果不是 JSON 格式，嘗試直接使用文字回應
                    if response.text.strip():
                        logger.info("嘗試直接使用文字回應")
                        colored_print('red', "嘗試直接使用文字回應")
                        return response.text.strip()
                    if attempt < api_config['retry_times'] - 1:
                        time.sleep(api_config['retry_delay'] * (attempt + 1))
                        continue
                    return "抱歉，我現在無法回應。"
                
                # 提取產生的文字（支援多種 API 格式）
                generated_text = ""
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    # OpenAI 格式
                    generated_text = response_data["choices"][0]["message"]["content"]
                elif "response" in response_data:
                    # 自訂格式 1
                    generated_text = response_data["response"]
                elif "content" in response_data:
                    # 自訂格式 2
                    generated_text = response_data["content"]
                elif "text" in response_data:
                    # 自訂格式 3
                    generated_text = response_data["text"]
                else:
                    logger.error(f"未知的 API 回應格式：{response_data}")
                    colored_print('red', f"❌ BUG: 未知的 API 回應格式：{response_data}")
                    if attempt < api_config['retry_times'] - 1:
                        time.sleep(api_config['retry_delay'] * (attempt + 1))
                        continue
                    return "抱歉，我現在無法回應。"
                
                if not generated_text or not generated_text.strip():
                    logger.error("API 回傳空的產生內容")
                    colored_print('red', "❌ BUG: API 回傳空的產生內容")
                    if attempt < api_config['retry_times'] - 1:
                        time.sleep(api_config['retry_delay'] * (attempt + 1))
                        continue
                    return "抱歉，我現在無法回應。"
                
                logger.info(f"API 呼叫成功，產生 {len(generated_text)} 個字元")
                colored_print('blue', f"API 呼叫成功，產生 {len(generated_text)} 個字元")
                return generated_text.strip()
                
            except requests.exceptions.Timeout:
                logger.error(f"API 呼叫超時（{api_config['timeout']}秒）")
                colored_print('red', f"❌ EXCEPTION: API 呼叫超時（{api_config['timeout']}秒）")
            except requests.exceptions.RequestException as e:
                logger.error(f"API 呼叫錯誤：{e}")
                colored_print('red', f"❌ EXCEPTION: API 呼叫錯誤：{e}")
            except Exception as e:
                logger.error(f"未預期的錯誤：{e}")
                colored_print('red', f"❌ EXCEPTION: 未預期的錯誤：{e}")
                import traceback
                logger.error(f"完整錯誤追蹤：{traceback.format_exc()}")
                colored_print('red', f"❌ 完整錯誤追蹤：{traceback.format_exc()}")
            
            if attempt < api_config['retry_times'] - 1:
                logger.info(f"等待 {api_config['retry_delay'] * (attempt + 1)} 秒後重試...")
                colored_print('red', f"等待 {api_config['retry_delay'] * (attempt + 1)} 秒後重試...")
                time.sleep(api_config['retry_delay'] * (attempt + 1))
        
        logger.error(f"經過 {api_config['retry_times']} 次嘗試後 API 呼叫仍然失敗")
        colored_print('red', f"❌ BUG: 經過 {api_config['retry_times']} 次嘗試後 API 呼叫仍然失敗")
        return "抱歉，我現在無法回應。"

    def chat(self) -> str:
        """
        根據對話歷史記錄和可用知識產生回應。
        
        此方法：
        1. 如果可用，從知識資料庫檢索相關資訊
        2. 根據模型類型選擇呼叫方式：
        - 自訂 API 模型：使用本地端點
        - 其他模型：使用 OpenAI 官方 API
        3. 根據模型特定需求處理和清理回應
        
        Returns:
            str: 產生的回應文字，發生錯誤時回傳預設訊息
        """
        try:
            messages = self.messages
            if self.retrievers:
                # 從最近的上下文檢索相關資訊（最後 3 個訊息）
                contexts = self.messages[1:]
                contexts = contexts[-3:]

                # 從所有配置的檢索器收集知識
                knowledge = ''
                for target_type, retriever in self.retrievers.items():
                    knowledge += rag(contexts, retriever, target_type)

                # 將檢索到的知識插入系統提示
                messages = copy.deepcopy(self.messages)
                messages[0]['content'] = messages[0]['content'].replace('{retrieved_knowledge}', '<背景資訊開始>\n\n' + knowledge + '\n\n<背景資訊結束>\n\n')

            # 根據模型類型選擇呼叫方式
            if self._is_api_model():
                # 使用自訂 API（本地端點）
                logger.info(f"偵測到自訂 API 模型：{self.model}，使用本地 API 呼叫")
                colored_print('blue', f"偵測到自訂 API 模型：{self.model}，使用本地 API 呼叫")
                response = self._call_api(messages, max_tokens=512)
                
                # 檢查 API 回應是否有效
                if not response or response.strip() == "":
                    logger.error("自訂 API 回傳空回應")
                    colored_print('red', "❌ BUG: 自訂 API 回傳空回應")
                    return f"{self.name}: 抱歉，我現在無法回應。"
                    
            else:
                # 使用 OpenAI 官方 API
                logger.info(f"使用 OpenAI 官方 API：{self.model}")
                colored_print('blue', f"使用 OpenAI 官方 API：{self.model}")
                
                # 修正：根據 get_response 函式的實際簽名來呼叫
                try:
                    from utils import get_response_with_retry
                    response = get_response_with_retry(model=self.model, messages=messages, max_tokens=512)
                except TypeError as e:
                    colored_print('red', f"❌ EXCEPTION: get_response_with_retry TypeError：{e}")
                    if "multiple values for argument 'model'" in str(e):
                        # 如果出現重複參數錯誤，嘗試只傳遞 messages
                        try:
                            from utils import get_response
                            response = get_response(messages)
                        except Exception as fallback_error:
                            logger.error(f"get_response 備用呼叫失敗：{fallback_error}")
                            colored_print('red', f"❌ EXCEPTION: get_response 備用呼叫失敗：{fallback_error}")
                            return f"{self.name}: 抱歉，發生了技術問題。"
                    else:
                        colored_print('red', f"❌ EXCEPTION: 未處理的 TypeError：{e}")
                        raise e
                except Exception as e:
                    colored_print('red', f"❌ EXCEPTION: get_response_with_retry 其他錯誤：{e}")
                    return f"{self.name}: 抱歉，發生了技術問題。"
                
                colored_print('blue', f"OpenAI API response: {response}")
                
                if not response or response.strip() == "":
                    logger.error("OpenAI API 回傳空回應")
                    colored_print('red', "❌ BUG: OpenAI API 回傳空回應")
                    return f"{self.name}: 抱歉，我現在無法回應。"

            def parse_response(response: str, character_name: str) -> str:
                """
                從（意外的）多角色回應中提取特定角色的話語。
                
                Args:
                    response (str): 完整回應文字
                    character_name (str): 要提取話語的角色名稱
                
                Returns:
                    str: 指定角色的提取話語
                """
                try:
                    lines = response.split('\n')
                    current_character = None
                    current_utterance = ""
                    parsed_utterances = []

                    for line in lines:
                        # 檢查行開頭的角色名稱
                        if ':' in line:
                            character = line.split(':', 1)[0].strip()
                            
                            if current_character != character:
                                # 儲存前一個角色的話語並開始新的
                                if current_utterance:
                                    parsed_utterances.append((current_character, current_utterance))
                                current_character = character
                                current_utterance = ""

                        current_utterance += line + "\n"
                
                    # 儲存最終話語
                    if current_utterance:
                        parsed_utterances.append((current_character, current_utterance))
                    
                    # 尋找指定角色的話語
                    for character, utterance in parsed_utterances:
                        if character == character_name:
                            return utterance

                    # 如果沒找到指定角色，回傳原始回應
                    return response
                    
                except Exception as e:
                    colored_print('red', f"❌ EXCEPTION: parse_response 錯誤：{e}")
                    return response

            # 根據模型類型和代理名稱處理回應
            if (self.model.startswith('llama') or self.model.startswith('step')) and self.name != 'NSP':
                response = parse_response(response, self.name)
            
            # 修復某些模型的重複問題
            if not any(self.model.lower().startswith(model_type) for model_type in ['gpt', 'claude']) and self.name != 'NSP':
                try:
                    res = fix_repeation(response)
                    if res:
                        logger.info(f'{self.model} 發現並修復重複：原始："{response}" 修復後："{res}"')
                        colored_print('red', f'{self.model} 發現並修復重複：原始："{response}" 修復後："{res}"')
                        response = res
                except Exception as e:
                    colored_print('red', f"❌ EXCEPTION: fix_repeation 錯誤：{e}")

            # 確保回應有效
            if not response or response.strip() == "":
                logger.warning("最終回應為空，使用預設回應")
                colored_print('red', "❌ BUG: 最終回應為空，使用預設回應")
                response = f"{self.name}: 抱歉，我現在無法回應。"

            return response

        except Exception as e:
            import traceback
            logger.error(f"取得回應時發生錯誤：{e}")
            colored_print('red', f"❌ EXCEPTION: 取得回應時發生錯誤：{e}")
            logger.error(f"完整錯誤追蹤：{traceback.format_exc()}")
            colored_print('red', f"❌ 完整錯誤追蹤：{traceback.format_exc()}")
            
            return f"{self.name}: 抱歉，發生了技術問題。"
    
    
    def update(self, role: str, message: str):
        """
        Updates the conversation history with a new message.
        
        Args:
            role (str): Role of the message sender
            message (str): Content of the message
        """
        if message:
            # Append message to last message if same role, otherwise add new message
            if self.messages and self.messages[-1]['role'] == role:
                self.messages[-1]['content'] = self.messages[-1]['content'] + '\n\n' + message
            else:
                self.messages.append({"role": role, "content": message})

        return

    def reset(self):
        """
        Resets the conversation history to initial state with only system prompt.
        """
        self.messages = self.messages[:1]
    