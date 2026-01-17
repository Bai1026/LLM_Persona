from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
from transformers import pipeline
import gc
import asyncio
import os

# 在最開始就設置環境變數，防止 torch.compile 警告
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_LOGS"] = ""  # 關閉 recompile 日誌

app = FastAPI(title="Gemma Text Generation API")

# 請求模型
class ChatRequest(BaseModel):
    # messages: List[Dict[str, Any]]
    messages: str
    max_new_tokens: Optional[int] = 8192

class ChatResponse(BaseModel):
    success: bool
    result: Optional[List[Dict[str, Any]]] = None
    response: Optional[str] = None
    input_messages_count: Optional[int] = None
    error: Optional[str] = None

# 全域變數存放模型
text_pipeline = None

@app.on_event("startup")
async def load_model():
    """應用啟動時載入模型"""
    global text_pipeline
    
    print("Loading model...")
    
    # 設置 torch 參數
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = False  # 避免動態優化
    torch.backends.cudnn.deterministic = True
    
    text_pipeline = pipeline(
        task="text-generation",
        model="google/gemma-3-4b-it",
        device="cuda",
        torch_dtype=torch.bfloat16,
        model_kwargs={
            # "torch_dtype": torch.bfloat16,
            "attn_implementation": "eager",  # 使用 eager attention 避免編譯問題
            # "use_cache": True,
        }
    )
    
    # 如果可能的話，禁用模型的編譯
    try:
        if hasattr(text_pipeline.model, 'forward'):
            text_pipeline.model.forward = torch.compiler.disable(text_pipeline.model.forward)
    except:
        pass  # 如果禁用失敗就忽略
    
    print(f"GPU Memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print("API ready!")

@app.post("/chat", response_model=ChatResponse)
async def generate_text(request: ChatRequest):
    """生成文本的主要端點"""
    try:
        if text_pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet")
        
        # 驗證輸入
        if not request.messages:
            raise HTTPException(status_code=400, detail="Messages cannot be empty")
        
        messages = request.messages
        # print(f"\n📣:\n{messages}\n")
        
        # 在執行緒池中運行模型推理（避免阻塞事件循環）
        result = await asyncio.get_event_loop().run_in_executor(
            None, 
            run_inference,
            messages, 
            request.max_new_tokens
        )
        
        print(f"\n💰:\n{result}\n")
        
        # 解析回應
        assistant_response = None
        if result is not None:
            try:
                assistant_response = result[0]['generated_text'][1]['content']
                if isinstance(assistant_response, str):
                    # print(f"\n🤖 (is str):\n{assistant_response}\n")
                    pass
                else:
                    assistant_response = assistant_response[0]["text"]
                    # print(f"\n🤖 (is dict):\n{assistant_response}\n")
            except (IndexError, KeyError, TypeError) as e:
                print(f"\n解析回應時出錯: {e}\n")
                assistant_response = None
        else:
            print(f"\n📣:\n{messages}\n")
            print(f"\n🤖: None\n")
        
        return ChatResponse(
            success=True,
            result=result,
            response=assistant_response,
            input_messages_count=len(messages)
        )
        
    except Exception as e:
        # 清理記憶體
        torch.cuda.empty_cache()
        gc.collect()
        
        return ChatResponse(
            success=False,
            error=str(e)
        )

# def run_inference(messages: List[Dict], max_tokens: int):
def run_inference(messages: str, max_tokens: int):
    """在同步函數中執行模型推理"""

    # 強制清除所有緩存
    # if hasattr(text_pipeline.model, 'past_key_values'):
    #     text_pipeline.model.past_key_values = None
    
    # 清除 CUDA 緩存
    torch.cuda.empty_cache()

    try:
        # 設置生成參數避免動態編譯
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "pad_token_id": text_pipeline.tokenizer.eos_token_id,
            "use_cache": False,
        }
        
        with torch.no_grad():
            # 嘗試強制禁用編譯
            try:
                with torch.compiler.disable():
                    result = text_pipeline(messages, **generation_kwargs)
            except:
                # 如果 torch.compiler.disable 不可用，直接執行
                result = text_pipeline(messages, **generation_kwargs)
    
        torch.cuda.empty_cache()
        gc.collect()
        
        return result
        
    except Exception as e:
        print(f"推理錯誤: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        return None

@app.get("/memory_status")
async def memory_status():
    """檢查 GPU 記憶體狀態"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    return {
        "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        "gpu_memory_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
    }

@app.get("/health")
async def health_check():
    """健康檢查"""
    return {
        "status": "healthy",
        "model_loaded": text_pipeline is not None,
        "cuda_available": torch.cuda.is_available()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,  # 直接傳入 app 物件
        host="127.0.0.1", 
        port=8002,
        workers=1,  # 重要：只用一個 worker 避免多次載入模型
        access_log=False,  # 關閉 access log 提升效能
        loop="asyncio"  # 使用標準 asyncio 更穩定
    )