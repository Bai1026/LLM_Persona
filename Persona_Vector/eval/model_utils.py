import os, re, json, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel
import socket





def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  
        return s.getsockname()[1]


# ---------- 工具 ----------
_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")

def _pick_latest_checkpoint(model_path: str) -> str:
    ckpts = [(int(m.group(1)), p) for p in Path(model_path).iterdir()
             if (m := _CHECKPOINT_RE.fullmatch(p.name)) and p.is_dir()]
    return str(max(ckpts, key=lambda x: x[0])[1]) if ckpts else model_path

def _is_lora(path: str) -> bool:
    return Path(path, "adapter_config.json").exists()

def _load_and_merge_lora(lora_path: str, dtype, device_map):
    cfg = PeftConfig.from_pretrained(lora_path)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name_or_path, torch_dtype=dtype, device_map=device_map
    )
    return PeftModel.from_pretrained(base, lora_path).merge_and_unload()

def _load_tokenizer(path_or_id: str):
    tok = AutoTokenizer.from_pretrained(path_or_id)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    
    # 為 Gemma 模型設定聊天模板
    if "gemma" in path_or_id.lower() and tok.chat_template is None:
        tok.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n{% elif message['role'] == 'assistant' %}<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
    
    return tok

def load_model(model_path: str, dtype=torch.bfloat16):
    # 檢查是否為 Gemma-3 模型，如果是則使用特殊處理
    if "gemma-3" in model_path.lower():
        return load_gemma3_model(model_path, dtype)
    
    if not os.path.exists(model_path):               # ---- Hub ----
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto"
        )
        tok = _load_tokenizer(model_path)
        return model, tok

    resolved = _pick_latest_checkpoint(model_path)
    print(f"loading {resolved}")
    if _is_lora(resolved):
        model = _load_and_merge_lora(resolved, dtype, "auto")
        tok = _load_tokenizer(model.config._name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved, torch_dtype=dtype, device_map="auto"
        )
        tok = _load_tokenizer(resolved)
    return model, tok

def load_gemma3_model(model_path: str, dtype=torch.bfloat16):
    """專門為 Gemma-3 模型載入，使用 transformers 而非 vLLM"""
    print(f"🎯 載入 Gemma-3 模型 (使用 transformers): {model_path}")
    
    # 載入分詞器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 載入模型，使用與 simple_gemma3_chat.py 相同的設定
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"  # 避免 FlashAttention2 問題
        )
        print(f"✅ Gemma-3 模型載入成功 - 設備: {next(model.parameters()).device}")
        
    except Exception as e:
        print(f"⚠️ 標準載入失敗，回退到 CPU: {e}")
        # 回退到 CPU 和 float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        print("✅ CPU 模式載入成功")
    
    return model, tokenizer

def load_vllm_model(model_path: str):
    from vllm import LLM

    if not os.path.exists(model_path):               # ---- Hub ----
        # 根據模型自動調整 max_model_len
        if "gemma" in model_path.lower():
            max_model_len = 8192
        elif "llama" in model_path.lower():
            max_model_len = 8192
        elif "qwen" in model_path.lower():
            max_model_len = 32768
        else:
            max_model_len = 8192  # 預設值
            
        llm = LLM(
            model=model_path,
            enable_prefix_caching=True,
            enable_lora=True,
            tensor_parallel_size=torch.cuda.device_count(),
            max_num_seqs=32,
            gpu_memory_utilization=0.9,
            max_model_len=max_model_len,
            max_lora_rank=128,
        )
        tok = llm.get_tokenizer()
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "left"
        
        # 為 Gemma 模型設定聊天模板
        if "gemma" in model_path.lower() and tok.chat_template is None:
            tok.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n{% elif message['role'] == 'assistant' %}<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
        
        return llm, tok, None

    # ---- 本地 ----
    resolved = _pick_latest_checkpoint(model_path)
    print(f"loading {resolved}")
    is_lora = _is_lora(resolved)

    base_path = (PeftConfig.from_pretrained(resolved).base_model_name_or_path
                 if is_lora else resolved)

    llm = LLM(
        model=base_path,
        enable_prefix_caching=True,
        enable_lora=True,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=32,
        gpu_memory_utilization=0.9,
        max_model_len=20000,
        max_lora_rank=128,
    )

    if is_lora:
        lora_path = resolved
    else:
        lora_path = None

    tok = llm.get_tokenizer()
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    
    # 為 Gemma 模型設定聊天模板
    if "gemma" in base_path.lower() and tok.chat_template is None:
        tok.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n{% elif message['role'] == 'assistant' %}<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
    
    return llm, tok, lora_path
