#!/usr/bin/env python3
"""
測試修補過的 eval_persona 與 Gemma-3 整合
"""

# 首先導入修補程式
from gemma3_patch import patch_gemma3_sampling

# 啟用 Gemma-3 支援
print("🔧 啟用 Gemma-3 支援...")
original_sample = patch_gemma3_sampling()

# 現在測試 eval_persona
import subprocess
import os
import pandas as pd

def test_patched_eval_persona():
    """測試修補過的 eval_persona"""
    
    print("🧪 測試修補過的 eval_persona...")
    
    # 建立測試輸出路徑
    output_path = "test_gemma3_patched_output.csv"
    
    # 刪除舊檔案
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # 建立指令
    cmd = [
        "python", "-c", """
# 導入修補程式
import sys
sys.path.append('.')
from gemma3_patch import patch_gemma3_sampling
patch_gemma3_sampling()

# 現在執行 eval_persona
import subprocess
result = subprocess.run([
    'python', '-m', 'eval.eval_persona',
    '--model', 'google/gemma-3-4b-it',
    '--trait', 'creative_professional', 
    '--output_path', 'test_gemma3_patched_output.csv',
    '--persona_instruction_type', 'pos',
    '--assistant_name', 'creative_professional',
    '--judge_model', 'gpt-4o-mini',
    '--version', 'extract'
])
"""
    ]
    
    print(f"🚀 執行修補測試...")
    
    # 設定環境變數
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = "/workspace/LLM_Persona/persona_vectors"
    
    try:
        result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        print("✅ 修補測試執行完成")
        
        if result.stdout:
            print("標準輸出:", result.stdout[-500:])  # 顯示最後 500 字元
        
        # 檢查結果
        if os.path.exists(output_path):
            print(f"📊 檢查結果檔案: {output_path}")
            df = pd.read_csv(output_path)
            
            if 'answer' in df.columns:
                non_empty = df[df['answer'].notna() & (df['answer'].str.strip() != '')]
                print(f"   非空回應比例: {len(non_empty)}/{len(df)} ({len(non_empty)/len(df)*100:.1f}%)")
                
                if len(non_empty) > 0:
                    print("   前 3 個回應範例:")
                    for i, row in non_empty.head(3).iterrows():
                        answer = str(row['answer'])[:100] + "..." if len(str(row['answer'])) > 100 else str(row['answer'])
                        print(f"     {i+1}: {answer}")
            
            if 'creative_professional' in df.columns:
                scores = df['creative_professional'].dropna()
                if len(scores) > 0:
                    print(f"   平均創意分數: {scores.mean():.4f}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 執行失敗: {e}")
        if e.stdout:
            print("標準輸出:", e.stdout)
        if e.stderr:
            print("錯誤輸出:", e.stderr)
        return False

def simple_direct_test():
    """直接測試修補功能"""
    
    print("\n🔬 直接測試修補功能...")
    
    try:
        # 模擬 eval_persona 的載入過程
        from eval.model_utils import load_vllm_model
        
        print("載入 Gemma-3 模型...")
        llm, tokenizer, lora_path = load_vllm_model("google/gemma-3-4b-it")
        
        print(f"模型載入成功，分詞器: {type(tokenizer).__name__}")
        
        # 測試聊天模板
        test_messages = [
            {"role": "user", "content": "What is creativity?"}
        ]
        
        try:
            prompt = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
            print(f"聊天模板測試成功:")
            print(f"  提示: {prompt}")
        except Exception as e:
            print(f"聊天模板測試失敗: {e}")
        
        # 測試修補過的 sample 函式
        from eval.eval_persona import sample
        
        print("測試修補過的 sample 函式...")
        conversations = [test_messages]
        
        try:
            texts, answers = sample(
                model=llm, 
                tokenizer=tokenizer, 
                conversations=conversations,
                max_tokens=50,
                temperature=0.7,
                min_tokens=5
            )
            
            print(f"產生測試完成:")
            print(f"  輸入: {texts[0][:100]}...")
            print(f"  輸出: '{answers[0]}'")
            
            if answers[0].strip():
                print("✅ 修補成功！模型有回應")
                return True
            else:
                print("⚠️ 回應仍然是空的")
                return False
                
        except Exception as e:
            print(f"sample 函式測試失敗: {e}")
            return False
            
    except Exception as e:
        print(f"直接測試失敗: {e}")
        return False

if __name__ == "__main__":
    print("🚀 開始 Gemma-3 修補測試...")
    
    # 先進行直接測試
    direct_success = simple_direct_test()
    
    if direct_success:
        print("\n" + "="*60)
        print("✅ 直接測試成功，進行完整測試...")
        test_patched_eval_persona()
    else:
        print("\n❌ 直接測試失敗，需要進一步調整修補程式")
