#!/usr/bin/env python3
"""
直接測試 Gemma-3 與 eval_persona 的整合，加入詳細偵錯
"""

import subprocess
import os
import pandas as pd

def test_eval_persona_gemma3():
    """測試 eval_persona 與 Gemma-3 的整合"""
    
    print("🧪 測試 eval_persona 與 Gemma-3 整合...")
    
    # 建立測試輸出路徑
    output_path = "test_gemma3_output.csv"
    
    # 刪除舊檔案
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # 建立最簡單的指令
    cmd = [
        "python", "-m", "eval.eval_persona",
        "--model", "google/gemma-3-4b-it",
        "--trait", "creative_professional",
        "--output_path", output_path,
        "--persona_instruction_type", "pos",
        "--assistant_name", "creative_professional",
        "--judge_model", "gpt-4o-mini",
        "--version", "extract"
    ]
    
    print(f"🚀 執行指令: {' '.join(cmd)}")
    
    # 設定環境變數
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["TRANSFORMERS_VERBOSITY"] = "info"  # 增加日誌詳細程度
    
    try:
        # 執行並即時顯示輸出
        process = subprocess.Popen(
            cmd, 
            env=env, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        print("📋 即時輸出:")
        print("-" * 50)
        
        # 即時顯示輸出
        for line in process.stdout:
            print(line.rstrip())
        
        # 等待完成
        return_code = process.wait()
        
        print("-" * 50)
        print(f"✅ 執行完成，返回碼: {return_code}")
        
        # 檢查結果
        if os.path.exists(output_path):
            print(f"📊 檢查輸出檔案: {output_path}")
            
            # 讀取並分析結果
            df = pd.read_csv(output_path)
            print(f"   總行數: {len(df)}")
            print(f"   欄位: {df.columns.tolist()}")
            
            # 檢查答案是否為空
            if 'answer' in df.columns:
                empty_answers = df['answer'].isna().sum()
                non_empty_answers = len(df) - empty_answers
                print(f"   空答案: {empty_answers}")
                print(f"   非空答案: {non_empty_answers}")
                
                # 顯示前幾個非空答案
                non_empty_df = df[df['answer'].notna()]
                if len(non_empty_df) > 0:
                    print("   前幾個回應範例:")
                    for i, row in non_empty_df.head(3).iterrows():
                        answer = str(row['answer'])[:100] + "..." if len(str(row['answer'])) > 100 else str(row['answer'])
                        print(f"     {i+1}: {answer}")
                else:
                    print("   ⚠️ 所有答案都是空的！")
            
            # 檢查分數
            if 'creative_professional' in df.columns:
                scores = df['creative_professional'].dropna()
                if len(scores) > 0:
                    avg_score = scores.mean()
                    print(f"   平均 creative_professional 分數: {avg_score:.4f}")
                else:
                    print("   ⚠️ 沒有有效的創意分數")
            
        else:
            print(f"❌ 輸出檔案不存在: {output_path}")
            return False
        
        return return_code == 0
        
    except Exception as e:
        print(f"❌ 執行失敗: {e}")
        return False

def check_eval_persona_script():
    """檢查 eval_persona 腳本是否存在且支援的參數"""
    
    print("🔍 檢查 eval_persona 腳本...")
    
    try:
        # 檢查 help
        result = subprocess.run(
            ["python", "-m", "eval.eval_persona", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print("📝 eval_persona 可用參數:")
        print(result.stdout[:1000] + "..." if len(result.stdout) > 1000 else result.stdout)
        
        return True
        
    except Exception as e:
        print(f"❌ 無法檢查 eval_persona: {e}")
        return False

if __name__ == "__main__":
    print("🚀 開始 Gemma-3 eval_persona 整合測試...")
    
    # 先檢查腳本
    if check_eval_persona_script():
        print("\n" + "="*60)
        # 執行測試
        success = test_eval_persona_gemma3()
        
        if success:
            print("\n✅ 測試成功完成")
        else:
            print("\n❌ 測試失敗")
    else:
        print("\n❌ eval_persona 腳本檢查失敗")
