import os
import subprocess
import pandas as pd
from typing import List

class Gemma3ActivationGenerator:
    """專門為 Gemma-3 設計的多角色激活資料產生器"""
    
    def __init__(self, model_name: str = "google/gemma-3-4b-it"):
        self.model_name = model_name
        self.model_short_name = model_name.split('/')[-1]
        
        # Gemma-3 特殊設定
        self.gemma3_config = {
            "torch_dtype": "bfloat16",
            "attn_implementation": "eager",  # 避免 FlashAttention2 問題
            "trust_remote_code": True
        }
    
    def generate_role_activations(self, roles: List[str], gpu_id: int = 0, force_regenerate: bool = False):
        """為多個角色產生激活資料，針對 Gemma-3 最佳化"""
        
        for role_name in roles:
            print(f"🎭 產生 {role_name} 的激活資料...")
            
            # 產生角色特徵資料
            success_pos = self._run_eval_persona(
                role_name=role_name,
                persona_type="pos",
                assistant_name=role_name,
                gpu_id=gpu_id,
                force_regenerate=force_regenerate
            )
            
            if success_pos:
                # 產生中性基準資料
                success_neg = self._run_eval_persona(
                    role_name=role_name,
                    persona_type="neg", 
                    assistant_name="helpful",
                    gpu_id=gpu_id,
                    output_suffix="neutral",
                    force_regenerate=force_regenerate
                )
                
                if success_pos and success_neg:
                    print(f"✅ {role_name} 的激活資料產生完成")
                else:
                    print(f"⚠️ {role_name} 的部分資料產生失敗")
            else:
                print(f"❌ {role_name} 的正向資料產生失敗，跳過負向資料")
    
    def _run_eval_persona(self, role_name: str, persona_type: str, 
                         assistant_name: str, gpu_id: int, output_suffix: str = None,
                         force_regenerate: bool = False) -> bool:
        """執行單一角色的評估，針對 Gemma-3 最佳化"""
        
        # 建立輸出路徑
        if output_suffix:
            output_path = f"eval_persona_extract/{self.model_short_name}/{role_name}_{output_suffix}_instruct.csv"
        else:
            output_path = f"eval_persona_extract/{self.model_short_name}/{role_name}_{persona_type}_instruct.csv"
        
        # 檢查是否需要重新產生
        if not force_regenerate and os.path.exists(output_path):
            print(f"   📁 檔案已存在: {output_path}")
            return self._check_data_quality(output_path, role_name)
        
        # 強制重新產生時刪除舊檔案
        if force_regenerate and os.path.exists(output_path):
            print(f"🗑️  刪除舊檔案: {output_path}")
            os.remove(output_path)
        
        # 建立輸出目錄
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 建立指令，移除不支援的參數
        cmd = [
            "python", "-m", "eval.eval_persona",
            "--model", self.model_name,
            "--trait", role_name,
            "--output_path", output_path,
            "--persona_instruction_type", persona_type,
            "--assistant_name", assistant_name,
            "--judge_model", "gpt-4o-mini",
            "--version", "extract"
        ]
        
        # 設定環境變數
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["TRANSFORMERS_VERBOSITY"] = "error"  # 減少警告訊息
        
        try:
            print(f"   🚀 執行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, env=env, check=True, 
                                  capture_output=True, text=True)
            
            print(f"✅ {role_name} ({persona_type}) 資料產生完成")
            
            # 檢查產生的資料品質
            return self._check_data_quality(output_path, role_name)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ {role_name} ({persona_type}) 資料產生失敗:")
            print(f"   返回碼: {e.returncode}")
            if e.stdout:
                print(f"   標準輸出: {e.stdout}")
            if e.stderr:
                print(f"   錯誤輸出: {e.stderr}")
            return False
        
        except Exception as e:
            print(f"❌ {role_name} ({persona_type}) 執行發生異常: {e}")
            return False
    
    def _check_data_quality(self, output_path: str, role_name: str) -> bool:
        """檢查產生的資料品質"""
        try:
            if not os.path.exists(output_path):
                print(f"   ❌ 檔案不存在: {output_path}")
                return False
            
            data = pd.read_csv(output_path)
            
            if role_name in data.columns:
                valid_scores = data[role_name].dropna()
                if len(valid_scores) > 0:
                    avg_score = valid_scores.mean()
                    valid_ratio = len(valid_scores) / len(data)
                    print(f"   📊 資料品質: 平均分數={avg_score:.2f}, 有效比例={valid_ratio:.2%}")
                    return valid_ratio > 0.5  # 至少 50% 的資料有效
                else:
                    print(f"   ⚠️  沒有有效的分數資料")
                    return False
            else:
                print(f"   ⚠️  找不到 {role_name} 欄位，可用欄位: {data.columns.tolist()}")
                return False
                
        except Exception as e:
            print(f"   ❌ 檢查資料品質時出錯: {e}")
            return False

def main():
    """主要執行函式"""
    
    print("🚀 開始 Gemma-3 角色激活資料產生...")
    
    generator = Gemma3ActivationGenerator()
    
    # 先測試單一角色
    test_roles = [
        "creative_professional",
        # "analytical_thinker", 
        # "empathetic_counselor"
    ]
    
    print(f"🎯 測試角色: {test_roles}")
    
    # 執行產生
    generator.generate_role_activations(
        roles=test_roles, 
        gpu_id=0, 
        force_regenerate=True
    )
    
    print("\n🎉 Gemma-3 激活資料產生完成！")

if __name__ == "__main__":
    main()