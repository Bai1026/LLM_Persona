# import subprocess
# import os
# from typing import List, Dict

# class MultiRoleActivationGenerator:
#     """多角色激活資料產生器"""
    
#     def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
#         self.model_name = model_name
#         self.model_short_name = model_name.split('/')[-1]
    
#     def generate_role_activations(self, roles: List[str], gpu_id: int = 0):
#         """為多個角色產生激活資料"""
        
#         for role_name in roles:
#             print(f"🎭 產生 {role_name} 的激活資料...")
            
#             # 產生角色特徵資料（相當於 pos）
#             self._run_eval_persona(
#                 role_name=role_name,
#                 persona_type="pos",
#                 assistant_name=role_name,
#                 gpu_id=gpu_id
#             )
            
#             # 產生中性基準資料（相當於 neg）
#             self._run_eval_persona(
#                 role_name=role_name,
#                 persona_type="neg", 
#                 assistant_name="helpful",
#                 gpu_id=gpu_id,
#                 output_suffix="neutral"
#             )
    
#     def _run_eval_persona(self, role_name: str, persona_type: str, 
#                          assistant_name: str, gpu_id: int, output_suffix: str = None):
#         """執行單一角色的評估"""
        
#         if output_suffix:
#             output_path = f"eval_persona_extract/{self.model_short_name}/{role_name}_{output_suffix}_instruct.csv"
#         else:
#             output_path = f"eval_persona_extract/{self.model_short_name}/{role_name}_{persona_type}_instruct.csv"
        
#         # 建立輸出目錄
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
#         cmd = [
#             "python", "-m", "eval.eval_persona",
#             "--model", self.model_name,
#             "--trait", role_name,
#             "--output_path", output_path,
#             "--persona_instruction_type", persona_type,
#             "--assistant_name", assistant_name,
#             "--judge_model", "gpt-4o-mini",
#             "--version", "extract"
#         ]
        
#         env = os.environ.copy()
#         env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
#         try:
#             subprocess.run(cmd, env=env, check=True)
#             print(f"✅ {role_name} ({persona_type}) 資料產生完成")
#         except subprocess.CalledProcessError as e:
#             print(f"❌ {role_name} ({persona_type}) 資料產生失敗: {e}")

# # 使用範例
# def main():
#     generator = MultiRoleActivationGenerator()
    
#     roles = [
#         "creative_professional",
#         # "analytical_thinker", 
#         # "empathetic_counselor"
#     ]
    
#     generator.generate_role_activations(roles, gpu_id=0)

# if __name__ == "__main__":
#     main()

import subprocess
import os
from typing import List, Dict

class MultiRoleActivationGenerator:
    """多角色激活資料產生器"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        self.model_short_name = model_name.split('/')[-1]
    
    def generate_role_activations(self, roles: List[str], gpu_id: int = 0, force_regenerate: bool = False):
        """為多個角色產生激活資料"""
        
        for role_name in roles:
            print(f"🎭 產生 {role_name} 的激活資料...")
            
            # 產生角色特徵資料（相當於 pos）
            self._run_eval_persona(
                role_name=role_name,
                persona_type="pos",
                assistant_name=role_name,
                gpu_id=gpu_id,
                force_regenerate=force_regenerate
            )
            
            # 產生中性基準資料（相當於 neg）
            self._run_eval_persona(
                role_name=role_name,
                persona_type="neg", 
                assistant_name="helpful",
                gpu_id=gpu_id,
                output_suffix="neutral",
                force_regenerate=force_regenerate
            )
    
    def _run_eval_persona(self, role_name: str, persona_type: str, 
                         assistant_name: str, gpu_id: int, output_suffix: str = None,
                         force_regenerate: bool = False):
        """執行單一角色的評估"""
        
        if output_suffix:
            output_path = f"eval_persona_extract/{self.model_short_name}/{role_name}_{output_suffix}_instruct.csv"
        else:
            output_path = f"eval_persona_extract/{self.model_short_name}/{role_name}_{persona_type}_instruct.csv"
        
        # 如果強制重新產生，先刪除舊檔案
        if force_regenerate and os.path.exists(output_path):
            print(f"🗑️  刪除舊檔案: {output_path}")
            os.remove(output_path)
        
        # 建立輸出目錄
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
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
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        try:
            subprocess.run(cmd, env=env, check=True)
            print(f"✅ {role_name} ({persona_type}) 資料產生完成")
            
            # 檢查產生的資料品質
            self._check_data_quality(output_path, role_name)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ {role_name} ({persona_type}) 資料產生失敗: {e}")
    
    def _check_data_quality(self, output_path: str, role_name: str):
        """檢查產生的資料品質"""
        try:
            import pandas as pd
            data = pd.read_csv(output_path)
            
            if role_name in data.columns:
                valid_scores = data[role_name].dropna()
                if len(valid_scores) > 0:
                    print(f"   📊 {role_name} 分數: 平均 {valid_scores.mean():.2f}, 範圍 {valid_scores.min():.2f}-{valid_scores.max():.2f}")
                else:
                    print(f"   ⚠️  {role_name} 所有分數都是 NaN")
            else:
                print(f"   ⚠️  找不到 {role_name} 欄位，可用欄位: {data.columns.tolist()}")
                
        except Exception as e:
            print(f"   ❌ 檢查資料品質時出錯: {e}")

# 使用範例
def main():
    generator = MultiRoleActivationGenerator()
    
    roles = [
        "creative_professional",
        "analytical_thinker", 
        "empathetic_counselor"
    ]
    
    # 強制重新產生以解決 NaN 問題
    generator.generate_role_activations(roles, gpu_id=0, force_regenerate=True)

if __name__ == "__main__":
    main()