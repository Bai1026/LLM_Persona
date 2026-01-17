import torch
import os
import pandas as pd
from typing import Dict, List, Tuple
from generate_vec import get_persona_effective, get_hidden_p_and_r_batched
from transformers import AutoModelForCausalLM, AutoTokenizer

class MultiRoleVectorGenerator:
    """多角色向量產生器 - 修正版"""
    
    def __init__(self, model_name: str, save_dir: str):
        self.model_name = model_name
        self.model_short_name = model_name.split('/')[-1]
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 載入模型
        print("🤖 載入模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_role_vectors(self, roles: List[str], strategy: str = "individual"):
        """產生多角色向量"""
        
        print(f"📊 使用策略: {strategy}")
        
        if strategy == "individual":
            return self._generate_individual_vectors(roles)
        elif strategy == "vs_baseline":
            return self._generate_baseline_vectors(roles)
        elif strategy == "pairwise":
            return self._generate_pairwise_vectors(roles)
        else:
            raise ValueError(f"未知策略: {strategy}")
    
    def _generate_individual_vectors(self, roles: List[str]) -> Dict[str, torch.Tensor]:
        """產生獨立角色向量（角色特徵 vs 中性基準）"""
        vectors = {}
        
        for role_name in roles:
            print(f"🎭 計算 {role_name} 向量...")
            
            pos_path = f"eval_persona_extract/{self.model_short_name}/{role_name}_pos_instruct.csv"
            neg_path = f"eval_persona_extract/{self.model_short_name}/{role_name}_neutral_instruct.csv"
            
            if not (os.path.exists(pos_path) and os.path.exists(neg_path)):
                print(f"⚠️  找不到 {role_name} 的資料檔案，跳過...")
                continue
            
            try:
                # 檢查資料品質
                quality_info = self._check_data_quality_adaptive(pos_path, neg_path, role_name)
                
                if not quality_info['is_sufficient']:
                    print(f"⚠️  {role_name} 資料品質不足，跳過...")
                    continue
                
                # 使用適應性閾值計算向量
                vector = self._compute_single_vector_adaptive(
                    pos_path, neg_path, role_name, quality_info
                )
                
                if vector is not None:
                    vectors[role_name] = vector
                    
                    # 儲存個別向量
                    vector_path = os.path.join(self.save_dir, f"{role_name}_response_avg_diff.pt")
                    torch.save(vector, vector_path)
                    print(f"💾 儲存: {vector_path}")
                    print(f"📏 向量形狀: {vector.shape}")
                else:
                    print(f"❌ {role_name} 向量計算失敗")
                    
            except Exception as e:
                print(f"❌ 處理 {role_name} 時發生錯誤: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return vectors
    
    def _check_data_quality_adaptive(self, pos_path: str, neg_path: str, role_name: str) -> Dict:
        """適應性資料品質檢查"""
        try:
            pos_data = pd.read_csv(pos_path)
            neg_data = pd.read_csv(neg_path)
            
            # 檢查必要欄位
            if role_name not in pos_data.columns or role_name not in neg_data.columns:
                print(f"   ❌ 找不到 {role_name} 欄位")
                return {'is_sufficient': False}
            
            # 分析分數分佈
            pos_scores = pos_data[role_name].dropna()
            neg_scores = neg_data[role_name].dropna()
            
            print(f"   📊 分數分佈:")
            print(f"      正面資料: 平均 {pos_scores.mean():.2f}, 範圍 {pos_scores.min():.2f}-{pos_scores.max():.2f}")
            print(f"      負面資料: 平均 {neg_scores.mean():.2f}, 範圍 {neg_scores.min():.2f}-{neg_scores.max():.2f}")
            
            # 計算適應性閾值
            pos_median = pos_scores.median()
            neg_median = neg_scores.median()
            
            # 如果負面資料分數偏高，使用相對閾值
            if neg_median > 70:
                pos_threshold = max(pos_median, 80)  # 正面要求更高
                neg_threshold = neg_median  # 負面使用中位數
            else:
                pos_threshold = max(50, pos_median)
                neg_threshold = min(50, neg_median)
            
            print(f"   🎯 適應性閾值: 正面 >= {pos_threshold:.1f}, 負面 <= {neg_threshold:.1f}")
            
            # 篩選高品質樣本
            pos_quality = pos_data[
                (pos_data[role_name] >= pos_threshold) & 
                (pos_data["coherence"] >= 50)
            ]
            
            neg_quality = neg_data[
                (neg_data[role_name] <= neg_threshold) & 
                (neg_data["coherence"] >= 50)
            ]
            
            print(f"   📊 高品質樣本: 正面 {len(pos_quality)}, 負面 {len(neg_quality)}")
            
            # 放寬要求：只要有合理的樣本數量即可
            is_sufficient = len(pos_quality) >= 3 and len(neg_quality) >= 3
            
            return {
                'is_sufficient': is_sufficient,
                'pos_threshold': pos_threshold,
                'neg_threshold': neg_threshold,
                'pos_count': len(pos_quality),
                'neg_count': len(neg_quality)
            }
            
        except Exception as e:
            print(f"   ❌ 檢查資料品質時出錯: {e}")
            return {'is_sufficient': False}
    
    def _compute_single_vector_adaptive(self, pos_path: str, neg_path: str, 
                                       trait: str, quality_info: Dict) -> torch.Tensor:
        """使用適應性閾值計算向量"""
        try:
            # 使用適應性閾值
            pos_threshold = quality_info.get('pos_threshold', 70)
            neg_threshold = quality_info.get('neg_threshold', 30)
            
            print(f"   🎯 使用閾值: 正面 >= {pos_threshold:.1f}, 負面 <= {neg_threshold:.1f}")
            
            # 手動篩選資料
            pos_data = pd.read_csv(pos_path)
            neg_data = pd.read_csv(neg_path)
            
            # 篩選高品質正面樣本
            pos_effective = pos_data[
                (pos_data[trait] >= pos_threshold) & 
                (pos_data["coherence"] >= 50)
            ]
            
            # 篩選高品質負面樣本
            neg_effective = neg_data[
                (neg_data[trait] <= neg_threshold) & 
                (neg_data["coherence"] >= 50)
            ]
            
            print(f"   📊 篩選結果: 正面 {len(pos_effective)}, 負面 {len(neg_effective)}")
            
            if len(pos_effective) == 0 or len(neg_effective) == 0:
                print(f"   ⚠️  篩選後資料不足")
                return None
            
            # 提取提示詞和回應
            pos_prompts = pos_effective['prompt'].tolist()
            pos_responses = pos_effective['answer'].tolist()
            neg_prompts = neg_effective['prompt'].tolist()
            neg_responses = neg_effective['answer'].tolist()
            
            # 限制資料數量避免記憶體問題
            max_samples = 20
            if len(pos_prompts) > max_samples:
                pos_prompts = pos_prompts[:max_samples]
                pos_responses = pos_responses[:max_samples]
            if len(neg_prompts) > max_samples:
                neg_prompts = neg_prompts[:max_samples]
                neg_responses = neg_responses[:max_samples]
            
            # 計算適當的批次大小
            batch_size = min(2, len(pos_prompts), len(neg_prompts))
            
            print(f"   🧠 提取正面激活 (樣本: {len(pos_prompts)}, 批次: {batch_size})...")
            _, _, pos_activations = get_hidden_p_and_r_batched(
                self.model, self.tokenizer, 
                pos_prompts, pos_responses,
                batch_size=batch_size
            )
            
            print(f"   🧠 提取負面激活 (樣本: {len(neg_prompts)}, 批次: {batch_size})...")
            _, _, neg_activations = get_hidden_p_and_r_batched(
                self.model, self.tokenizer,
                neg_prompts, neg_responses, 
                batch_size=batch_size
            )
            
            # 計算差異向量
            print(f"   🎯 計算向量差異...")
            response_avg_diff = torch.stack([
                pos_activations[l].mean(0).float() - 
                neg_activations[l].mean(0).float() 
                for l in range(len(pos_activations))
            ], dim=0)
            
            return response_avg_diff
            
        except Exception as e:
            print(f"   ❌ 向量計算錯誤: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_baseline_vectors(self, roles: List[str], baseline_role: str = None) -> Dict[str, torch.Tensor]:
        """產生相對於基準角色的向量"""
        print("🔄 產生基準對比向量...")
        return {}
    
    def _generate_pairwise_vectors(self, roles: List[str]) -> Dict[str, torch.Tensor]:
        """產生角色對比向量"""
        print("🔄 產生角色對比向量...")
        return {}

# 使用範例
def main():
    
    # generator = MultiRoleVectorGenerator(
    #     model_name="Qwen/Qwen2.5-7B-Instruct",
    #     save_dir="persona_vectors/Qwen2.5-7B-Instruct/multi_role/"
    # )

    # # TODO: change to Llama-3.1-8B-Instruct
    # generator = MultiRoleVectorGenerator(
    #     model_name="meta-llama/Llama-3.1-8B-Instruct",
    #     save_dir="persona_vectors/Llama-3.1-8B-Instruct/multi_role/"
    # )

    generator = MultiRoleVectorGenerator(
        model_name="google/gemma-3-4b-it",
        save_dir="persona_vectors/gemma-3-4b-it/multi_role/"
    )
    
    roles = [
        "creative_professional",
        # "analytical_thinker", 
        # "empathetic_counselor",
        # "academic_researcher",
        # "customer_user",
        # "digital_nomad",
        "environmentalist",
        "futurist",
        # "industry_insider",
        # "social_entrepreneur",
        # "startup_founder",
        # "visionary_millionaire"
    ]
    
    for role in roles:
        print(f"🎯 先測試 {role} 的資料產生...")
        try:
            generator.generate_role_vectors([role], strategy="individual")
            print(f"✅ {role} 向量產生成功！")
        except Exception as e:
            print(f"❌ {role} 向量計算失敗: {e}")
        print("--------------------------------------------------")

    # 先測試一個角色
    # print("🎯 產生獨立角色向量...")
    # individual_vectors = generator.generate_role_vectors(roles, "individual")
    
    print("✅ 測試完成！")

if __name__ == "__main__":
    main()