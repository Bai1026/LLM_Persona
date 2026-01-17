import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

class MultiPersonaHandler:
    """處理多重 Persona 向量的類別"""
    
    def __init__(self, vector_paths: List[str], fusion_method: str = "weighted_average"):
        self.vector_paths = vector_paths
        self.fusion_method = fusion_method
        self.personas = {}
        self.fused_vector = None
        
        # 載入所有 persona 向量
        self.load_all_personas()
        
    def load_all_personas(self):
        """載入所有 persona 向量檔案"""
        for i, path in enumerate(self.vector_paths):
            persona_name = f"persona_{i+1}"
            print(f"📁 載入 {persona_name}: {path}")
            
            vector = torch.load(path, map_location='cpu')
            self.personas[persona_name] = vector
            print(f"✅ {persona_name} 向量維度: {vector.shape}")
    
    def fuse_vectors(self, weights: Optional[List[float]] = None) -> torch.Tensor:
        """融合多個 persona 向量"""
        if not self.personas:
            raise ValueError("沒有可用的 persona 向量")
        
        vectors = list(self.personas.values())
        
        if self.fusion_method == "weighted_average":
            return self._weighted_average_fusion(vectors, weights)
        elif self.fusion_method == "concatenate":
            return self._concatenate_fusion(vectors)
        elif self.fusion_method == "attention":
            return self._attention_fusion(vectors)
        elif self.fusion_method == "dynamic":
            return self._dynamic_fusion(vectors)
        else:
            raise ValueError(f"不支援的融合方法: {self.fusion_method}")
    
    def _weighted_average_fusion(self, vectors: List[torch.Tensor], weights: Optional[List[float]] = None) -> torch.Tensor:
        """加權平均融合"""
        if weights is None:
            weights = [1.0 / len(vectors)] * len(vectors)
        
        if len(weights) != len(vectors):
            raise ValueError("權重數量必須與向量數量相同")
        
        fused = torch.zeros_like(vectors[0])
        for vector, weight in zip(vectors, weights):
            fused += weight * vector
        
        return fused
    
    def _concatenate_fusion(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        """串接融合"""
        return torch.cat(vectors, dim=-1)
    
    def _attention_fusion(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        """注意力機制融合"""
        # 簡化的注意力機制
        stacked = torch.stack(vectors, dim=0)  # [num_vectors, ...]
        
        # 計算注意力權重
        attention_scores = torch.softmax(torch.randn(len(vectors)), dim=0)
        
        # 應用注意力權重
        fused = torch.sum(attention_scores.unsqueeze(-1) * stacked, dim=0)
        return fused
    
    def _dynamic_fusion(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        """動態融合（根據輸入調整）"""
        # 這裡可以根據當前輸入動態調整融合策略
        # 暫時使用等權重平均
        return self._weighted_average_fusion(vectors)

class MultiPersonaChatbot:
    """支援多重 Persona 的聊天機器人"""
    
    def __init__(self, model_name: str, vector_paths: List[str], 
                 layer_idx: int = 20, steering_coef: float = 2.0,
                 fusion_method: str = "weighted_average"):
        
        # 導入本地的 PersonaChatbot
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent))
        
        from interactive_chat import PersonaChatbot
        
        # 處理檔案路徑，自動添加完整路徑
        processed_paths = []
        for path in vector_paths:
            if not Path(path).exists():

                # 嘗試在 persona_vectors/Qwen2.5-7B-Instruct/multi_role/ 目錄中尋找
                full_path = Path(__file__).parent.parent / "persona_vectors" / "Qwen2.5-7B-Instruct" / "multi_role" / path
                # full_path = Path(__file__).parent.parent / "persona_vectors" / "Llama-3.1-8B-Instruct" / "multi_role" / path
                # full_path = Path(__file__).parent.parent / "persona_vectors" / "gemma-2-9b-it" / "multi_role" / path
                # full_path = Path(__file__).parent.parent / "persona_vectors" / "gemma-3-4b-it" / "multi_role" / path

                if full_path.exists():
                    processed_paths.append(str(full_path))
                    print(f"📁 找到向量檔案: {full_path}")
                else:
                    raise FileNotFoundError(f"找不到向量檔案: {path}")
            else:
                processed_paths.append(path)
        
        self.base_chatbot = PersonaChatbot(model_name, processed_paths[0], layer_idx, steering_coef)
        self.persona_handler = MultiPersonaHandler(processed_paths, fusion_method)
        self.current_weights = [1.0/len(vector_paths)] * len(vector_paths)
        
        # 使用融合後的向量
        self.update_persona_weights(self.current_weights)

        # TODO: 手動更改 weights
        # self.update_persona_weights([0.6, 0.2, 0.1, 0.1])
    
    def update_persona_weights(self, weights: List[float]):
        """更新 persona 權重並重新融合向量"""
        self.current_weights = weights
        fused_vector = self.persona_handler.fuse_vectors(weights)
        
        # 更新基礎聊天機器人的 persona 向量
        self.base_chatbot.persona_vector = fused_vector
        print(f"🎭 更新 persona 權重: {weights}")
    
    def set_persona_mode(self, mode: str):
        """設定特定的 persona 模式"""
        num_personas = len(self.persona_handler.personas)
        
        if mode == "balanced":
            # 平衡模式：所有 persona 等權重
            weights = [1.0/num_personas] * num_personas
        elif mode == "persona_1":
            # 主要使用第一個 persona
            weights = [0.8, 0.1, 0.1][:num_personas]
        elif mode == "persona_2":
            # 主要使用第二個 persona
            weights = [0.1, 0.8, 0.1][:num_personas]
        elif mode == "persona_3":
            # 主要使用第三個 persona
            weights = [0.1, 0.1, 0.8][:num_personas]
        elif mode == "creative":
            # 創意模式：動態權重
            weights = [0.4, 0.3, 0.3][:num_personas]
        else:
            raise ValueError(f"不支援的模式: {mode}")
        
        self.update_persona_weights(weights)
    
    def generate_response(self, user_input: str, max_tokens: int = 16384) -> str:
        """產生回應"""
        return self.base_chatbot.generate_response(user_input, max_tokens)
    
    def reset_conversation(self):
        """重設對話歷史"""
        self.base_chatbot.conversation_history = []