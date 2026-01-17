#!/usr/bin/env python3
"""
測試修正後的 multi_persona_handler.py 是否能正常載入 Gemma-3 向量
"""

import sys
from pathlib import Path
import torch

# 添加路徑
sys.path.append(str(Path(__file__).parent))

def test_multi_persona_handler():
    """測試 MultiPersonaHandler 和 MultiPersonaChatbot"""
    
    print("🧪 測試 MultiPersonaHandler...")
    
    # 檢查向量檔案是否存在
    vector_dir = Path(__file__).parent.parent / "persona_vectors" / "gemma-3-4b-it" / "multi_role"
    print(f"📁 檢查向量目錄: {vector_dir}")
    
    if not vector_dir.exists():
        print(f"❌ 向量目錄不存在: {vector_dir}")
        return False
    
    # 查找可用的向量檔案
    vector_files = list(vector_dir.glob("*.pt"))
    print(f"📊 找到 {len(vector_files)} 個向量檔案:")
    for f in vector_files:
        print(f"   - {f.name}")
    
    if len(vector_files) < 1:
        print("❌ 沒有找到向量檔案")
        return False
    
    # 測試載入 MultiPersonaHandler
    try:
        from multi_persona_handler import MultiPersonaHandler
        
        # 使用前幾個檔案進行測試
        test_files = [str(f) for f in vector_files[:min(3, len(vector_files))]]
        print(f"🎯 測試檔案: {[Path(f).name for f in test_files]}")
        
        handler = MultiPersonaHandler(test_files)
        print("✅ MultiPersonaHandler 載入成功")
        
        # 測試融合
        fused_vector = handler.fuse_vectors()
        print(f"✅ 向量融合成功，維度: {fused_vector.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ MultiPersonaHandler 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_persona_chatbot():
    """測試 MultiPersonaChatbot"""
    
    print("\n🤖 測試 MultiPersonaChatbot...")
    
    try:
        from multi_persona_handler import MultiPersonaChatbot
        
        # 檢查向量檔案
        vector_dir = Path(__file__).parent.parent / "persona_vectors" / "gemma-3-4b-it" / "multi_role"
        vector_files = list(vector_dir.glob("*.pt"))
        
        if len(vector_files) < 1:
            print("❌ 沒有向量檔案可供測試")
            return False
        
        # 使用第一個檔案測試
        test_vector = vector_files[0].name
        print(f"🎯 使用向量檔案: {test_vector}")
        
        # 創建聊天機器人（不實際載入模型，只測試初始化）
        chatbot = MultiPersonaChatbot(
            model_name="google/gemma-3-4b-it",
            vector_paths=[test_vector],
            layer_idx=20,
            steering_coef=2.0
        )
        
        print("✅ MultiPersonaChatbot 初始化成功")
        
        # 測試模式設定
        chatbot.set_persona_mode("balanced")
        print("✅ persona 模式設定成功")
        
        return True
        
    except Exception as e:
        print(f"❌ MultiPersonaChatbot 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主測試函式"""
    
    print("🚀 開始測試修正後的 multi_persona_handler...")
    print("=" * 60)
    
    # 測試 MultiPersonaHandler
    handler_success = test_multi_persona_handler()
    
    # 測試 MultiPersonaChatbot（如果 handler 測試成功）
    chatbot_success = False
    if handler_success:
        chatbot_success = test_multi_persona_chatbot()
    
    # 總結
    print("\n" + "=" * 60)
    print("📊 測試總結:")
    print(f"   MultiPersonaHandler: {'✅ 成功' if handler_success else '❌ 失敗'}")
    print(f"   MultiPersonaChatbot: {'✅ 成功' if chatbot_success else '❌ 失敗'}")
    
    if handler_success and chatbot_success:
        print("\n🎉 所有測試通過！Gemma-3 支援已就緒")
    else:
        print("\n⚠️ 部分測試失敗，需要進一步調整")

if __name__ == "__main__":
    main()
