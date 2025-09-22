import logging
import warnings

# 設定日誌級別來抑制特定警告
logging.getLogger("vllm.model_executor.models").setLevel(logging.ERROR)

# 或者使用 warnings 模組來過濾特定警告
warnings.filterwarnings("ignore", message=".*Regarding multimodal models.*")

# 在您的主程式開始前導入這個模組
print("警告抑制設定已啟用")
