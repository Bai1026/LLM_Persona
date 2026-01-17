# 檢查您現在的資料結構
import pandas as pd

# 正面特質資料 (展現同理心諮商師特質)
pos_data = pd.read_csv("eval_persona_extract/Qwen2.5-7B-Instruct/empathetic_counselor_pos_instruct.csv")

# 中性對照資料 (一般助手回應)
neg_data = pd.read_csv("eval_persona_extract/Qwen2.5-7B-Instruct/empathetic_counselor_neutral_instruct.csv")

print("📊 資料內容:")
print(f"正面資料: {len(pos_data)} 筆回應")
print(f"中性資料: {len(neg_data)} 筆回應")
print(f"欄位: {pos_data.columns.tolist()}")