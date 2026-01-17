import pandas as pd
from pathlib import Path

# 定義三個檔案路徑
files = [
    'Creativity_A.xlsx',
    'Creativity_B.xlsx',
    'Creativity_C.xlsx'
]

# 讀取三個檔案
dfs = []
for file in files:
    df = pd.read_excel(file)
    print(f"\n讀取檔案: {file}")
    print(f"欄位: {df.columns.tolist()}")
    print(f"形狀: {df.shape}")
    print(f"前幾列:\n{df.head()}\n")
    dfs.append(df)

# 找出電子郵件地址的欄位名稱
email_col = None
for col in dfs[0].columns:
    if '電子郵件' in col or 'email' in col.lower() or '信箱' in col:
        email_col = col
        break

if email_col is None:
    print("可用的欄位名稱:")
    for i, col in enumerate(dfs[0].columns):
        print(f"{i}: {col}")
    print("\n請確認電子郵件地址的欄位名稱")
else:
    print(f"\n使用電子郵件欄位: {email_col}")
    
    # 合併資料
    # 為每個 dataframe 的欄位加上來源標記（除了 email）
    for i, df in enumerate(dfs):
        # 重命名欄位，加上 problem 編號
        new_cols = {}
        for col in df.columns:
            if col != email_col:
                new_cols[col] = f"{col}_problem{i+1}"
        df.rename(columns=new_cols, inplace=True)
    
    # 使用 email 作為 key 進行合併
    merged_df = dfs[0]
    for i in range(1, len(dfs)):
        merged_df = pd.merge(merged_df, dfs[i], on=email_col, how='outer')
    
    print(f"\n合併後的資料形狀: {merged_df.shape}")
    print(f"合併後的欄位: {merged_df.columns.tolist()}")
    print(f"\n前幾列:\n{merged_df.head()}")
    
    # 儲存結果
    output_file = 'merged_human_evaluation.xlsx'
    merged_df.to_excel(output_file, index=False)
    print(f"\n已儲存合併結果到: {output_file}")
