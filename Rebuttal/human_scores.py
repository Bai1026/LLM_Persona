
# # import csv from human_scores.csv
# import pandas as pd
# from pathlib import Path
# from rich import print

# # read data from csv
# data_df = pd.read_csv('human_scores_2.csv', header=None)
# # 只取前 72 個欄位的評分資料（從第 2 欄開始，跳過第一欄的 "People" 標籤）
# # 並且跳過第一列的標題和最後幾列的平均值
# RATINGS = data_df.iloc[1:, 1:73].to_numpy()
# # print(RATINGS)
# print(f"原始資料維度: {RATINGS.shape}")

# # 轉換為 float，將無法轉換的值設為 NaN
# RATINGS = pd.DataFrame(RATINGS).apply(pd.to_numeric, errors='coerce').to_numpy()
# # print(f"前幾列資料:\n{RATINGS[:3, :10]}")  # 顯示前 3 個評分者的前 10 個分數

# # 每三個欄位為一組，分別對應三個方法 (SA, LLMD, BILLY)
# METHODS = ['SA', 'LLMD', 'BILLY']
# METHODS_COUNT = 3
# METRICS = ['Originality', 'Elaboration']
# METRICS_COUNT = 12
# TASK = ['AUT', 'INS', 'SCI', 'SIMI']
# TASK_COUNT = 4

# RESULTS = []

# repeat_count = 24
# for i in range(repeat_count):
#     RESULTS.append(0)

# print(RESULTS)

# print("\n" + "="*60)
# print("開始計算每個方法和指標的平均分數")
# print("="*60)

# # --- 循環計算每個方法和指標的平均分數 ---

# for i in range(3):
#     # print(i)
#     for y in range(repeat_count):
#         RESULTS[y] += RATINGS[1][i*y].astype(float)

# print(RESULTS)

# # output to csv
# output_df = pd.DataFrame(RESULTS)
# output_df.to_csv('human_scores_avg.csv', index=False, header=False)
# print("\n已儲存平均分數到: human_scores_avg.csv")

from rich import print

data = [
9.0,
11.325,
10.2425,
8.66,
10.9125,
10.41,
10.33,
10.66,
11.415,
9.99,
11.08,
10.9975,
11.33,
11.918,
11.0775,
9.33,
11.085,
9.4125,
12.33,
12.0825,
11.9975,
11.33,
11.6575,
11.245000000000001
]

for num in data:
    print(f"{num/3:.9f}", end=", ")