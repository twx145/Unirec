# 看数据内容.py
import pandas as pd

# 改成你的文件路径
df = pd.read_parquet("./data/temp.parquet")

print("数据行数：", len(df))
print("\n所有列名：")
print(list(df.columns))
print("\n数据：")
print(df.iloc[0,])