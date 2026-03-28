import pandas as pd

# ====================== 你只需要改这里 ======================
INPUT_FILE = "./data/sample_data.parquet"          # 你的原始数据集路径
OUTPUT_FILE = "./data/temp.parquet"# 输出新文件路径

# 提取方式 三选一
ROWS_TO_EXTRACT = [10]  # 1. 提取指定多行（行号）
# ROWS_TO_EXTRACT = 5           # 2. 只提取第 5 行
# ROWS_TO_EXTRACT = slice(0,100) # 3. 提取前 100 行（0~99）
# ===========================================================

# 读取数据
df = pd.read_parquet(INPUT_FILE)

# 提取指定行
if isinstance(ROWS_TO_EXTRACT, int):
    extracted = df.iloc[[ROWS_TO_EXTRACT]]
else:
    extracted = df.iloc[ROWS_TO_EXTRACT]

# 保存新文件（格式和官方完全一致）
extracted.to_parquet(
    OUTPUT_FILE,
    index=False,
    engine="pyarrow"
)

# 输出结果（纯英文/数字，无报错）
print("extract success!")
print(f"raw data: {df.shape[0]} rows")
print(f"new file: {OUTPUT_FILE}, {extracted.shape[0]} rows")
print("\nextracted data preview:")
print(extracted.head())