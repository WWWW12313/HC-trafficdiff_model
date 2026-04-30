import pandas as pd
import numpy as np

# 加载数据
df = pd.read_csv('tab-ddpm-main/nyc_2017_pristine_v8.csv')

print("=" * 60)
print(f"Shape: {df.shape}")
print("\nColumns:")
for col in df.columns:
    print(f"  - {col}")
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nFirst few rows:")
print(df.head(2))
