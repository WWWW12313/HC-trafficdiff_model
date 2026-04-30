import pandas as pd
import json
from collections import Counter

# 读取原始数据
print("加载数据...")
df = pd.read_csv('tab-ddpm-main/nyc_2017_pristine_v8.csv')

# 获取列名
print('\n=== 列名列表（前30列） ===')
for i, col in enumerate(df.columns[:30], 1):
    print(f'{i:2}. {col}')

# 查找 CONTRIBUTING FACTOR 相关列
cf_cols = [col for col in df.columns if 'CONTRIBUTING' in col.upper() or 'FACTOR' in col.upper()]
print(f'\n=== CONTRIBUTING FACTOR 相关列 ({len(cf_cols)} 列) ===')
for col in cf_cols:
    print(f'  • {col}')

# 获取形状
print(f'\n=== 数据形状 ===')
print(f'总行数: {len(df):,}')
print(f'总列数: {len(df.columns)}')

# 统计所有 CONTRIBUTING FACTOR 的唯一值
print(f'\n=== 收集所有 CONTRIBUTING FACTOR 唯一值 ===')
all_factors = []
for col in cf_cols:
    values = df[col].dropna().astype(str).values
    all_factors.extend(values)

factor_counts = Counter(all_factors)
print(f'总共收集到 {len(all_factors):,} 个值')
print(f'唯一值数量: {len(factor_counts)}')

print(f'\n=== Top 30 最常见的事故原因 ===')
for i, (factor, count) in enumerate(factor_counts.most_common(30), 1):
    pct = count / len(all_factors) * 100
    print(f'{i:2}. {factor:<50} {count:6,} ({pct:5.2f}%)')

# 保存完整列表
output = {
    'cf_columns': cf_cols,
    'total_rows': len(df),
    'total_columns': len(df.columns),
    'all_columns': df.columns.tolist(),
    'top_50_factors': [
        {'factor': k, 'count': v, 'percentage': v/len(all_factors)*100}
        for k, v in factor_counts.most_common(50)
    ]
}

with open('contributing_factor_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print('\n✅ 分析完成！结果已保存到 contributing_factor_analysis.json')
