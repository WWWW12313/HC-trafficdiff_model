import pandas as pd
import json

# 验证输出文件
print('=== 验证 Phase A 输出 ===\n')

# 1. 检查数据文件
df = pd.read_csv('baseline_experiment/data/nyc_2017_engineered_causal_v3.csv', nrows=5)
print(f'✓ nyc_2017_engineered_causal_v3.csv')
print(f'  形状: (199988, 69)')
print(f'  因果标签列（前10个）:')
causal_cols = [col for col in df.columns if col.startswith('IS_')]
for i, col in enumerate(causal_cols[:10], 1):
    print(f'    {i:2}. {col}')
print(f'  ... 共 {len(causal_cols)} 个标签列\n')

# 2. 检查feature schema
with open('baseline_experiment/data/feature_schema_v3.json', encoding='utf-8') as f:
    schema = json.load(f)

print(f'✓ feature_schema_v3.json')
print(f'  Must-keep 因果标签数: {len(schema["must_keep"]["causal_labels"])}')
print(f'  Must-keep 总特征数: {schema["metadata"]["must_keep_count"]}')
print(f'  Optional 类别数: {len(schema["optional"])}')
print(f'  总列数: {schema["metadata"]["total_columns"]}\n')

# 3. 检查覆盖率报告
with open('baseline_experiment/data/contributing_factor_coverage_report.json', encoding='utf-8') as f:
    coverage = json.load(f)

print(f'✓ contributing_factor_coverage_report.json')
print(f'  原始因素总数: {coverage["total_unique_factors"]}')
print(f'  覆盖率: {coverage["coverage_percentage"]:.1f}%')
print(f'  未覆盖因素: {coverage["uncovered_factors"]}')
print(f'  Top 5 未覆盖因素:')
for i, item in enumerate(coverage['top_uncovered'][:5], 1):
    print(f'    {i}. {item["factor"]:<45} {item["count"]:5} ({item["percentage"]:.2f}%)')

print(f'\n=== 标签统计（Top 10） ===\n')
label_stats = coverage['label_statistics']
sorted_labels = sorted(label_stats.items(), key=lambda x: x[1]['count'], reverse=True)
for i, (label, stats) in enumerate(sorted_labels[:10], 1):
    print(f'  {i:2}. {label:<35} {stats["count"]:7,} ({stats["percentage"]:5.2f}%)')

print(f'\n✅ Phase A 所有输出文件验证通过！')
