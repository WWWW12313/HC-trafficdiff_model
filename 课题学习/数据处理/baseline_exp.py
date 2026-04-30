import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
import os
import warnings
import json
import shutil # 引入 shutil 用于清理旧文件

# 忽略警告
warnings.filterwarnings('ignore')

# =============================================================================
# 1. 实验配置
# =============================================================================
INPUT_FILE = 'NYC_Crashes_2017_Clean_Causal.csv'
OUTPUT_DIR = 'baseline_experiment_results'
EPOCHS = 300
BATCH_SIZE = 500

# 如果文件夹存在，先清理旧的 metadata.json 防止报错
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
else:
    # 尝试删除旧的 metadata.json，解决 ValueError
    meta_path = os.path.join(OUTPUT_DIR, 'metadata.json')
    if os.path.exists(meta_path):
        os.remove(meta_path)
        print("⚠️ 检测到旧的 metadata.json，已自动删除。")

print(f"--- [Step 1] 加载数据: {INPUT_FILE} ---")
df = pd.read_csv(INPUT_FILE)

# =============================================================================
# 2. 特征工程适配
# =============================================================================
print("--- [Step 2] 正在适配特征 ---")

if 'CRASH_DATETIME' in df.columns:
    train_df = df.drop(columns=['CRASH_DATETIME'])
else:
    train_df = df.copy()

# 定义哪些列必须作为分类变量处理
categorical_columns = [
    'MONTH', 'DAY_OF_WEEK', 'HOUR', 
    'BOROUGH', 'WEATHER_TYPE', 'PRECIPITATION', 
    'ROAD_SURFACE', 'MEDIAN', 'ROAD_TYPE', 'PRIMARY_CAUSE',
    'LANES' 
]

# 转换为字符串类型，确保 pandas 层面一致性
for col in categorical_columns:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype(str)

print(f"  -> 训练数据维度: {train_df.shape}")

# =============================================================================
# 3. 构建元数据
# =============================================================================
print("\n--- [Step 3] 构建元数据 ---")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train_df)

# 强制更新元数据，覆盖自动检测的结果
print("  -> 正在强制修正元数据类型...")
for col in categorical_columns:
    if col in train_df.columns:
        metadata.update_column(column_name=col, sdtype='categorical')

# 保存元数据 (前面已删除了旧文件，这里不会再报错)
metadata.save_to_json(os.path.join(OUTPUT_DIR, 'metadata.json'))
print("✅ 元数据保存成功。")

# =============================================================================
# 4. 训练 Baseline A: CTGAN
# =============================================================================
print("\n========================================")
print(f"   开始训练 CTGAN (Epochs={EPOCHS})")
print("========================================")

ctgan = CTGANSynthesizer(
    metadata,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=True,
    cuda=True
)

ctgan.fit(train_df)

# 保存模型前先删除旧的，防止冲突
ctgan_path = os.path.join(OUTPUT_DIR, 'ctgan_model.pkl')
if os.path.exists(ctgan_path):
    os.remove(ctgan_path)
ctgan.save(ctgan_path)

# 生成数据
synthetic_data_ctgan = ctgan.sample(num_rows=len(train_df))
synthetic_data_ctgan.to_csv(os.path.join(OUTPUT_DIR, 'synthetic_ctgan.csv'), index=False)
print("✅ CTGAN 完成。")

# =============================================================================
# 5. 训练 Baseline B: TVAE
# =============================================================================
print("\n========================================")
print(f"   开始训练 TVAE (Epochs={EPOCHS})")
print("========================================")

tvae = TVAESynthesizer(
    metadata,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    cuda=True
)

tvae.fit(train_df)

# 保存模型
tvae_path = os.path.join(OUTPUT_DIR, 'tvae_model.pkl')
if os.path.exists(tvae_path):
    os.remove(tvae_path)
tvae.save(tvae_path)

synthetic_data_tvae = tvae.sample(num_rows=len(train_df))
synthetic_data_tvae.to_csv(os.path.join(OUTPUT_DIR, 'synthetic_tvae.csv'), index=False)
print("✅ TVAE 完成。")

# =============================================================================
# 6. 自动化评估
# =============================================================================
print("\n--- [Step 6] 生成评估报告 ---")

print("正在计算 CTGAN 得分...")
report_ctgan = evaluate_quality(
    real_data=train_df,
    synthetic_data=synthetic_data_ctgan,
    metadata=metadata,
    verbose=False
)

print("正在计算 TVAE 得分...")
report_tvae = evaluate_quality(
    real_data=train_df,
    synthetic_data=synthetic_data_tvae,
    metadata=metadata,
    verbose=False
)

print("\n" + "="*50)
print("   BASELINE 实验结果汇总")
print("="*50)
print(f"CTGAN Overall Score: {report_ctgan.get_score():.4f}")
print(f"TVAE  Overall Score: {report_tvae.get_score():.4f}")
print("="*50)

results = {
    "CTGAN": report_ctgan.get_score(),
    "TVAE": report_tvae.get_score()
}
with open(os.path.join(OUTPUT_DIR, 'scores.json'), 'w') as f:
    json.dump(results, f)

print(f"\n所有结果已保存至: {OUTPUT_DIR}")