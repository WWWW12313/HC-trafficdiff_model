import pandas as pd
import numpy as np

def filter_pristine_data(csv_path="nyc_2017_final_v8.csv", output_path="nyc_2017_pristine.csv"):
    print(f"1. 正在加载原始数据集: {csv_path} ...")
    df = pd.read_csv(csv_path)
    total_initial = len(df)
    print(f"-> 初始数据总条数: {total_initial} 条\n")

    # =========================================================
    # 步骤 A: 处理“结构性缺失” (合法的空值，不应被删除)
    # =========================================================
    # 1. 车辆特征：如果没有第二辆车，那叫 "None_Involved"，不是数据缺失
    veh_cols = ['CONTRIBUTING FACTOR VEHICLE 2', 'VEHICLE TYPE CODE 2']
    for col in veh_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None_Involved')
            
    # 2. 街道特征：路段中间没有交叉口，是正常物理现象
    street_cols = ['CROSS STREET NAME', 'OFF STREET NAME']
    for col in street_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Midblock_No_Intersection')

    # =========================================================
    # 步骤 B: 聚焦核心因果特征，进行严格的“真正缺失”检查
    # =========================================================
    # 🚨 核心改动 1：物理删除 100% 缺失的垃圾列
    if 'VISIBILITY_KM' in df.columns:
        df = df.drop(columns=['VISIBILITY_KM'])
        print("-> 已永久剔除 100% 缺失的列: VISIBILITY_KM")

    # 🚨 核心改动 2：从核心因果列表中移除 VISIBILITY_KM
    core_causal_cols = [
        # 环境层 (已移除 VISIBILITY_KM)
        'TEMP_C', 'prcp', 'WIND_SPEED_KMH', 
        'DIST_TO_SIGNAL_M', 'REAL_SPEED_LIMIT', 'INFERRED_LANES', 'HAS_TRAFFIC_SIGNAL', 
        # 车辆与人层
        'CONTRIBUTING FACTOR VEHICLE 1', 'VEHICLE TYPE CODE 1', 'TOTAL_VEHICLES', 
        # 结果层
        'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED'
    ]
    
    # 仅保留在数据集中实际存在的列
    core_causal_cols = [col for col in core_causal_cols if col in df.columns]
    # =========================================================
    # 步骤 C: 统计并剔除残缺数据
    # =========================================================
    print("2. 正在扫描核心因果特征的真正缺失情况...")
    
    # 创建一个布尔掩码：只要核心列中有任何一个是 NaN，就标记为 True (即残缺数据)
    missing_mask = df[core_causal_cols].isna().any(axis=1)
    
    # 拆分出“良好数据”和“残缺数据”
    df_pristine = df[~missing_mask].copy()
    df_missing_only = df[missing_mask].copy()
    
    total_pristine = len(df_pristine)
    total_missing = len(df_missing_only)
    missing_ratio = (total_missing / total_initial) * 100

    # =========================================================
    # 步骤 D: 打印详细的统计报告
    # =========================================================
    print("-" * 40)
    print("【数据质检与过滤报告】")
    print(f"总计输入数据 : {total_initial} 条")
    print(f"✅ 结构良好数据: {total_pristine} 条 (用于生成和训练)")
    print(f"❌ 剔除残缺数据: {total_missing} 条 (缺失率: {missing_ratio:.2f}%)")
    print("-" * 40)
    
    # 查看究竟是哪些列导致了数据被剔除 (找出罪魁祸首)
    if total_missing > 0:
        print("\n导致数据被剔除的主要原因 (各列缺失行数):")
        missing_counts = df_missing_only[core_causal_cols].isna().sum()
        # 过滤掉缺失数为0的列并降序排列
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
        for col, count in missing_counts.items():
            print(f" - {col}: 缺失 {count} 条")
            
    # 保存这批纯净的数据
    df_pristine.to_csv(output_path, index=False)
    print(f"\n>>> 过滤完成！结构良好的纯净数据已保存至: {output_path}")
    
    return df_pristine, df_missing_only

if __name__ == "__main__":
    # 运行过滤，并获取纯净的 dataframe
    pristine_data, bad_data = filter_pristine_data("nyc_2017_final_v8.csv", "nyc_2017_pristine_v8.csv")