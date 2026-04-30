import pandas as pd
import numpy as np
import warnings
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

warnings.filterwarnings("ignore")

def preprocess_causal_v8(file_path):
    print(f"1. 加载包含 OSM & MetaWeather 数据的 V8 数据集: {file_path}")
    df = pd.read_csv(file_path)
    
    # =========================================================
    # 步骤 A：时间特征提取 (转为有物理规律的周期特征)
    # =========================================================
    print("2. 正在提取时间周期特征...")
    if 'CRASH DATE' in df.columns:
        df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'], errors='coerce')
        df['CRASH_MONTH'] = df['CRASH DATE'].dt.month
        df['CRASH_DAY'] = df['CRASH DATE'].dt.day
        df['CRASH_DAYOFWEEK'] = df['CRASH DATE'].dt.dayofweek
    
    if 'CRASH TIME' in df.columns:
        df['CRASH_HOUR'] = df['CRASH TIME'].astype(str).apply(
            lambda x: int(x.split(':')[0]) if ':' in x else 0
        )

    # =========================================================
    # 步骤 B：极简剔除 (仅删除导致报错或绝对冗余的列)
    # =========================================================
    cols_to_drop_strictly = [
        'COLLISION_ID', 'LOCATION', 'CRASH DATE', 'CRASH TIME', 'CRASH_FULL_TIME', 'coco'
    ]
    cols_to_drop = [col for col in cols_to_drop_strictly if col in df.columns]
    df_clean = df.drop(columns=cols_to_drop)
    print(f"3. 冗余列清理完毕。当前保留的特征总维度: {df_clean.shape[1]} 列")

    # =========================================================
    # 步骤 C：硬编码区分连续与离散特征，并执行因果逻辑填补
    # =========================================================
    print("4. 执行基于因果逻辑的数据划分与精细化缺失值填补...")
    
    # 1. 手动定义真正的“连续/数值型变量”
    true_numeric_cols = [
        'LATITUDE', 'LONGITUDE', 
        'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED',
        'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED',
        'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED',
        'NUMBER OF MOTORIST INJURED', 'NUMBER OF MOTORIST KILLED',
        'TEMP_C', 'prcp', 'WIND_SPEED_KMH', 'VISIBILITY_KM', 
        'DIST_TO_SIGNAL_M', 'REAL_SPEED_LIMIT', 'INFERRED_LANES', 'TOTAL_VEHICLES'
    ]
    true_numeric_cols = [c for c in true_numeric_cols if c in df_clean.columns]

    # 2. 剩下的全部视为“离散/分类变量” (包含ZIP CODE、布尔值、车型、街道等)
    true_categorical_cols = [c for c in df_clean.columns if c not in true_numeric_cols]

    # 3. 连续变量处理：用中位数填补
    for col in true_numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # 4. 离散变量处理：基于业务逻辑的因果填补
    for col in true_categorical_cols:
        # 多车特征处理 (如果为空，说明这辆车不存在，而不是未知)
        if 'VEHICLE TYPE CODE' in col or 'CONTRIBUTING FACTOR' in col:
            df_clean[col] = df_clean[col].fillna('None_Involved').astype(str)
            
        # 街道名称处理 (如果为空，说明事故发生在路段中间，没有交叉口)
        elif 'CROSS STREET NAME' in col or 'OFF STREET NAME' in col:
            df_clean[col] = df_clean[col].fillna('Midblock_No_Intersection').astype(str)
            
        # OSM补充特征处理
        elif col in ['HAS_TRAFFIC_SIGNAL', 'OSM_ONEWAY', 'HAS_DIVIDER', 'IS_MULTI_VEHICLE']:
            df_clean[col] = df_clean[col].fillna('OSM_Missing').astype(str)
            
        # 其他常规离散特征 (如 ZIP CODE, BOROUGH, 天气状态等)
        else:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                # 处理像 ZIP CODE 这种本质是数字的分类列，防止变成浮点数文本 '11229.0'
                df_clean[col] = df_clean[col].fillna(-999).astype(int).astype(str).replace('-999', 'Unspecified')
            else:
                df_clean[col] = df_clean[col].fillna('Unspecified').astype(str)

    print(f"-> 成功识别并处理 {len(true_numeric_cols)} 个连续特征和 {len(true_categorical_cols)} 个离散特征")
    return df_clean

def train_and_generate_tabddpm(df_train, n_generate=1000, output_path="synthetic_2017_causal_tabddpm.csv"):
    print("\n5. 转换数据格式以适配 Synthcity TabDDPM...")
    loader = GenericDataLoader(df_train)
    
    print("6. 初始化 TabDDPM (Diffusion) 模型...")
    # n_iter=2000 为默认，如果需要快速测试可临时改为 500 或 1000
    model = Plugins().get("ddpm", n_iter=2000, batch_size=256) 
    
    print("7. 开始训练模型 (包含全量高基数特征，请耐心等待)...")
    model.fit(loader)
    print(">>> 模型训练完成！")
    
    print(f"8. 正在生成 {n_generate} 条合成事故记录...")
    synthetic_data = model.generate(count=n_generate).dataframe()
    
    # =========================================================
    # 步骤 D：物理常识后处理 (修正扩散模型生成的浮点误差)
    # =========================================================
    print("9. 应用后处理规则 (确保物理逻辑自洽)...")
    
    # 确保人数、车道数、车辆数为整数且不小于 0
    count_keywords = ['NUMBER OF', 'TOTAL_VEHICLES', 'INFERRED_LANES', 'REAL_SPEED_LIMIT']
    for col in synthetic_data.columns:
        if any(keyword in col for keyword in count_keywords):
            if synthetic_data[col].dtype in ['float64']:
                synthetic_data[col] = synthetic_data[col].round().clip(lower=0)
                
    # 确保距红绿灯距离、能见度、降水、风速等不能为负数
    non_negative_cols = ['DIST_TO_SIGNAL_M', 'VISIBILITY_KM', 'prcp', 'WIND_SPEED_KMH']
    for col in non_negative_cols:
        if col in synthetic_data.columns and synthetic_data[col].dtype in ['float64']:
            synthetic_data[col] = synthetic_data[col].clip(lower=0.0)

    # 保存最终结果
    synthetic_data.to_csv(output_path, index=False)
    print(f">>> 成功！生成的因果网络数据已保存至: {output_path}\n")
    
    return synthetic_data

if __name__ == "__main__":
    file_name = "nyc_2017_final_v8.csv"
    
    # 1. 执行精细化预处理
    df_preprocessed = preprocess_causal_v8(file_name)
    
    # ⚠️【安全提示】：
    # 包含全纽约所有街道名称的数据矩阵极其庞大，全量丢入大概率会导致内存溢出 (OOM)。
    # 强烈建议先用 2000 条数据进行全流程跑通。
    # 如果你的服务器显存足够 (如 24GB+)，且 5000 条测试成功，再逐步放开限制。
    sample_size = min(2000, len(df_preprocessed))
    df_train_sample = df_preprocessed.sample(n=sample_size, random_state=42)
    
    # 2. 训练并生成
    synthetic_df = train_and_generate_tabddpm(
        df_train=df_train_sample, 
        n_generate=sample_size, 
        output_path="synthetic_nyc_2017_v8_causal_tabddpm.csv"
    )
    
    # 3. 数据预览
    print("【生成数据预览 (部分核心连续特征)】")
    check_cols = ['TEMP_C', 'DIST_TO_SIGNAL_M', 'TOTAL_VEHICLES', 'NUMBER OF PERSONS INJURED']
    check_cols = [c for c in check_cols if c in synthetic_df.columns]
    print(synthetic_df[check_cols].describe().loc[['mean', 'min', 'max']])