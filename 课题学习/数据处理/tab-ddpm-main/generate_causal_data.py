import torch
import pandas as pd
import numpy as np
import os
import joblib # 🌟 新增：用于加载归一化器

from tab_ddpm.modules import MLPDiffusion
from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion

def generate_synthetic_accidents(num_samples=100000, batch_size=4096, output_name="synthetic_nyc_accidents.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 准备在 {device} 上生成 {num_samples} 条虚拟事故数据...")

    # 1. 🌟 修改：加载强大的分位数归一化器和拓扑排序
    if not os.path.exists("causal_scaler.pkl"):
        print("❌ 找不到 causal_scaler.pkl！请确认它在当前目录下。")
        return
        
    meta_info = joblib.load("causal_scaler.pkl")
    scaler = meta_info['scaler']
    topological_order = meta_info['topological_order']
    num_features = len(topological_order)

    # 2. 初始化网络架构 (必须与训练时完全咬合)
    model = MLPDiffusion(
        d_in=num_features, num_classes=0, is_y_cond=False,
        rtdl_params={'d_layers': [512, 512, 512], 'dropout': 0.0}
    ).to(device)
    
    diffusion = GaussianMultinomialDiffusion(
        num_classes=np.array([], dtype=int), 
        num_numerical_features=num_features,
        denoise_fn=model,
        num_timesteps=1000 
    ).to(device)

    # 3. 注入灵魂：加载训练好的权重
    model_path = "causal_ddpm_final.pt"
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型权重文件 {model_path}！")
        return
    
    # 🌟 兼容旧模型：加载state_dict并处理缺失的key
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    
    # 如果旧模型缺少新添加的buffer，跳过这些key
    missing_keys, unexpected_keys = diffusion.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"⚠️  模型权重缺少以下key（将使用默认值）: {missing_keys}")
    if unexpected_keys:
        print(f"⚠️  模型权重有未预期的key（将被忽略）: {unexpected_keys}")
    
    diffusion.eval()
    print(f"✅ 成功加载因果扩散模型权重: {model_path}")

    # 4. 逆向扩散生成假数据 (分批进行防爆显存)
    all_samples = []
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch = min(batch_size, num_samples - i)
            print(f"⏳ 正在逆向去噪采样... 已生成: {i} / {num_samples}")
            
            # 🌟 无条件采样：直接调用 gaussian_ddim_sample
            # 从标准正态分布采样初始噪声
            z_norm = torch.randn((current_batch, num_features), device=device)
            # DDIM 采样（更快更稳定）
            x_gen_num = diffusion.gaussian_ddim_sample(z_norm, T=diffusion.num_timesteps, out_dict={})
            all_samples.append(x_gen_num.cpu().numpy())

    generated_data = np.vstack(all_samples)
    print("✨ 原始张量生成完毕！")

    # 5. 🌟 修改：反归一化 (核心逻辑变更)
    print("🔄 正在执行反归一化，恢复真实世界物理量级...")
    # 第一步：把模型的输出从 [-1, 1] 线性映射回 [0, 1]
    data_0_to_1 = (generated_data + 1.0) / 2.0
    # 限制在 [0, 1] 范围内，防止模型偶尔抽风输出越界值报错
    data_0_to_1 = np.clip(data_0_to_1, 0.0, 1.0) 
    
    # 第二步：使用 QuantileTransformer 逆向还原真实分布
    restored_data = scaler.inverse_transform(data_0_to_1)
    
    df_restored = pd.DataFrame(restored_data, columns=topological_order)
    
    # 🌟 新增：解码分类变量并应用物理约束字典字典映射！
    print("🔄 正在基于物理约束规则解码和修复分类变量...")
    if 'encoders' in meta_info and 'cat_cols' in meta_info:
        encoders = meta_info['encoders']
        cat_cols = meta_info['cat_cols']
        for col in cat_cols:
            if col in df_restored.columns and col in encoders:
                # 约束到合法的类别索引范围内
                max_val = len(encoders[col].categories_[0]) - 1
                cat_indices = np.round(df_restored[col]).clip(0, max_val).astype(int)
                df_restored[col] = encoders[col].inverse_transform(cat_indices.values.reshape(-1, 1)).flatten()

    # ==========================================
    # 🌟 核心增强：物理约束与因果逻辑后处理
    # ==========================================
    # 1. 位置与道路逻辑修复 (KNN 映射到真实底图)
    print("🌍 正在将生成坐标与街道对齐到纽约真实地图路网...")
    if os.path.exists("nyc_2017_pristine_v8.csv"):
        real_df = pd.read_csv("nyc_2017_pristine_v8.csv")
        # 清理原始数据的地理位置
        real_coords_df = real_df[['LATITUDE', 'LONGITUDE', 'ON STREET NAME', 'CROSS STREET NAME', 'OFF STREET NAME', 'BOROUGH', 'ZIP CODE']].dropna(subset=['LATITUDE', 'LONGITUDE'])
        
        # 将生成的坐标限制在纽约范围
        if 'LATITUDE' in df_restored.columns and 'LONGITUDE' in df_restored.columns:
            df_restored['LATITUDE'] = df_restored['LATITUDE'].clip(40.48, 40.92)
            df_restored['LONGITUDE'] = df_restored['LONGITUDE'].clip(-74.26, -73.69)
            
            # 使用 BallTree 进行最近邻查询匹配真实的关联属性
            from sklearn.neighbors import BallTree
            print("   -> 构建空间索引进行路网匹配...")
            tree = BallTree(np.radians(real_coords_df[['LATITUDE', 'LONGITUDE']].values), metric='haversine')
            _, ind = tree.query(np.radians(df_restored[['LATITUDE', 'LONGITUDE']].values), k=1)
            
            # 强制覆盖为真实且关联的街道与区号，保证生成街道相互联通且地名有效！
            spatial_cols = ['ON STREET NAME', 'CROSS STREET NAME', 'OFF STREET NAME', 'BOROUGH', 'ZIP CODE']
            idx_1d = ind.flatten()
            for col in spatial_cols:
                if col in df_restored.columns and col in real_coords_df.columns:
                    df_restored[col] = real_coords_df.iloc[idx_1d][col].values
            
            # 也可以覆盖坐标以使其完全落在路上（可选，如果需要严格真实地理映射的话）
            df_restored['LATITUDE'] = real_coords_df.iloc[idx_1d]['LATITUDE'].values
            df_restored['LONGITUDE'] = real_coords_df.iloc[idx_1d]['LONGITUDE'].values

    # 2. 事故连带效应修复 (Unspecified 和 Cause Transfer)
    print("🚘 正在处理事故因果传导与环境约束...")
    causes = [f'CONTRIBUTING FACTOR VEHICLE {i}' for i in range(1, 6)]
    valid_causes = [c for c in causes if c in df_restored.columns]
    
    def fix_accident_logic(row):
        # 让车辆数与存在原因的位置对应起来
        # 如果第一辆车是Unspecified，大概率所有车都是Unspecified
        c1 = valid_causes[0] if len(valid_causes) > 0 else None
        if c1 and row[c1] == 'Unspecified':
            for c in valid_causes[1:]:
                row[c] = 'Unspecified'
        else:
            # 如果后车有责任，前车通常会有 Other Vehicular 进行传导
            c2 = valid_causes[1] if len(valid_causes) > 1 else None
            if c1 and c2 and row[c2] not in ['Unspecified', 'nan', '', np.nan]:
                if row[c1] in ['Unspecified', 'nan', '', np.nan]:
                    row[c1] = 'Other Vehicular'
        
        # 3. 环境天气物理约束
        if 'TEMP_C' in row and 'WEATHER_CONDITION' in row:
            # 高温不可能下雪
            if row['TEMP_C'] > 5.0 and 'Snow' in str(row['WEATHER_CONDITION']):
                row['WEATHER_CONDITION'] = 'Clear/Cloudy'
                
        return row
    
    df_restored = df_restored.apply(fix_accident_logic, axis=1)

    # 4. 常数与类型矫正
    int_columns = ['NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED', 
                   'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED',
                   'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED',
                   'NUMBER OF MOTORIST INJURED', 'NUMBER OF MOTORIST KILLED', 
                   'TOTAL_VEHICLES']
    for col in int_columns:
        if col in df_restored.columns:
            df_restored[col] = df_restored[col].round().clip(lower=0) 

    if 'OSM_LANES_TAG' in df_restored.columns:
        df_restored['OSM_LANES_TAG'] = df_restored['OSM_LANES_TAG'].round().clip(1, 7)
    
    if 'REAL_SPEED_LIMIT' in df_restored.columns:
        # 限速必须是 5 的倍数
        df_restored['REAL_SPEED_LIMIT'] = (np.round(df_restored['REAL_SPEED_LIMIT'] / 5) * 5).clip(20, 65)

    if 'OSM_SPEED_TAG' in df_restored.columns:
        df_restored['OSM_SPEED_TAG'] = (np.round(df_restored['OSM_SPEED_TAG'] / 5) * 5).clip(10, 65)

    # ==========================================
    # 🌟 列顺序恢复：让它和原始数据集对齐！（解决“有关信息不在一起”的问题）
    # ==========================================
    if os.path.exists("nyc_2017_pristine_v8.csv"):
        real_df_cols = pd.read_csv("nyc_2017_pristine_v8.csv", nrows=0).columns
        ordered_cols = [c for c in real_df_cols if c in df_restored.columns]
        df_restored = df_restored[ordered_cols]

    # 6. 保存最终成果
    df_restored.to_csv(output_name, index=False)
    print(f"🎉 大功告成！生成的因果约束数据已保存至: {output_name}")
    
    return df_restored

def restore_crash_datetime(df: pd.DataFrame, target_year: int = 2018, drop_aux: bool = False) -> pd.DataFrame:
    out = df.copy()

    if "CRASH_HOUR" not in out.columns and {"CRASH_HOUR_SIN", "CRASH_HOUR_COS"}.issubset(out.columns):
        angle = np.arctan2(out["CRASH_HOUR_SIN"], out["CRASH_HOUR_COS"])
        out["CRASH_HOUR"] = np.round((angle % (2 * np.pi)) * 24 / (2 * np.pi)).astype(int) % 24

    hour = np.clip(np.round(out.get("CRASH_HOUR", 0)).astype(int), 0, 23)
    minute = np.clip(np.round(out.get("CRASH_MINUTE", 0)).astype(int), 0, 59)

    if {"CRASH_MONTH", "CRASH_DAY"}.issubset(out.columns):
        month = np.clip(np.round(out["CRASH_MONTH"]).astype(int), 1, 12)
        day = np.clip(np.round(out["CRASH_DAY"]).astype(int), 1, 31)
        dt = pd.to_datetime(
            dict(year=target_year, month=month, day=day, hour=hour, minute=minute),
            errors="coerce",
        )
        fallback = pd.to_datetime(
            dict(year=target_year, month=month, day=1, hour=hour, minute=minute),
            errors="coerce",
        )
        dt = dt.fillna(fallback)
    else:
        doy = np.clip(np.round(out.get("CRASH_DOY", 1)).astype(int), 1, 366)
        dt = pd.to_datetime(f"{target_year}-01-01") + pd.to_timedelta(doy - 1, unit="D")
        dt = dt + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")

    out["CRASH DATE"] = dt.dt.strftime("%m/%d/%Y")
    out["CRASH TIME"] = dt.dt.strftime("%H:%M:%S")

    if drop_aux:
        drop_cols = [
            "CRASH_MONTH", "CRASH_DAY", "CRASH_WEEKDAY", "CRASH_HOUR", "CRASH_MINUTE",
            "CRASH_DOY", "CRASH_HOUR_SIN", "CRASH_HOUR_COS", "CRASH_DOY_SIN", "CRASH_DOY_COS"
        ]
        out = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")

    return out

