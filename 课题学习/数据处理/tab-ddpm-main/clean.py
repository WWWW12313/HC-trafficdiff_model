import torch
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import OrdinalEncoder

from tab_ddpm.modules import MLPDiffusion
from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion

def generate_and_decode_accidents(num_samples=10000, batch_size=2048, output_name="clean_synthetic_accidents.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 准备在 {device} 上生成并解码虚拟数据...")

    # ==========================================
    # 🌟 第一步：重建文本解码器 (OrdinalEncoder)
    # ==========================================
    print("📚 正在读取原始数据重建文本词典...")
    orig_df = pd.read_csv("nyc_2017_pristine_v8.csv")
    cols_to_drop = ['COLLISION_ID', 'LOCATION', 'CRASH DATE', 'CRASH TIME', 'CRASH_FULL_TIME', 'coco']
    orig_df = orig_df.drop(columns=[c for c in cols_to_drop if c in orig_df.columns])
    
    # 找到所有文本分类列
    cat_cols = orig_df.select_dtypes(exclude=['float64', 'int64', 'int32']).columns
    encoder = OrdinalEncoder()
    # 拟合并记住文本到数字的映射关系
    if len(cat_cols) > 0:
        encoder.fit(orig_df[cat_cols].astype(str))
    
    # ==========================================
    # 🌟 第二步：加载模型与生成数据 (和之前类似)
    # ==========================================
    meta_info = joblib.load("causal_scaler.pkl")
    scaler = meta_info['scaler']
    topological_order = meta_info['topological_order']
    num_features = len(topological_order)

    model = MLPDiffusion(d_in=num_features, num_classes=0, is_y_cond=False, rtdl_params={'d_layers': [512, 512, 512], 'dropout': 0.0}).to(device)
    diffusion = GaussianMultinomialDiffusion(num_classes=np.array([], dtype=int), num_numerical_features=num_features, denoise_fn=model, num_timesteps=1000).to(device)
    
    diffusion.load_state_dict(torch.load("causal_ddpm_final.pt", map_location=device, weights_only=False), strict=False)
    diffusion.eval()

    all_samples = []
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch = min(batch_size, num_samples - i)
            print(f"⏳ 采样中: {i} / {num_samples}")
            z_norm = torch.randn((current_batch, num_features), device=device)
            x_gen_num = diffusion.gaussian_ddim_sample(z_norm, T=diffusion.num_timesteps, out_dict={})
            all_samples.append(x_gen_num.cpu().numpy())

    generated_data = np.vstack(all_samples)

    # ==========================================
    # 🌟 第三步：反归一化、清洗与文本还原
    # ==========================================
    print("🔄 正在还原物理量级并清洗数据...")
    data_0_to_1 = np.clip((generated_data + 1.0) / 2.0, 0.0, 1.0) 
    restored_data = scaler.inverse_transform(data_0_to_1)
    df_restored = pd.DataFrame(restored_data, columns=topological_order)
    
    # 🧹 清洗：暴力干掉所有因为 NaN 导致的空行
    initial_len = len(df_restored)
    df_restored = df_restored.dropna()
    print(f"🗑️ 过滤了 {initial_len - len(df_restored)} 行由于数值不稳定产生的无效空数据。")

    # 🔠 还原文本：把数字翻译回真实的类别（如路名、事故原因）
    if len(cat_cols) > 0:
        for col in cat_cols:
            if col in df_restored.columns:
                # 扩散模型输出的是浮点数，必须先四舍五入，并限制在词典范围内
                max_class_idx = len(encoder.categories_[list(cat_cols).index(col)]) - 1
                df_restored[col] = df_restored[col].round().clip(lower=0, upper=max_class_idx)
        
        # 批量反向转换为文本
        df_restored[cat_cols] = encoder.inverse_transform(df_restored[cat_cols])

    # 🔢 化整：处理必然是整数的数值列
    int_columns = ['NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PERSONS INJURED', 'TOTAL_VEHICLES', 'OSM_LANES_TAG']
    for col in int_columns:
        if col in df_restored.columns:
            df_restored[col] = df_restored[col].round().clip(lower=0)

    df_restored.to_csv(output_name, index=False)
    print(f"🎉 完美的文本数据已生成至: {output_name}")

if __name__ == "__main__":
    # 先生成 5000 条看看纯净版的数据效果
    generate_and_decode_accidents(num_samples=5000)