import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import platform
import os
from matplotlib import font_manager

# ==========================================
# 0. 基础配置与智能字体
# ==========================================
def set_chinese_font():
    system_name = platform.system()
    font_name = None
    if system_name == "Windows":
        fonts = ["SimHei", "Microsoft YaHei", "SimSun"]
    elif system_name == "Darwin":
        fonts = ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS"]
    else:
        fonts = ["Droid Sans Fallback", "WenQuanYi Micro Hei", "Noto Sans CJK JP"]
    
    available_fonts = set([f.name for f in font_manager.fontManager.ttflist])
    for f in fonts:
        if f in available_fonts:
            font_name = f
            break
            
    if font_name:
        print(f"✅ 已设置中文字体: {font_name}")
        plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_theme(style="whitegrid", font=font_name)
    else:
        print("⚠️ 未找到常用中文字体，使用默认字体。")

set_chinese_font()
warnings.filterwarnings('ignore')

# 创建新文件夹用于存放图片
output_dir = 'advanced_plots_v2'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"📂 已创建新文件夹: {output_dir}")

# ==========================================
# 1. 加载数据
# ==========================================
file_path = 'NYC_Crashes_2017_Clean_Causal.csv'
try:
    df = pd.read_csv(file_path)
    print("✅ 数据加载成功，开始绘图...")

    # ==========================================
    # 图 1: 环境对事故严重程度的影响
    # ==========================================
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    surf_order = ['dry', 'wet', 'snowy', 'icy']
    valid_order = [x for x in surf_order if x in df['ROAD_SURFACE'].unique()]
    
    sns.pointplot(data=df, x='ROAD_SURFACE', y='NUMBER OF PERSONS INJURED', 
                  order=valid_order, capsize=.1, color='firebrick', ax=ax[0])
    ax[0].set_title('不同路面状况下的平均受伤人数 (带95%置信区间)', fontsize=12, fontweight='bold')
    ax[0].set_xlabel('路面状况')
    ax[0].set_ylabel('平均受伤人数')

    speed_vals = [25, 30, 35, 40, 45, 50, 55, 65]
    speed_df = df[df['SPEED_LIMIT'].isin(speed_vals)]
    if not speed_df.empty:
        fatality_rate = speed_df.groupby('SPEED_LIMIT')['NUMBER OF PERSONS KILLED'].mean() * 1000
        sns.barplot(x=fatality_rate.index, y=fatality_rate.values, palette="Reds", ax=ax[1])
        ax[1].set_title('不同限速下的死亡率 (每1000起事故)', fontsize=12, fontweight='bold')
        ax[1].set_xlabel('限速 (mph)')
        ax[1].set_ylabel('死亡人数 / 1000起')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/1_env_severity_impact.png', dpi=300)
    print(f"✅ 图 1 已保存至 {output_dir}")

    # ==========================================
    # 图 2: 环境对事故致因的诱导
    # ==========================================
    top_causes = df['PRIMARY_CAUSE'].value_counts().nlargest(5).index.tolist()
    if 'Pavement Slippery' not in top_causes:
        top_causes.append('Pavement Slippery')
    
    subset = df[df['PRIMARY_CAUSE'].isin(top_causes)]
    cause_surf_ct = pd.crosstab(subset['ROAD_SURFACE'], subset['PRIMARY_CAUSE'], normalize='index') * 100
    cause_surf_ct = cause_surf_ct.reindex([x for x in surf_order if x in cause_surf_ct.index])

    ax = cause_surf_ct.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='viridis')
    plt.title('不同路面状况下事故主因的构成变化', fontsize=14, fontweight='bold')
    plt.xlabel('路面状况')
    plt.ylabel('占比 (%)')
    plt.legend(title='事故主因', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/2_surface_cause_shift.png', dpi=300)
    print(f"✅ 图 2 已保存至 {output_dir}")

    # ==========================================
    # 图 3: 时空热力图
    # ==========================================
    plt.figure(figsize=(14, 8))
    heatmap_data = pd.crosstab(df['HOUR'], df['BOROUGH'], normalize='columns')
    sns.heatmap(heatmap_data, cmap="YlOrRd", annot=False, fmt=".1%", linewidths=.5)
    plt.title('各行政区事故发生时间的相对密度热力图', fontsize=14, fontweight='bold')
    plt.xlabel('行政区')
    plt.ylabel('小时 (0-23)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/3_spatiotemporal_heatmap.png', dpi=300)
    print(f"✅ 图 3 已保存至 {output_dir}")
    
    print("\n🎉 所有图表绘制完成！")

except FileNotFoundError:
    print(f"❌ 错误: 找不到文件 {file_path}")