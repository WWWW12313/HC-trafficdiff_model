import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

# 修复了最新版 causal-learn 的导入路径问题
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode

def build_human_vehicle_env_graph(csv_path="nyc_2017_pristine_v8.csv"):
    print(f"1. 加载并检查数据: {csv_path} ...")
    df = pd.read_csv(csv_path)
    
    # 抽取子集 (FCI 计算复杂度极高，3000条足够用来发现底层统计因果规律)
    df_sample = df.sample(min(3000, len(df)), random_state=42).copy()
    
    # =========================================================
    # 1.1 必须剔除无法参与因果计算的纯文本和ID列 (包括全空的 VISIBILITY_KM)
    # =========================================================
    cols_to_drop = ['COLLISION_ID', 'LOCATION', 'CRASH DATE', 'CRASH TIME', 'CRASH_FULL_TIME', 'coco', 'VISIBILITY_KM']
    df_sample = df_sample.drop(columns=[c for c in cols_to_drop if c in df_sample.columns])
    
    # =========================================================
    # 1.2 严格的数据类型转换与安全兜底 (防止遗漏的NaN引发报错)
    # =========================================================
    numeric_cols = df_sample.select_dtypes(include=['float64', 'int64', 'int32']).columns
    cat_cols = df_sample.select_dtypes(exclude=['float64', 'int64', 'int32']).columns
    
    # 数值列用中位数填补 (兜底)
    for col in numeric_cols:
        median_val = df_sample[col].median()
        if pd.isna(median_val): median_val = 0.0
        df_sample[col] = df_sample[col].fillna(median_val)
        
    # 分类列填补并强制转为字符串
    for col in cat_cols:
        df_sample[col] = df_sample[col].fillna('Unknown').astype(str)

    # =========================================================
    # 1.3 离散变量编码
    # =========================================================
    print("-> 正在将分类/布尔特征转换为数值 (Ordinal Encoding)...")
    encoder = OrdinalEncoder()
    if len(cat_cols) > 0:
        df_sample[cat_cols] = encoder.fit_transform(df_sample[cat_cols])
    
    # =========================================================
    # 1.4 终极安全防线：清除 Inf 和 NaN，并转为 float64
    # =========================================================
    df_sample = df_sample.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_sample = df_sample.astype(np.float64)
    columns = list(df_sample.columns)
    
    print(f"-> 数据准备完毕，最终参与因果发现的特征数量: {len(columns)}，完全数值化。")

    # =========================================================
    # 2. 定义【人-车-环境-结果】四大物理层级
    # =========================================================
    print("2. 正在根据物理法则划分节点层级...")
    
    # 第 1 层：客观环境 (Environment) - 永远是原因 (已移除 VISIBILITY_KM)
    env_vars = ['TEMP_C', 'prcp', 'WIND_SPEED_KMH', 'CRASH_MONTH', 'CRASH_HOUR', 
                'DIST_TO_SIGNAL_M', 'REAL_SPEED_LIMIT', 'INFERRED_LANES', 'HAS_TRAFFIC_SIGNAL', 'OSM_TYPE']
    
    # 第 2 层：车辆状态与驾驶人行为 (Human & Vehicle) - 受环境影响，并导致事故
    veh_human_vars = ['CONTRIBUTING FACTOR VEHICLE 1', 'CONTRIBUTING FACTOR VEHICLE 2', 
                      'VEHICLE TYPE CODE 1', 'VEHICLE TYPE CODE 2', 'TOTAL_VEHICLES', 'IS_MULTI_VEHICLE']
    
    # 第 3 层：事故结果 (Outcomes) - 永远是结果，不能导致前两层发生
    outcome_vars = ['NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED', 
                    'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF MOTORIST INJURED']
    
    # 获取各个变量在数据矩阵中的列索引 (Index)
    env_indices = [columns.index(c) for c in env_vars if c in columns]
    veh_human_indices = [columns.index(c) for c in veh_human_vars if c in columns]
    outcome_indices = [columns.index(c) for c in outcome_vars if c in columns]

    # =========================================================
    # 3. 注入背景知识 (Background Knowledge)
    # =========================================================
    print("3. 注入因果约束法则：禁止结果导致原因...")
    bk = BackgroundKnowledge()
    
    cg_nodes = [GraphNode(f"X{i+1}") for i in range(len(columns))]
    
    for outcome_idx in outcome_indices:
        # 法则 A：事故结果 不能导致 客观环境
        for env_idx in env_indices:
            bk.add_forbidden_by_node(cg_nodes[outcome_idx], cg_nodes[env_idx])
        # 法则 B：事故结果 不能导致 车辆/人的状态
        for vh_idx in veh_human_indices:
            bk.add_forbidden_by_node(cg_nodes[outcome_idx], cg_nodes[vh_idx])
            
    for vh_idx in veh_human_indices:
        # 法则 C：车辆/人的状态 不能导致 客观环境
        for env_idx in env_indices:
            bk.add_forbidden_by_node(cg_nodes[vh_idx], cg_nodes[env_idx])

    # =========================================================
    # 4. 运行带有背景知识约束的 FCI 算法
    # =========================================================
    print("4. 正在运行 FCI 算法寻找隐变量和因果关系 (大概需要 1-3 分钟)...")
    data_np = df_sample.values
    
    # alpha 是显著性水平。加入先验知识后，算法不会乱搜
    G, edges = fci(data_np, alpha=0.01, background_knowledge=bk, verbose=False)
    
    print("\n>>> 因果矩阵计算完成！")
    causal_adj_matrix = G.graph
    
    # 打印发现的因果关系
    print("\n【FCI 发现的核心因果关系】(A --> B 表示 A 是 B 的原因):")
    found_edges = 0
    for i in range(causal_adj_matrix.shape[0]):
        for j in range(causal_adj_matrix.shape[1]):
            # 在 causallearn 中: adj[i,j]==-1 且 adj[j,i]==1 表示 i -> j
            if causal_adj_matrix[i, j] == -1 and causal_adj_matrix[j, i] == 1:
                print(f"[{columns[i]}]  -->  [{columns[j]}]")
                found_edges += 1
                
    if found_edges == 0:
        print("未发现符合显著性条件的强因果关系。可尝试调高 alpha 值 (如 alpha=0.05)。")

    return causal_adj_matrix, columns, G

if __name__ == "__main__":
    # 直接读取刚才洗好的完美数据
    adj_matrix, cols, graph = build_human_vehicle_env_graph("nyc_2017_pristine_v8.csv")