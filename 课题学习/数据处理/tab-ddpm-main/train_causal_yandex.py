import os
import json
import time
import math
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Tuple

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer
from torch.utils.data import DataLoader, Dataset

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    swanlab = None
    SWANLAB_AVAILABLE = False

from tab_ddpm.modules import MLPDiffusion
from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from tab_ddpm.utils import index_to_log_onehot

# ==========================================
# 1. 全局因果边定义
# ==========================================
fci_discovered_edges = [
    # ==== 1. 基础设施与路网地理系统 ====
    ("BOROUGH", "ZIP CODE"),
    ("ZIP CODE", "LATITUDE"), ("ZIP CODE", "LONGITUDE"),
    ("LATITUDE", "LONGITUDE"),
    ("OSM_SPEED_TAG", "REAL_SPEED_LIMIT"),
    ("OSM_LANES_TAG", "INFERRED_LANES"),
    ("HAS_DIVIDER", "REAL_SPEED_LIMIT"),
    ("BOROUGH", "IS_OVERSIZED_VEHICLE"),
    ("REAL_SPEED_LIMIT", "IS_SPEEDING"),

    # ==== 2. 天气系统 ====
    ("TEMP_C", "REAL_WEATHER"),
    ("prcp", "REAL_WEATHER"),
    ("WIND_SPEED_KMH", "REAL_WEATHER"),
    ("REAL_WEATHER", "WEATHER_CONDITION"),
    ("WEATHER_CONDITION", "IS_VISION_OBSCURED"),
    ("WEATHER_CONDITION", "IS_POOR_ROAD_CONDITION"),
    ("WEATHER_CONDITION", "IS_ANIMAL_RELATED"),

    # ==== 3. 时间与多重动态影响 ====
    ("CRASH TIME", "HAS_TRAFFIC_SIGNAL"),
    ("CRASH TIME", "IS_VISION_OBSCURED"),
    ("CRASH TIME", "TOTAL_VEHICLES"),
    ("CRASH TIME", "IS_AGGRESSIVE_DRIVING"),
    ("CRASH TIME", "IS_ALCOHOL_INVOLVED"),
    ("CRASH_TIME_PERIOD", "TOTAL_VEHICLES"),
    ("CRASH_TIME_PERIOD", "IS_AGGRESSIVE_DRIVING"),
    ("CRASH_WEEKDAY", "IS_ALCOHOL_INVOLVED"),
    ("IS_WEEKEND", "IS_ALCOHOL_INVOLVED"),

    # ==== 4. 车辆因素与状况 ====
    ("IS_OVERSIZED_VEHICLE", "IS_VISION_OBSCURED"),
    ("IS_VEHICLE_DEFECT", "IS_SPEEDING"),
    ("VEHICLE TYPE CODE 1", "IS_VEHICLE_DEFECT"),
    ("VEHICLE TYPE CODE 1", "IS_OVERSIZED_VEHICLE"),

    # ==== 5. 驾驶员异常状态诱发事故模式 ====
    ("IS_INEXPERIENCED_DRIVER", "IS_DISTRACTED"),
    ("IS_FATIGUED", "IS_DISTRACTED"),
    ("IS_ALCOHOL_INVOLVED", "IS_AGGRESSIVE_DRIVING"),
    ("IS_ALCOHOL_INVOLVED", "IS_SPEEDING"),

    # ==== 6. 危险驾驶行为传导 ====
    ("IS_AGGRESSIVE_DRIVING", "IS_SPEEDING"),
    ("IS_AGGRESSIVE_DRIVING", "IS_FOLLOWING_TOO_CLOSE"),
    ("IS_AGGRESSIVE_DRIVING", "IS_IMPROPER_LANE_USE"),
    ("IS_DISTRACTED", "IS_FAILURE_TO_YIELD"),
    ("IS_DISTRACTED", "IS_TRAFFIC_SIGNAL_VIOLATION"),
    ("IS_SPEEDING", "IS_IMPROPER_TURNING"),
    ("HAS_TRAFFIC_SIGNAL", "IS_TRAFFIC_SIGNAL_VIOLATION"),
    ("INFERRED_LANES", "IS_IMPROPER_LANE_USE"),
    ("IS_TRAFFIC_SIGNAL_VIOLATION", "IS_MULTI_VEHICLE"),

    # ==== 7. 行人与非机动车 ====
    ("BOROUGH", "IS_PEDESTRIAN_CYCLIST_ERROR"),
    ("IS_VISION_OBSCURED", "IS_PEDESTRIAN_CYCLIST_ERROR"),

    # ==== 8. 事故形态影响最终伤亡结果 ====
    ("TOTAL_VEHICLES", "IS_MULTI_VEHICLE"),
    ("IS_SPEEDING", "NUMBER OF PERSONS INJURED"),
    ("IS_ALCOHOL_INVOLVED", "NUMBER OF PERSONS INJURED"),
    ("TOTAL_VEHICLES", "NUMBER OF PERSONS INJURED"),
    ("IS_PEDESTRIAN_CYCLIST_ERROR", "NUMBER OF PERSONS INJURED"),
    ("IS_MULTI_VEHICLE", "NUMBER OF PERSONS INJURED"),
    ("IS_FAILURE_TO_YIELD", "NUMBER OF PERSONS INJURED"),
    ("IS_VEHICLE_DEFECT", "NUMBER OF PERSONS INJURED")
]

# 与实验描述统一命名，后续可直接替换成 LLM 发现的因果边。
NEW_CAUSAL_EDGES = fci_discovered_edges


def _norm_col(name: str) -> str:
    return str(name).strip().upper().replace("_", " ")


def _get_datetime_series(df: pd.DataFrame) -> pd.Series:
    cols_upper = {c.upper(): c for c in df.columns}
    dt_col = cols_upper.get("CRASH DATETIME")
    if dt_col is not None:
        return pd.to_datetime(df[dt_col], errors="coerce")
    date_col = cols_upper.get("CRASH DATE")
    time_col = cols_upper.get("CRASH TIME")
    if date_col is not None and time_col is not None:
        return pd.to_datetime(
            df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip(),
            errors="coerce",
        )
    if date_col is not None:
        return pd.to_datetime(df[date_col], errors="coerce")
    return pd.Series(pd.NaT, index=df.index)


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """从日期/时间列提取 3 个分类时间特征（字符串），删除所有原始时间列。"""
    out = df.copy()
    dt = _get_datetime_series(out)

    crash_hour = dt.dt.hour.fillna(0).astype(int)
    crash_month = dt.dt.month.fillna(1).astype(int)
    crash_weekday = dt.dt.weekday.fillna(0).astype(int)

    # 季节（0:冬 Dec-Feb, 1:春 Mar-May, 2:夏 Jun-Aug, 3:秋 Sep-Nov）
    out["CRASH_SEASON"] = ((crash_month % 12) // 3).astype(str)

    # 时段（0:深夜 0-5, 1:早高峰 6-9, 2:日间 10-16, 3:晚高峰 17-20, 4:夜间 21-23）
    out["CRASH_TIME_PERIOD"] = pd.cut(
        crash_hour, bins=[-1, 5, 9, 16, 20, 23],
        labels=["0", "1", "2", "3", "4"], include_lowest=True,
    ).astype(str)

    # 周末（"1"=周末, "0"=工作日）
    out["IS_WEEKEND"] = crash_weekday.isin([5, 6]).astype(int).astype(str)

    # 删除所有原始时间列
    for c in ["CRASH DATE", "CRASH TIME", "CRASH DATETIME", "CRASH_FULL_TIME"]:
        if c in out.columns:
            out.drop(columns=[c], inplace=True)

    return out


def _edge_alias(col_name: str) -> str:
    n = _norm_col(col_name)
    if n == "CRASH DATE":
        return "CRASH_SEASON"
    if n == "CRASH TIME":
        return "CRASH_TIME_PERIOD"
    return col_name


# ==========================================
# 车型合并映射（高基数 → 7 类）
# ==========================================
_VEHICLE_TYPE_MAP = {
    'Sedan': 'Sedan',
    'Station Wagon/Sport Utility Vehicle': 'SUV',
    'Taxi': 'Taxi',
    'Pick-up Truck': 'Pickup',
    'Box Truck': 'Truck',
    'Bus': 'Bus',
    'Bike': 'Bike',
}

# 天气合并映射（9 → 5 类）
_WEATHER_MAP = {
    'Clear/Cloudy': 'Clear',
    'Light Rain': 'LightRain',
    'Rain': 'Rain',
    'Rain (Inferred)': 'Rain',
    'Heavy Rain': 'Rain',
    'Light Snow': 'Snow',
    'Snow': 'Snow',
    'Snow (Inferred)': 'Snow',
    'Fog': 'Fog',
}

# 道路类型 Top-5
_ROAD_TYPE_TOP = {'residential', 'secondary', 'primary', 'motorway', 'tertiary'}


def _consolidate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    合并高基数分类变量、移除文本/冗余列、将所有二元特征统一为 str dtype。
    调用后，所有离散特征均为 object/str 类型，方便后续 select_dtypes 自动检测。

    连续特征（保留为 float/int）：
        LATITUDE, LONGITUDE, TEMP_C, prcp, WIND_SPEED_KMH, DIST_TO_SIGNAL_M, REAL_SPEED_LIMIT
    分类特征（转为 str）：
        IS_WEEKEND, CRASH_SEASON, CRASH_TIME_PERIOD,
        VEHICLE TYPE CODE 1/2, REAL_WEATHER, OSM_TYPE,
        TOTAL_VEHICLES, INFERRED_LANES, HAS_TRAFFIC_SIGNAL, HAS_DIVIDER,
        OSM_ONEWAY, IS_MULTI_VEHICLE, 22× IS_XXX, 5× CAUSE_XXX
    """
    out = df.copy()

    # --- 1. 删除 ID / 稀疏 / 冗余列 ---
    drop_cols = [
        "COLLISION_ID",
        "VEHICLE TYPE CODE 3", "VEHICLE TYPE CODE 4", "VEHICLE TYPE CODE 5",
        "coco",               # 与 REAL_WEATHER 重复
        "OSM_SPEED_TAG",      # 文本，与 REAL_SPEED_LIMIT 重复
        "OSM_LANES_TAG",      # 与 INFERRED_LANES 重复
        "WEATHER_CONDITION",  # 与 REAL_WEATHER 重复
        # 伤亡子列（仅保留目标列 NUMBER OF PERSONS INJURED）
        "NUMBER OF PERSONS KILLED",
        "NUMBER OF PEDESTRIANS INJURED", "NUMBER OF PEDESTRIANS KILLED",
        "NUMBER OF CYCLIST INJURED", "NUMBER OF CYCLIST KILLED",
        "NUMBER OF MOTORIST INJURED", "NUMBER OF MOTORIST KILLED",
    ]
    out.drop(columns=[c for c in drop_cols if c in out.columns], inplace=True)

    # --- 2. 合并 VEHICLE TYPE CODE 1 & 2（40+ → 8 类别） ---
    for col in ["VEHICLE TYPE CODE 1", "VEHICLE TYPE CODE 2"]:
        if col in out.columns:
            out[col] = out[col].map(
                lambda x: (
                    "None"
                    if pd.isna(x) or str(x).strip() in ("", "nan", "None_Involved")
                    else _VEHICLE_TYPE_MAP.get(str(x).strip(), "Other")
                )
            )

    # --- 3. 合并 REAL_WEATHER（9 → 5 类别） ---
    if "REAL_WEATHER" in out.columns:
        out["REAL_WEATHER"] = out["REAL_WEATHER"].map(
            lambda x: _WEATHER_MAP.get(str(x).strip(), "Clear")
        )

    # --- 4. 合并 OSM_TYPE（23 → 6 类别） ---
    if "OSM_TYPE" in out.columns:
        out["OSM_TYPE"] = out["OSM_TYPE"].map(
            lambda x: str(x).strip() if str(x).strip() in _ROAD_TYPE_TOP else "other"
        )

    # --- 5. 截断低基数数值 → str（作为分类） ---
    if "TOTAL_VEHICLES" in out.columns:
        out["TOTAL_VEHICLES"] = out["TOTAL_VEHICLES"].clip(upper=4).astype(int).astype(str)
    if "INFERRED_LANES" in out.columns:
        out["INFERRED_LANES"] = out["INFERRED_LANES"].clip(upper=4).astype(int).astype(str)

    # --- 6. 所有二元 int/float/bool 特征 → str ---
    binary_like = ["HAS_TRAFFIC_SIGNAL", "HAS_DIVIDER", "OSM_ONEWAY"]
    is_cols = [c for c in out.columns if c.startswith("IS_")]
    binary_like.extend(is_cols)
    for col in binary_like:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int).astype(str)

    # --- 7. CAUSE_XXX → 二元 str ---
    cause_cols = [c for c in out.columns if c.startswith("CAUSE_")]
    for col in cause_cols:
        val = pd.to_numeric(out[col], errors="coerce").fillna(-1.0)
        out[col] = np.where(val > 0, "1", "0")

    return out


# ==========================================
# 2. 数据集类
# ==========================================
class CausalTabularDataset(Dataset):
    def __init__(self, csv_path, causal_edges):
        self.df = pd.read_csv(csv_path)
        self.df.columns = (
            self.df.columns.astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
        )

        # Step 1: 时间特征 → 3 个分类列，删除原始时间列
        self.df = _add_time_features(self.df)

        # Step 2: 地理信息最小化：移除路名等文本列，仅保留经纬度
        geo_drop_cols = [
            "LOCATION", "ON STREET NAME", "CROSS STREET NAME",
            "OFF STREET NAME", "ZIP CODE", "BOROUGH",
        ]
        existing_geo_drop = [c for c in geo_drop_cols if c in self.df.columns]
        if existing_geo_drop:
            self.df = self.df.drop(columns=existing_geo_drop)
            print(f"🗺️ 已移除位置标识列: {existing_geo_drop}")

        # Step 3: 特征合并（合并高基数、去冗余、二元→字符串）
        self.df = _consolidate_features(self.df)

        # Step 4: 目标列 Quantile 正态化
        self.target_col = None
        self.target_transformer = None
        target_col_candidates = {_norm_col(c): c for c in self.df.columns}
        self.target_col = target_col_candidates.get(_norm_col("NUMBER OF PERSONS INJURED"))
        if self.target_col is not None:
            y_raw = pd.to_numeric(self.df[self.target_col], errors="coerce").fillna(0.0).to_numpy().reshape(-1, 1)
            n_quantiles = min(1000, max(10, len(y_raw)))
            self.target_transformer = QuantileTransformer(
                output_distribution="normal", random_state=42, n_quantiles=n_quantiles,
            )
            y_trans = self.target_transformer.fit_transform(y_raw).reshape(-1)
            self.df[self.target_col] = y_trans.astype(np.float32)
            print(f"🎯 已对目标列 {self.target_col} 应用 Quantile 正态化。")

        # Step 5: 因果边映射
        norm_to_real = {_norm_col(c): c for c in self.df.columns}
        mapped_edges = []
        for s, t in (causal_edges or []):
            s_real = norm_to_real.get(_norm_col(_edge_alias(s)))
            t_real = norm_to_real.get(_norm_col(_edge_alias(t)))
            if s_real is not None and t_real is not None:
                mapped_edges.append((s_real, t_real))

        # Step 6: 识别分类列
        # _consolidate_features 已将所有离散特征转为 str，
        # 此处自动检测 object dtype 即可。
        all_cat_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()

        self.encoders = {}
        for col in all_cat_cols:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self.df[col] = enc.fit_transform(self.df[[col]].astype(str)).astype(int)
            self.encoders[col] = enc
        self.cat_cols = all_cat_cols

        print(f"🔍 分类列 ({len(self.cat_cols)}):")
        for col in self.cat_cols:
            print(f"  📝 {col}: {len(self.encoders[col].categories_[0])} 类别")

        # Step 7: 连续列转 float32
        num_cols = [c for c in self.df.columns if c not in self.cat_cols]
        for col in num_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0.0).astype(np.float32)
        print(f"📊 连续列 ({len(num_cols)}): {num_cols}")

        # Step 8: 构建 DAG + 拓扑排序
        current_cols = set(self.df.columns)
        mapped_edges = [(s, t) for s, t in mapped_edges if s in current_cols and t in current_cols]
        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.df.columns)
        self.G.add_edges_from(mapped_edges)
        if not nx.is_directed_acyclic_graph(self.G):
            for cycle in list(nx.simple_cycles(self.G)):
                if len(cycle) > 1 and self.G.has_edge(cycle[-1], cycle[0]):
                    self.G.remove_edge(cycle[-1], cycle[0])

        self.topological_order = list(nx.topological_sort(self.G))
        self.df_ordered = self.df[self.topological_order]

        # Step 9: 分离连续列和离散列索引（拓扑序）
        self.cat_indices = [i for i, col in enumerate(self.topological_order) if col in self.cat_cols]
        self.num_indices = [i for i, col in enumerate(self.topological_order) if col not in self.cat_cols]
        self.target_num_index = None
        if self.target_col is not None and self.target_col in self.topological_order:
            target_global_idx = self.topological_order.index(self.target_col)
            if target_global_idx in self.num_indices:
                self.target_num_index = self.num_indices.index(target_global_idx)

        # Step 10: 连续列归一化
        self.scaler = QuantileTransformer(output_distribution="normal", random_state=42)
        df_normalized = self.df_ordered.copy()
        if self.num_indices:
            num_cols_ordered = [self.topological_order[i] for i in self.num_indices]
            df_normalized[num_cols_ordered] = self.scaler.fit_transform(
                self.df_ordered[num_cols_ordered]
            )
        df_normalized = df_normalized.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        self.df_normalized = df_normalized

        # Step 11: 转张量
        self.tensor_data = torch.tensor(df_normalized.values, dtype=torch.float32)
        self.tensor_data = torch.nan_to_num(self.tensor_data, nan=0.0)
        for idx in self.cat_indices:
            self.tensor_data[:, idx] = self.tensor_data[:, idx].long().float()

        # 诊断
        print(f"📐 总特征数: {self.tensor_data.shape[1]}")
        print(f"📊 分类列数: {len(self.cat_cols)}, 连续列数: {len(self.num_indices)}")
        print(f"📏 数据范围: [{self.tensor_data.min().item():.4f}, {self.tensor_data.max().item():.4f}]")
        if self.target_num_index is not None:
            print(f"🎯 目标列在连续特征中的索引: {self.target_num_index}")

    def __len__(self):
        return len(self.tensor_data)

    def __getitem__(self, idx):
        return self.tensor_data[idx]


# ==========================================
# 3. EMA 工具函数
# ==========================================
def update_ema(target_params, source_params, rate=0.999):
    """与原版 TabDDPM 完全一致的 EMA 更新。"""
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)


def build_column_info_for_tensor_space(dataset: CausalTabularDataset, num_classes_array: np.ndarray) -> Dict[str, object]:
    """构建原始特征到扩散输入张量维度区间的映射。"""
    expanded_slices: Dict[str, Tuple[int, int]] = {}
    offset = 0

    # 连续特征：每个特征 1 维
    for idx in dataset.num_indices:
        feat = dataset.topological_order[idx]
        expanded_slices[feat] = (offset, offset + 1)
        offset += 1

    # 离散特征：每个特征展开为 one-hot 的 n_class 维
    cat_cols_in_topo_order = [dataset.topological_order[i] for i in dataset.cat_indices]
    for feat, n_class in zip(cat_cols_in_topo_order, num_classes_array.tolist()):
        width = int(n_class)
        expanded_slices[feat] = (offset, offset + width)
        offset += width

    return {
        "expanded_slices": expanded_slices,
        "total_dim": offset,
    }


def build_tensor_penalty_mask(column_info: Dict[str, object], causal_edges: List[Tuple[str, str]]) -> torch.Tensor:
    """
    构建张量级惩罚掩码 penalty_mask，形状 [D, D]。

    规则：
    - 默认 1.0（惩罚）。
    - 同一特征内部（例如 one-hot 的多个维度）置 0.0（允许协同）。
    - 存在因果关系的特征对置 0.0（允许协同）。
    """
    expanded_slices = column_info.get("expanded_slices", {})
    if not isinstance(expanded_slices, dict) or not expanded_slices:
        raise ValueError("column_info['expanded_slices'] is empty or invalid")

    total_dim = int(column_info.get("total_dim", 0))
    if total_dim <= 0:
        total_dim = max(int(v[1]) for v in expanded_slices.values())

    penalty_mask = torch.ones((total_dim, total_dim), dtype=torch.float32)

    # 同一特征 block 不惩罚
    for _, slc in expanded_slices.items():
        s, e = int(slc[0]), int(slc[1])
        penalty_mask[s:e, s:e] = 0.0

    # 因果边对应 block 不惩罚（对称置零，外积矩阵是对称统计）
    for src, dst in causal_edges:
        if src not in expanded_slices or dst not in expanded_slices:
            continue
        s1, e1 = expanded_slices[src]
        s2, e2 = expanded_slices[dst]
        penalty_mask[s1:e1, s2:e2] = 0.0
        penalty_mask[s2:e2, s1:e1] = 0.0

    return penalty_mask


def predict_eps_hat_for_causal_penalty(
    diffusion: GaussianMultinomialDiffusion,
    x_mixed: torch.Tensor,
    out_dict: Dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    按扩散训练同构流程生成 x_t 并预测 eps_hat。

    维度：
    - x_mixed: [B, D_raw]（连续 + 离散索引）
    - x_in:    [B, D_expand]（连续 + 离散 one-hot/log 表示）
    - eps_hat: [B, D_expand]
    """
    b = x_mixed.shape[0]
    device = x_mixed.device
    t, _ = diffusion.sample_time(b, device, 'uniform')

    x_num = x_mixed[:, :diffusion.num_numerical_features]
    x_cat = x_mixed[:, diffusion.num_numerical_features:]

    x_num_t = x_num
    log_x_cat_t = x_cat

    if x_num.shape[1] > 0:
        noise = torch.randn_like(x_num)
        x_num_t = diffusion.gaussian_q_sample(x_num, t, noise=noise)

    if x_cat.shape[1] > 0:
        log_x_cat = index_to_log_onehot(x_cat.long(), diffusion.num_classes)
        log_x_cat_t = diffusion.q_sample(log_x_start=log_x_cat, t=t)

    x_in = torch.cat([x_num_t, log_x_cat_t], dim=1)
    y_cond = out_dict.get('y') if isinstance(out_dict, dict) else None
    eps_hat = diffusion._denoise_fn(x_in, t, y=y_cond)
    return eps_hat


def causal_outer_product_penalty(eps_hat: torch.Tensor, penalty_mask: torch.Tensor) -> torch.Tensor:
    """
    基于噪声外积的因果惩罚。

    张量维度变化：
    - eps_hat: [B, D]
    - eps_hat.unsqueeze(2): [B, D, 1]
    - eps_hat.unsqueeze(1): [B, 1, D]
    - bmm 后: [B, D, D]
    - batch 均值: [D, D]
    """
    # [B, D, 1]
    eps_col = eps_hat.unsqueeze(2)
    # [B, 1, D]
    eps_row = eps_hat.unsqueeze(1)
    # [B, D, D]
    outer_batch = torch.bmm(eps_col, eps_row)
    # [D, D]
    mean_outer = outer_batch.mean(dim=0)

    # Hadamard 乘法，仅保留需要惩罚的无因果关系项
    penalized = mean_outer * penalty_mask
    # Frobenius 范数平方
    loss_causal = torch.sum(penalized ** 2)
    return loss_causal


def compute_hybrid_causal_weight(
    base_loss_now: float,
    prev_base_loss: float | None,
    step: int,
    total_steps: int,
    w_max: float,
    noisy_ratio: float = 0.2,
) -> Tuple[float, float, float]:
    """
    w_hybrid = w_max * 0.5 * ( exp(-|ΔL|) + 1 / (1 + sigma_approx) )
    """
    if prev_base_loss is None:
        delta_l = 0.0
    else:
        delta_l = abs(float(base_loss_now) - float(prev_base_loss))

    progress = step / max(total_steps - 1, 1)
    if progress < noisy_ratio:
        sigma_approx = (noisy_ratio - progress) / max(noisy_ratio, 1e-8)
    else:
        sigma_approx = 0.0

    w_hybrid = float(w_max) * 0.5 * (math.exp(-delta_l) + (1.0 / (1.0 + sigma_approx)))
    return w_hybrid, delta_l, sigma_approx


# ==========================================
# 4. 训练逻辑（对齐原版 TabDDPM steps 制训练）
# ==========================================
def train_yandex_causal():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"🖥️ 使用设备: {device} | {gpu_name}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    csv_path = "../baseline_experiment/data/nyc_2017_engineered_causal_v3.csv"
    if not os.path.exists(csv_path):
        print(f"❌ 错误：找不到数据集文件 {csv_path}")
        return

    print("📊 正在加载数据并构建因果图...")
    dataset = CausalTabularDataset(csv_path, NEW_CAUSAL_EDGES)

    # ★ 训练档位（对齐原版 TabDDPM：steps 制训练、大 batch、线性退火）
    #   原版 baseline: steps=500, batch=1024, lr=0.001, d_layers=[256,256]
    #   200K 数据 / batch=1024 ≈ 195 步/epoch
    train_mode = os.getenv("TRAIN_MODE", "balanced").strip().lower()
    if train_mode not in {"quick", "balanced", "full"}:
        train_mode = "balanced"

    profiles = {
        "quick":    {"steps": 1000,  "batch_size": 1024, "lr": 0.001, "weight_decay": 1e-5,
                     "d_layers": [256, 256]},
        "balanced": {"steps": 3000,  "batch_size": 1024, "lr": 0.002, "weight_decay": 1e-5,
                     "d_layers": [512, 512, 512]},
        "full":     {"steps": 5000,  "batch_size": 1024, "lr": 0.002, "weight_decay": 1e-5,
                     "d_layers": [768, 768, 768, 768]},
    }
    profile = profiles[train_mode]
    total_steps = profile["steps"]
    batch_size = profile["batch_size"]
    lr = profile["lr"]
    weight_decay = profile["weight_decay"]
    d_layers = profile["d_layers"]

    moment_matching_weight = 0.001
    causal_w_max = float(os.getenv("CAUSAL_W_MAX", "0.3"))

    num_workers = min(4, max((os.cpu_count() or 4) // 2, 2)) if device.type == "cuda" else 0
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(num_workers > 0) if num_workers > 0 else False,
    )

    # ★ 无限迭代器 — 跨 epoch 无缝读取全部数据（对齐原版 TabDDPM）
    def _infinite_iter():
        while True:
            for batch in dataloader:
                yield batch
    data_iter = iter(_infinite_iter())

    print(
        f"⚙️ 训练档位: {train_mode} | steps={total_steps}, batch={batch_size}, "
        f"lr={lr}, d_layers={d_layers}, lr_schedule=linear_annealing, EMA=0.999"
    )
    num_features = dataset.tensor_data.shape[1]

    # ★ 直接使用数据集中计算好的索引
    num_cat_features = len(dataset.cat_indices)
    
    # ★ 条件化生成(is_y_cond): 我们将 y 剥离出来作为网络条件
    # 真正的数值特征数量应减去 1 (因为包含了 target)
    num_num_features = len(dataset.num_indices)
    d_in_features = num_num_features - 1 if dataset.target_num_index is not None else num_num_features
    print(f"📊 离散特征数: {num_cat_features}")
    print(f"📊 连续特征总量(含目标): {num_num_features}, 扩散维度: {d_in_features}")


    meta_info = {
        "scaler": dataset.scaler,
        "topological_order": dataset.topological_order,
        "encoders": dataset.encoders,
        "cat_cols": dataset.cat_cols,
        "cat_indices": dataset.cat_indices,
        "num_indices": dataset.num_indices,
        "target_col": dataset.target_col,
        "target_num_index": dataset.target_num_index,
        "target_transformer": dataset.target_transformer,
        "time_postprocess": {"enabled": True},
    }
    meta_path = os.path.join(script_dir, "causal_scaler.pkl")
    joblib.dump(meta_info, meta_path)
    print(f"💾 已保存数据集元信息 ({meta_path})。")

    # ★ 离散列的类别数数组（必须按拓扑排序后的顺序）
    cat_cols_in_topo_order = [dataset.topological_order[i] for i in dataset.cat_indices]
    num_classes_array = np.array(
        [len(dataset.encoders[col].categories_[0]) for col in cat_cols_in_topo_order],
        dtype=int
    )
    
    print(f"🎯 拓扑序离散列: {cat_cols_in_topo_order}")
    print(f"🎯 num_classes_array: {num_classes_array}")
    print(f"🎯 num_classes_array.sum(): {num_classes_array.sum()}")
    
    # ★ 计算正确的输入维度：数值特征 + one-hot 编码的分类特征
    d_in_total = len(dataset.num_indices) + num_classes_array.sum()
    print(f"🎯 模型输入总维度 d_in: {d_in_total} = {len(dataset.num_indices)}(数值) + {num_classes_array.sum()}(分类 one-hot)")

    # 构建张量级惩罚掩码，后续直接用于外积因果惩罚。
    column_info = build_column_info_for_tensor_space(dataset, num_classes_array)
    mapped_edges_for_penalty = list(dataset.G.edges())
    penalty_mask = build_tensor_penalty_mask(column_info, mapped_edges_for_penalty).to(device)
    print(f"🧩 penalty_mask shape: {tuple(penalty_mask.shape)} | device={penalty_mask.device}")

    model = MLPDiffusion(
        d_in=d_in_total,
        num_classes=0,
        is_y_cond=False, # 遵循原版回归设定
        rtdl_params={"d_layers": d_layers, "dropout": 0.0},
    ).to(device)

    diffusion = GaussianMultinomialDiffusion(
        num_classes=num_classes_array,
        num_numerical_features=len(dataset.num_indices),
        denoise_fn=model,
        num_timesteps=1000,
        moment_matching_weight=moment_matching_weight,
        moment_matching_target_index=dataset.target_num_index,
    ).to(device)

    # ★ 不要 monkey patch forward
    # GaussianMultinomialDiffusion.forward = ...

    # ★ Optimizer: 对齐原版 AdamW (betas=0.9/0.999, weight_decay=1e-5)
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    # 不使用 scheduler — 由训练循环中的线性退火手动管理 LR

    # ★ EMA 模型（对齐原版 TabDDPM: deepcopy + 每步更新）
    ema_model = deepcopy(model)
    for p in ema_model.parameters():
        p.detach_()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_tag = "M4_CausalDDPM"
    run_name = f"{model_tag}_v4_s{total_steps}_{device.type}_{timestamp}"
    output_dir = os.path.join("runs", run_name)
    os.makedirs(output_dir, exist_ok=True)
    use_swanlab = SWANLAB_AVAILABLE and os.getenv("DISABLE_SWANLAB", "0").strip().lower() not in {"1", "true", "yes", "on"}
    if use_swanlab:
        swanlab.init(
            project="tab-ddpm-causal",
            experiment_name=run_name,
            config={
                "batch_size": batch_size,
                "lr": lr,
                "total_steps": total_steps,
                "moment_matching_weight": moment_matching_weight,
                "num_features": int(num_features),
                "d_layers": d_layers,
                "lr_schedule": "linear_annealing",
                "ema_rate": 0.999,
                "causal_w_max": causal_w_max,
            },
        )
    else:
        print("⚠️ swanlab 未安装或已通过环境变量禁用，仅输出控制台训练信息。")

    # ================================================================
    # ★ 训练循环 — 完全对齐原版 TabDDPM (scripts/train.py:Trainer)
    #   1) steps 制（非 epoch 制）
    #   2) 无限迭代器遍历全量数据
    #   3) 线性学习率退火：lr * (1 - step/total_steps)
    #   4) 每步 EMA 更新 (rate=0.999)
    #   5) 无梯度裁剪 / 无梯度累积
    #   6) 每 100 步记录、每 500 步打印
    # ================================================================
    step = 0
    curr_loss_multi = 0.0
    curr_loss_gauss = 0.0
    curr_count = 0
    log_every = 100
    print_every = 500
    best_loss = float("inf")
    prev_base_loss = None
    loss_history = []

    diffusion.train()
    start_time = time.time()
    print(f"\n🚀 开始训练 | 共 {total_steps} 步, batch={batch_size}, lr={lr}")

    while step < total_steps:
        x_batch = next(data_iter)
        x_batch = x_batch.to(device, non_blocking=True)

        if torch.isnan(x_batch).any():
            continue

        # 拆分数值 / 分类特征（拓扑序 → [num, cat_float] 拼接）
        if dataset.cat_indices:
            x_cat = x_batch[:, dataset.cat_indices].long()
            x_num = x_batch[:, dataset.num_indices]
            x_mixed = torch.cat([x_num, x_cat.float()], dim=1)
        else:
            x_mixed = x_batch

        # ★ 单步前向 + 反向（无梯度累积，对齐原版）
        optimizer.zero_grad()
        # 1) 基础重建损失
        loss_multi, loss_gauss = diffusion.mixed_loss(x_mixed, {})
        loss_base = loss_multi + loss_gauss

        # 2) 基于噪声外积的因果惩罚
        # eps_hat 形状: [B, D_expand]，D_expand = num_num + sum(num_classes)
        eps_hat = predict_eps_hat_for_causal_penalty(diffusion, x_mixed, out_dict={})
        loss_causal = causal_outer_product_penalty(eps_hat, penalty_mask)

        # 3) 自适应混合权重
        base_scalar = float(loss_base.detach().item())
        w_hybrid, delta_l, sigma_approx = compute_hybrid_causal_weight(
            base_loss_now=base_scalar,
            prev_base_loss=prev_base_loss,
            step=step,
            total_steps=total_steps,
            w_max=causal_w_max,
            noisy_ratio=0.2,
        )

        # 4) 总损失: L_total = L_base + w_hybrid * L_causal
        loss = loss_base + (loss_causal * w_hybrid)
        prev_base_loss = base_scalar

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        optimizer.step()

        # ★ 线性学习率退火（对齐原版: lr * (1 - step/steps)）
        frac_done = step / total_steps
        new_lr = lr * (1.0 - frac_done)
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        # ★ EMA 更新 — 每步（对齐原版: update_ema every step, rate=0.999）
        update_ema(ema_model.parameters(), model.parameters(), rate=0.999)

        # 统计
        bs = x_batch.shape[0]
        curr_count += bs
        curr_loss_multi += loss_multi.item() * bs
        curr_loss_gauss += loss_gauss.item() * bs

        if (step + 1) % log_every == 0:
            mloss = round(curr_loss_multi / max(curr_count, 1), 4)
            gloss = round(curr_loss_gauss / max(curr_count, 1), 4)
            total_loss = round(float(loss.item()), 4)

            loss_history.append({
                "step": step + 1,
                "mloss": mloss,
                "gloss": gloss,
                "base_loss": round(float(loss_base.item()), 4),
                "causal_loss": round(float(loss_causal.item()), 6),
                "hybrid_weight": round(float(w_hybrid), 6),
                "delta_l": round(float(delta_l), 6),
                "sigma_approx": round(float(sigma_approx), 6),
                "loss": total_loss,
                "lr": round(float(new_lr), 6),
            })

            if total_loss < best_loss:
                best_loss = total_loss
                # 保存 denoise_fn 权重 + EMA 权重（对齐原版）
                torch.save(model.state_dict(),
                           os.path.join(output_dir, "model_best.pt"))
                torch.save(ema_model.state_dict(),
                           os.path.join(output_dir, "model_ema_best.pt"))

            if (step + 1) % print_every == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (step + 1) * (total_steps - step - 1) / 60.0
                print(
                    f"Step [{step+1:5d}/{total_steps}] "
                    f"MLoss: {mloss:.4f}  GLoss: {gloss:.4f}  "
                    f"Base: {float(loss_base.item()):.4f}  Causal: {float(loss_causal.item()):.6f}  "
                    f"w_hybrid: {w_hybrid:.4f}  Sum: {total_loss:.4f}  LR: {new_lr:.2e}  "
                    f"ETA: {eta:.1f}m"
                )

            if use_swanlab:
                swanlab.log({
                    "train/mloss": mloss,
                    "train/gloss": gloss,
                    "train/base_loss": float(loss_base.item()),
                    "train/causal_loss": float(loss_causal.item()),
                    "train/hybrid_weight": float(w_hybrid),
                    "train/delta_l": float(delta_l),
                    "train/sigma_approx": float(sigma_approx),
                    "train/loss": total_loss,
                    "train/lr": new_lr,
                }, step=step + 1)

            curr_count = 0
            curr_loss_multi = 0.0
            curr_loss_gauss = 0.0

        step += 1

    elapsed = time.time() - start_time
    print(f"\n🎉 训练完成！用时 {elapsed/60:.1f} 分钟，最佳 loss: {best_loss:.4f}")

    # ★ 保存最终模型（对齐原版: model.pt + model_ema.pt）
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    torch.save(ema_model.state_dict(), os.path.join(output_dir, "model_ema.pt"))
    # 同时保存完整 diffusion state_dict（兼容评估脚本）
    torch.save(diffusion.state_dict(), os.path.join(output_dir, "causal_ddpm_best.pt"),
               pickle_protocol=4)

    # Loss CSV（与原版 loss.csv 格式一致: step, mloss, gloss, loss）
    loss_df = pd.DataFrame(loss_history)
    loss_csv_path = os.path.join(output_dir, "loss.csv")
    loss_df.to_csv(loss_csv_path, index=False)

    summary = {
        "run_name": run_name,
        "model": model_tag,
        "dataset": csv_path,
        "device": str(device),
        "gpu_name": gpu_name,
        "train_mode": train_mode,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "d_layers": d_layers,
        "lr_schedule": "linear_annealing",
        "ema_rate": 0.999,
        "moment_matching_weight": moment_matching_weight,
        "best_loss": float(best_loss),
        "final_step": step,
        "elapsed_minutes": round(elapsed / 60, 1),
        "num_workers": num_workers,
    }
    with open(os.path.join(output_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"🏷️ Run: {run_name}")
    print(f"📉 Loss日志: {loss_csv_path}")
    print(f"🏆 最终模型已保存！最佳 Loss: {best_loss:.4f}")

    if use_swanlab:
        swanlab.finish()


if __name__ == "__main__":
    train_yandex_causal()