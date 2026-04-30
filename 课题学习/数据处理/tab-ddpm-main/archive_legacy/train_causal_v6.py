"""
train_causal_v6.py  —  CausalDDPM v6 训练脚本
================================================
适配 v3 特征集 (3 连续 + ~44 分类)，从 data/nyc_crash_v3/ 的 npy + info.json 加载数据。

与 v5 (train_causal_yandex.py) 的关键差异:
  1. 数据加载: 从 npy 文件加载 (与原版 TabDDPM 一致)，而非 CSV
  2. 因果边定义: 更新以适配 v3 特征集(移除了 OSM/天气/地理列)
  3. 特征维度: 3 连续 + ~44 分类 (v2: 7+40)
  4. 新增 DAY_OF_WEEK 及 VEHICLE TYPE CODE 3/4/5 相关因果边
  5. 输出到 exp/nyc_crash_v3/causal_m4_v6/
"""

import os
import json
import time
from copy import deepcopy
from datetime import datetime

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, Dataset

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    swanlab = None
    SWANLAB_AVAILABLE = False

from tab_ddpm.modules import MLPDiffusion
from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion

# ==========================================
# 1. v3 因果边定义 (适配 v3 特征集)
# ==========================================
# v3 移除了 OSM/天气/地理文本列，因果边仅涉及训练中实际存在的特征
fci_v3_edges = [
    # ==== 1. 空间 ====
    ("LATITUDE", "LONGITUDE"),

    # ==== 2. 时间 → 驾驶行为 ====
    ("CRASH_TIME_PERIOD", "TOTAL_VEHICLES"),
    ("CRASH_TIME_PERIOD", "IS_AGGRESSIVE_DRIVING"),
    ("DAY_OF_WEEK", "IS_ALCOHOL_INVOLVED"),
    ("IS_WEEKEND", "IS_ALCOHOL_INVOLVED"),
    ("CRASH_SEASON", "IS_POOR_ROAD_CONDITION"),

    # ==== 3. 车辆因素 ====
    ("VEHICLE TYPE CODE 1", "IS_VEHICLE_DEFECT"),
    ("VEHICLE TYPE CODE 1", "IS_OVERSIZED_VEHICLE"),
    ("IS_OVERSIZED_VEHICLE", "IS_VISION_OBSCURED"),
    ("IS_VEHICLE_DEFECT", "IS_SPEEDING"),
    ("TOTAL_VEHICLES", "IS_MULTI_VEHICLE"),

    # ==== 4. 多车事故链 ====
    ("TOTAL_VEHICLES", "VEHICLE TYPE CODE 3"),
    ("TOTAL_VEHICLES", "VEHICLE TYPE CODE 4"),
    ("TOTAL_VEHICLES", "VEHICLE TYPE CODE 5"),

    # ==== 5. 驾驶员异常状态 ====
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
    ("IS_TRAFFIC_SIGNAL_VIOLATION", "IS_MULTI_VEHICLE"),

    # ==== 7. 行人与非机动车 ====
    ("IS_VISION_OBSCURED", "IS_PEDESTRIAN_CYCLIST_ERROR"),

    # ==== 8. 事故形态 → 伤亡 ====
    ("IS_SPEEDING", "NUMBER OF PERSONS INJURED"),
    ("IS_ALCOHOL_INVOLVED", "NUMBER OF PERSONS INJURED"),
    ("TOTAL_VEHICLES", "NUMBER OF PERSONS INJURED"),
    ("IS_PEDESTRIAN_CYCLIST_ERROR", "NUMBER OF PERSONS INJURED"),
    ("IS_MULTI_VEHICLE", "NUMBER OF PERSONS INJURED"),
    ("IS_FAILURE_TO_YIELD", "NUMBER OF PERSONS INJURED"),
    ("IS_VEHICLE_DEFECT", "NUMBER OF PERSONS INJURED"),
    ("REAL_SPEED_LIMIT", "IS_SPEEDING"),
    ("REAL_SPEED_LIMIT", "NUMBER OF PERSONS INJURED"),
]


# ==========================================
# 2. 数据集类 (从 npy 加载)
# ==========================================
class CausalTabularDatasetV3(Dataset):
    """从 TabDDPM 标准 npy 格式加载数据，并构建因果 DAG。

    v6.1 改进: 将 y (NUMBER_OF_PERSONS_INJURED) 作为分类变量
    纳入多项扩散，而非通过 QuantileTransformer 处理为连续值。
    y 分箱: {0, 1, 2, 3, 4, 5, 6, 7+} → 8 个类别。
    """

    Y_MAX_CLASS = 7   # y ∈ {0,1,...,7}，其中 7 代表 "≥7"

    def __init__(self, data_dir, causal_edges, split="train"):
        self.data_dir = data_dir

        # 加载 info.json
        with open(os.path.join(data_dir, "info.json"), "r") as f:
            self.info = json.load(f)

        # 加载 npy
        X_num = np.load(os.path.join(data_dir, f"X_num_{split}.npy"), allow_pickle=True).astype(np.float32)
        X_cat = np.load(os.path.join(data_dir, f"X_cat_{split}.npy"), allow_pickle=True).astype(np.int64)
        y = np.load(os.path.join(data_dir, f"y_{split}.npy"), allow_pickle=True).astype(np.float32)

        self.num_cols = self.info["num_columns"]
        self.cat_cols = self.info["cat_columns"]
        self.target_col = self.info["target_col"]
        cat_sizes = self.info.get("cat_sizes", None)

        print(f"📊 加载 {split} 数据: num={X_num.shape}, cat={X_cat.shape}, y={y.shape}")

        # Step 1: 目标列 → 分类变量 (取代 QuantileTransformer)
        # y 是离散计数 (NUMBER_OF_PERSONS_INJURED)，高度偏态 (81% 为 0)
        # 分箱: {0, 1, 2, 3, 4, 5, 6, 7+} → 8 个类别
        y_int = np.clip(np.round(y).astype(int), 0, None)
        y_clipped = np.clip(y_int, 0, self.Y_MAX_CLASS)  # ≥7 统一为 7
        self.y_num_classes = self.Y_MAX_CLASS + 1  # 8 classes

        # 解码映射: class 7 → 训练集中 y≥7 的均值
        y_high = y[y_int >= self.Y_MAX_CLASS]
        self.y_decode_value_for_max = float(np.mean(y_high)) if len(y_high) > 0 else float(self.Y_MAX_CLASS)
        # 各类别的精确解码值
        self.y_decode_map = {i: float(i) for i in range(self.Y_MAX_CLASS)}
        self.y_decode_map[self.Y_MAX_CLASS] = self.y_decode_value_for_max

        # y 分布诊断
        y_bincount = np.bincount(y_clipped, minlength=self.y_num_classes)
        print(f"🎯 y 分类化 ({self.y_num_classes} classes): {dict(zip(range(self.y_num_classes), y_bincount))}")
        print(f"   class {self.Y_MAX_CLASS} decode → {self.y_decode_value_for_max:.2f}")

        self.target_transformer = None  # 不再需要 QuantileTransformer

        # Step 2: 连续列归一化
        self.scaler = QuantileTransformer(output_distribution="normal", random_state=42)
        X_num_normalized = self.scaler.fit_transform(X_num).astype(np.float32)

        # Step 3: 构建完整特征列名 (num + cat + target)
        all_columns = list(self.num_cols) + list(self.cat_cols) + [self.target_col]

        # Step 4: 构建 DAG + 拓扑排序
        current_cols = set(all_columns)
        mapped_edges = [
            (s, t) for s, t in (causal_edges or [])
            if s in current_cols and t in current_cols
        ]
        self.G = nx.DiGraph()
        self.G.add_nodes_from(all_columns)
        self.G.add_edges_from(mapped_edges)
        # 处理环
        if not nx.is_directed_acyclic_graph(self.G):
            for cycle in list(nx.simple_cycles(self.G)):
                if len(cycle) > 1 and self.G.has_edge(cycle[-1], cycle[0]):
                    self.G.remove_edge(cycle[-1], cycle[0])

        self.topological_order = list(nx.topological_sort(self.G))

        # Step 5: 按拓扑序重排列索引
        col_to_idx = {col: i for i, col in enumerate(all_columns)}
        self.cat_indices = []  # 拓扑序中的位置
        self.num_indices = []  # 拓扑序中的位置
        self.target_cat_topo_pos = None  # y 在 cat_indices 中的位置

        # 构建拓扑序数据矩阵
        n_samples = X_num.shape[0]
        n_total = len(all_columns)
        data_matrix = np.zeros((n_samples, n_total), dtype=np.float32)

        for topo_idx, col in enumerate(self.topological_order):
            orig_idx = col_to_idx[col]
            if col == self.target_col:
                # y 作为分类变量
                data_matrix[:, topo_idx] = y_clipped.astype(np.float32)
                self.cat_indices.append(topo_idx)
                self.target_cat_topo_pos = len(self.cat_indices) - 1
            elif col in self.num_cols:
                num_pos = self.num_cols.index(col)
                data_matrix[:, topo_idx] = X_num_normalized[:, num_pos]
                self.num_indices.append(topo_idx)
            elif col in self.cat_cols:
                cat_pos = self.cat_cols.index(col)
                data_matrix[:, topo_idx] = X_cat[:, cat_pos].astype(np.float32)
                self.cat_indices.append(topo_idx)

        # 类别数数组（拓扑序）— 包含 y
        if cat_sizes is not None:
            cat_col_to_size = dict(zip(self.cat_cols, cat_sizes))
        else:
            # 从数据推断
            cat_col_to_size = {}
            for col in self.cat_cols:
                cat_pos = self.cat_cols.index(col)
                cat_col_to_size[col] = int(X_cat[:, cat_pos].max()) + 1
        # 加入 y 的类别数
        cat_col_to_size[self.target_col] = self.y_num_classes

        self.cat_cols_topo = [self.topological_order[i] for i in self.cat_indices]
        self.num_classes_array = np.array(
            [cat_col_to_size[col] for col in self.cat_cols_topo], dtype=int
        )

        # 转张量
        self.tensor_data = torch.tensor(data_matrix, dtype=torch.float32)
        self.tensor_data = torch.nan_to_num(self.tensor_data, nan=0.0)

        # 从 column_mapping 加载编码器信息
        mapping_path = os.path.join(data_dir, "column_mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, "r") as f:
                self.column_mapping = json.load(f)
        else:
            self.column_mapping = {}

        # 诊断
        print(f"🔍 拓扑排序: {self.topological_order}")
        print(f"📊 分类列 ({len(self.cat_indices)}): {self.cat_cols_topo}")
        print(f"📊 连续列 ({len(self.num_indices)}): {[self.topological_order[i] for i in self.num_indices]}")
        print(f"🎯 num_classes_array: {self.num_classes_array} (sum={self.num_classes_array.sum()})")
        print(f"🎯 目标列 '{self.target_col}' 在分类特征中的索引: {self.target_cat_topo_pos}")
        print(f"📐 总特征数: {self.tensor_data.shape[1]}")
        print(f"📏 数据范围: [{self.tensor_data.min().item():.4f}, {self.tensor_data.max().item():.4f}]")

    def __len__(self):
        return len(self.tensor_data)

    def __getitem__(self, idx):
        return self.tensor_data[idx]


# ==========================================
# 3. EMA 工具函数
# ==========================================
def update_ema(target_params, source_params, rate=0.999):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)


# ==========================================
# 4. 训练主函数
# ==========================================
def train_causal_v6(
    data_dir="data/nyc_crash_v3",
    output_base="exp/nyc_crash_v3/causal_m4_v6",
    train_mode=None,
    # 以下参数可选覆盖 profile 默认值 (消融实验用)
    steps=None, lr_override=None, batch_size_override=None,
    d_layers_override=None, num_timesteps_override=None,
    scheduler_override=None, lr_scheduler_override=None,
    causal_weight_override=None, dropout_override=None,
    ema_rate_override=None, weight_decay_override=None,
):
    """CausalDDPM v6 训练。

    Args:
        data_dir: v3 数据集目录
        output_base: 输出目录前缀
        train_mode: quick/balanced/full (默认从环境变量 TRAIN_MODE 读取)
        *_override: 覆盖 profile 默认值的可选参数 (消融实验用)
    """
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

    print("📊 正在加载 v3 数据集...")
    dataset = CausalTabularDatasetV3(data_dir, fci_v3_edges, split="train")

    # 训练档位
    if train_mode is None:
        train_mode = os.getenv("TRAIN_MODE", "balanced").strip().lower()
    if train_mode not in {"quick", "balanced", "full"}:
        train_mode = "balanced"

    profiles = {
        "quick":    {"steps": 1000,  "batch_size": 1024, "lr": 0.001, "weight_decay": 1e-5,
                     "d_layers": [256, 256]},
        "balanced": {"steps": 5000,  "batch_size": 1024, "lr": 0.002, "weight_decay": 1e-5,
                     "d_layers": [768, 768, 768, 768]},
        "full":     {"steps": 10000, "batch_size": 1024, "lr": 0.002, "weight_decay": 1e-5,
                     "d_layers": [768, 768, 768, 768]},
    }
    profile = profiles[train_mode]
    total_steps = steps if steps is not None else profile["steps"]
    batch_size = batch_size_override if batch_size_override is not None else profile["batch_size"]
    lr = lr_override if lr_override is not None else profile["lr"]
    weight_decay = weight_decay_override if weight_decay_override is not None else profile["weight_decay"]
    d_layers = d_layers_override if d_layers_override is not None else profile["d_layers"]
    num_timesteps = num_timesteps_override if num_timesteps_override is not None else 1000
    scheduler = scheduler_override if scheduler_override is not None else "cosine"
    dropout = dropout_override if dropout_override is not None else 0.0
    ema_rate = ema_rate_override if ema_rate_override is not None else 0.999
    moment_matching_weight = 0.001

    num_workers = min(4, max((os.cpu_count() or 4) // 2, 2)) if device.type == "cuda" else 0
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(num_workers > 0) if num_workers > 0 else False,
    )

    def _infinite_iter():
        while True:
            for batch in dataloader:
                yield batch
    data_iter = iter(_infinite_iter())

    print(
        f"⚙️ 训练档位: {train_mode} | steps={total_steps}, batch={batch_size}, "
        f"lr={lr}, d_layers={d_layers}"
    )

    # 模型维度
    num_num_features = len(dataset.num_indices)
    d_in_total = num_num_features + dataset.num_classes_array.sum()
    print(f"🎯 模型输入总维度 d_in: {d_in_total} = {num_num_features}(数值) + {dataset.num_classes_array.sum()}(分类 one-hot)")

    model = MLPDiffusion(
        d_in=d_in_total,
        num_classes=0,
        is_y_cond=False,
        rtdl_params={"d_layers": d_layers, "dropout": dropout},
    ).to(device)

    diffusion = GaussianMultinomialDiffusion(
        num_classes=dataset.num_classes_array,
        num_numerical_features=num_num_features,
        denoise_fn=model,
        num_timesteps=num_timesteps,
        gaussian_loss_type="mse",
        scheduler=scheduler,
        moment_matching_weight=moment_matching_weight,
        moment_matching_target_index=None,  # y 已移到分类变量，不再矩匹配
    ).to(device)

    optimizer = torch.optim.AdamW(
        diffusion.parameters(), lr=lr,
        weight_decay=weight_decay, betas=(0.9, 0.999),
    )

    ema_model = deepcopy(model)
    for p in ema_model.parameters():
        p.detach_()

    # 保存元信息
    meta_info = {
        "scaler": dataset.scaler,
        "topological_order": dataset.topological_order,
        "column_mapping": dataset.column_mapping,
        "cat_cols": dataset.cat_cols,
        "num_cols": dataset.num_cols,
        "cat_cols_topo": dataset.cat_cols_topo,
        "cat_indices": dataset.cat_indices,
        "num_indices": dataset.num_indices,
        "target_col": dataset.target_col,
        "target_cat_topo_pos": dataset.target_cat_topo_pos,
        "y_num_classes": dataset.y_num_classes,
        "y_decode_map": dataset.y_decode_map,
        "y_max_class": dataset.Y_MAX_CLASS,
        "target_transformer": None,  # 不再使用 QuantileTransformer
        "num_classes_array": dataset.num_classes_array,
        "info": dataset.info,
    }
    os.makedirs(output_base, exist_ok=True)
    meta_path = os.path.join(output_base, "causal_meta_v6.pkl")
    joblib.dump(meta_info, meta_path)
    print(f"💾 已保存数据集元信息 ({meta_path})")

    # 输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"M4_CausalDDPM_v6_s{total_steps}_{device.type}_{timestamp}"
    output_dir = os.path.join("runs", run_name)
    os.makedirs(output_dir, exist_ok=True)

    use_swanlab = SWANLAB_AVAILABLE and os.getenv("DISABLE_SWANLAB", "0").strip().lower() not in {"1", "true"}
    if use_swanlab:
        assert swanlab is not None
        swanlab.init(
            project="tab-ddpm-causal-v6",
            experiment_name=run_name,
            config={
                "batch_size": batch_size, "lr": lr, "total_steps": total_steps,
                "d_layers": d_layers, "feature_set": "v3",
            },
        )

    # === 训练循环 ===
    step = 0
    curr_loss_multi = 0.0
    curr_loss_gauss = 0.0
    curr_count = 0
    log_every = 100
    print_every = 500
    best_loss = float("inf")
    loss_history = []

    diffusion.train()
    start_time = time.time()
    print(f"\n🚀 开始训练 | 共 {total_steps} 步, batch={batch_size}, lr={lr}")

    while step < total_steps:
        x_batch = next(data_iter).to(device, non_blocking=True)

        if torch.isnan(x_batch).any():
            continue

        if dataset.cat_indices:
            x_cat = x_batch[:, dataset.cat_indices].long()
            x_num = x_batch[:, dataset.num_indices]
            x_mixed = torch.cat([x_num, x_cat.float()], dim=1)
        else:
            x_mixed = x_batch

        optimizer.zero_grad()
        loss_multi, loss_gauss = diffusion.mixed_loss(x_mixed, {})
        loss = loss_multi + loss_gauss

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        optimizer.step()

        # Cosine 学习率退火 (with warmup)
        warmup_steps = min(500, total_steps // 10)
        if step < warmup_steps:
            new_lr = lr * (step + 1) / warmup_steps
        else:
            import math
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            new_lr = lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        update_ema(ema_model.parameters(), model.parameters(), rate=ema_rate)

        bs = x_batch.shape[0]
        curr_count += bs
        curr_loss_multi += loss_multi.item() * bs
        curr_loss_gauss += loss_gauss.item() * bs

        if (step + 1) % log_every == 0:
            mloss = round(curr_loss_multi / max(curr_count, 1), 4)
            gloss = round(curr_loss_gauss / max(curr_count, 1), 4)
            total_loss = round(mloss + gloss, 4)
            loss_history.append({
                "step": step + 1, "mloss": mloss, "gloss": gloss,
                "loss": total_loss, "lr": round(float(new_lr), 6),
            })

            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(model.state_dict(), os.path.join(output_dir, "model_best.pt"))
                torch.save(ema_model.state_dict(), os.path.join(output_dir, "model_ema_best.pt"))

            if (step + 1) % print_every == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (step + 1) * (total_steps - step - 1) / 60.0
                print(
                    f"Step [{step+1:5d}/{total_steps}] "
                    f"MLoss: {mloss:.4f}  GLoss: {gloss:.4f}  "
                    f"Sum: {total_loss:.4f}  LR: {new_lr:.2e}  "
                    f"ETA: {eta:.1f}m"
                )

            if use_swanlab:
                assert swanlab is not None
                swanlab.log({"train/loss": total_loss, "train/lr": new_lr}, step=step + 1)

            curr_count = 0
            curr_loss_multi = 0.0
            curr_loss_gauss = 0.0

        step += 1

    elapsed = time.time() - start_time
    print(f"\n🎉 训练完成！用时 {elapsed/60:.1f} 分钟，最佳 loss: {best_loss:.4f}")

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    torch.save(ema_model.state_dict(), os.path.join(output_dir, "model_ema.pt"))
    torch.save(diffusion.state_dict(), os.path.join(output_dir, "causal_ddpm_best.pt"),
               pickle_protocol=4)

    # Loss CSV
    loss_df = pd.DataFrame(loss_history)
    loss_df.to_csv(os.path.join(output_dir, "loss.csv"), index=False)

    # 同步保存到 exp/ 目录
    import shutil
    for fname in ["model.pt", "model_ema.pt", "causal_ddpm_best.pt", "loss.csv"]:
        src = os.path.join(output_dir, fname)
        dst = os.path.join(output_base, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    summary = {
        "run_name": run_name, "dataset": data_dir,
        "device": str(device), "gpu_name": gpu_name,
        "train_mode": train_mode, "total_steps": total_steps,
        "batch_size": batch_size, "lr": lr,
        "d_layers": d_layers, "best_loss": float(best_loss),
        "elapsed_minutes": round(elapsed / 60, 1),
        "feature_set": "v3",
        "num_timesteps": num_timesteps, "scheduler": scheduler,
        "ema_rate": ema_rate, "dropout": dropout,
    }
    for out_path in [output_dir, output_base]:
        with open(os.path.join(out_path, "train_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"🏷️ Run: {run_name}")
    print(f"📉 最佳 Loss: {best_loss:.4f}")
    print(f"📁 模型保存: {output_dir} + {output_base}")

    # =======================================================
    # 采样阶段: 生成合成数据 → 保存为 npy 供评估使用
    # =======================================================
    print(f"\n📦 开始采样 {dataset.info['train_size']} 条合成数据...")
    diffusion.eval()

    # 使用 EMA 模型采样
    orig_state = deepcopy(model.state_dict())
    model.load_state_dict(ema_model.state_dict())

    num_samples = dataset.info["train_size"]
    sample_batch_size = min(2048, num_samples)

    # 构建 y 分布 (用于采样, 即使 is_y_cond=False 也需要传入)
    y_all = np.load(os.path.join(data_dir, "y_train.npy"), allow_pickle=True).astype(np.float32)
    y_int = np.clip(np.round(y_all).astype(int), 0, None)
    max_y = min(int(y_int.max()) + 1, 100)
    y_counts = np.bincount(y_int, minlength=max_y).astype(np.float32)
    y_counts = y_counts / y_counts.sum()
    y_dist_tensor = torch.tensor(y_counts, dtype=torch.float32).to(device)

    with torch.no_grad():
        try:
            x_gen, y_gen = diffusion.sample_all(num_samples, sample_batch_size, y_dist_tensor)
            generated = x_gen.cpu()
        except Exception as e:
            print(f"⚠️ sample_all 失败 ({e}), 使用逐批采样...")
            all_samples = []
            remaining = num_samples
            while remaining > 0:
                bs = min(sample_batch_size, remaining)
                sample, out_dict = diffusion.sample(bs, y_dist_tensor)
                all_samples.append(sample.cpu())
                remaining -= bs
            generated = torch.cat(all_samples, dim=0)[:num_samples]

    model.load_state_dict(orig_state)

    # 拆分为 X_num 和 X_cat
    # sample() 返回 [z_norm(num_num), z_cat(num_cat_cols)] — z_cat 已是类别索引
    gen_np = generated.numpy()
    num_num = len(dataset.num_indices)
    num_cat = len(dataset.cat_indices)
    print(f"📐 generated shape: {gen_np.shape}, num_num={num_num}, num_cat={num_cat}")
    X_num_syn = gen_np[:, :num_num].astype(np.float32)
    X_cat_syn = gen_np[:, num_num:num_num + num_cat].astype(np.int64)

    # === y 从分类变量解码 ===
    y_cat_idx = dataset.target_cat_topo_pos
    y_syn_class = np.clip(X_cat_syn[:, y_cat_idx], 0, dataset.Y_MAX_CLASS).astype(int)
    # 解码: class → numeric value
    y_syn = np.array([dataset.y_decode_map[c] for c in y_syn_class], dtype=np.float32)
    print(f"🎯 y 合成分布: mean={y_syn.mean():.4f}, zero_ratio={np.mean(y_syn == 0)*100:.1f}%")
    y_bincount_syn = np.bincount(y_syn_class, minlength=dataset.y_num_classes)
    print(f"   y_syn 各类: {dict(zip(range(dataset.y_num_classes), y_bincount_syn))}")

    # === 连续特征反归一化 (从拓扑序映射回原始列顺序) ===
    num_topo_cols = [dataset.topological_order[i] for i in dataset.num_indices]
    X_num_final = np.zeros((num_samples, len(dataset.num_cols)), dtype=np.float32)
    for i, col in enumerate(num_topo_cols):
        if col in dataset.num_cols:
            orig_idx = dataset.num_cols.index(col)
            X_num_final[:, orig_idx] = X_num_syn[:, i]
    X_num_final = dataset.scaler.inverse_transform(X_num_final).astype(np.float32)

    # === 分类特征映射回原始列顺序 (跳过 y) ===
    X_cat_final = np.zeros((num_samples, len(dataset.cat_cols)), dtype=np.int64)
    for j, col in enumerate(dataset.cat_cols_topo):
        if col == dataset.target_col:
            continue  # y 不放入 X_cat_final
        cat_values = np.clip(X_cat_syn[:, j], 0, dataset.num_classes_array[j] - 1)
        if col in dataset.cat_cols:
            orig_cat_idx = dataset.cat_cols.index(col)
            X_cat_final[:, orig_cat_idx] = cat_values

    # 保存到 output_base
    np.save(os.path.join(output_base, "X_num_train.npy"), X_num_final)
    np.save(os.path.join(output_base, "X_cat_train.npy"), X_cat_final)
    np.save(os.path.join(output_base, "y_train.npy"), y_syn)

    # 复制 info.json 和 column_mapping.json
    import shutil as _shutil
    for fname in ["info.json", "column_mapping.json"]:
        src_f = os.path.join(data_dir, fname)
        dst_f = os.path.join(output_base, fname)
        if os.path.exists(src_f):
            _shutil.copy2(src_f, dst_f)

    print(f"✅ 合成数据已保存: X_num={X_num_final.shape}, X_cat={X_cat_final.shape}, y={y_syn.shape}")
    print(f"   输出目录: {output_base}")

    if use_swanlab:
        assert swanlab is not None
        swanlab.finish()

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CausalDDPM v6 训练")
    parser.add_argument("--data", type=str, default="data/nyc_crash_v3")
    parser.add_argument("--output", type=str, default="exp/nyc_crash_v3/causal_m4_v6")
    parser.add_argument("--mode", type=str, choices=["quick", "balanced", "full"], default=None)
    args = parser.parse_args()
    train_causal_v6(data_dir=args.data, output_base=args.output, train_mode=args.mode)
