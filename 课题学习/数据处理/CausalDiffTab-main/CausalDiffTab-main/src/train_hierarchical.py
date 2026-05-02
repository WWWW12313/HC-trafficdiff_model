"""
Hierarchical CausalDiffTab - 分层训练入口
==========================================
支持三个 Stage 的训练:
  Stage 1: 空间扩散 (LATITUDE, LONGITUDE) - 纯连续模型
    Stage 2: 上下文扩散 (天气 + OSM/路网) - 条件于 Stage 1 的可训练环境层
  Stage 3: 全特征条件生成 - 以空间 + 离线上下文为条件

三档训练计划:
  quick    - 采样 500,  训练 10 epochs   (快速验证)
  balanced - 采样 2000, 训练 200 epochs  (迭代调参)
  full     - 全量数据,  训练 4000 epochs (正式训练)

使用方法:
  python src/train_hierarchical.py --stage 1 --tier quick --device cuda:0
    python src/train_hierarchical.py --stage 2 --tier quick --device cuda:0
  python src/train_hierarchical.py --stage 3 --tier balanced --device cuda:0
  python src/train_hierarchical.py --stage 3 --tier full --device cuda:0
"""

import os
import sys
import json
import glob
import pickle
import random
import argparse
from pathlib import Path
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

# 确保 CausalDiffTab 根目录在 path 中
CDT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CDT_ROOT))

import src as cdt_src
from tabdiff.modules.main_modules import UniModMLP, Model
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from tabdiff.trainer import Trainer
from utils_train import TabDiffDataset, preprocess

import time
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def _safe_remove_file(path: str, retries: int = 5, delay_sec: float = 0.2) -> bool:
    """Best-effort file remove for Windows where checkpoints may be briefly locked."""
    for _ in range(max(1, retries)):
        try:
            if os.path.exists(path):
                os.remove(path)
            return True
        except PermissionError:
            time.sleep(delay_sec)
        except FileNotFoundError:
            return True
    return False


# ============================================================
# 1. 配置
# ============================================================

TIER_SETTINGS = {
    "quick": {
        "steps": 10,
        "batch_size": 256,
        "check_val_every": 5,
        "sample_size": 500,
        "num_timesteps": 10,
        "sample_batch_size": 500,
    },
    "balanced": {
        "steps": 200,
        "batch_size": 512,
        "check_val_every": 50,
        "sample_size": 2000,
        "num_timesteps": 50,
        "sample_batch_size": 2000,
    },
    "full": {
        "steps": 4000,
        "batch_size": 4096,
        "check_val_every": 500,
        "sample_size": None,
        "num_timesteps": 50,
        "sample_batch_size": 50000,
    },
}


def build_config(stage: int, device: str, tier: str = "quick") -> dict:
    """
    构建训练配置 (基于 CausalDiffTab 默认 + 分层调整)
    tier: 'quick' / 'balanced' / 'full'
    """
    curr_dir = str(CDT_ROOT / "tabdiff")
    config_path = os.path.join(curr_dir, "configs", "tabdiff_configs.toml")
    raw_config = cdt_src.load_config(config_path)

    t = TIER_SETTINGS[tier]
    raw_config["train"]["main"]["steps"] = t["steps"]
    raw_config["train"]["main"]["batch_size"] = t["batch_size"]
    raw_config["train"]["main"]["check_val_every"] = t["check_val_every"]
    raw_config["diffusion_params"]["num_timesteps"] = t["num_timesteps"]
    raw_config["sample"]["batch_size"] = t["sample_batch_size"]

    if stage == 1:
        raw_config["train"]["main"]["lr"] = 0.002
        raw_config["train"]["main"]["closs_weight_schedule"] = "fixed"
        raw_config["unimodmlp_params"]["num_layers"] = 2
        raw_config["unimodmlp_params"]["d_token"] = 4
        raw_config["unimodmlp_params"]["dim_t"] = 256
        raw_config["unimodmlp_params"]["factor"] = 16
        raw_config["unimodmlp_params"]["dropout"] = 0.0
    elif stage == 2:
        raw_config["train"]["main"]["lr"] = 0.0015
        raw_config["train"]["main"]["closs_weight_schedule"] = "anneal"
        raw_config["unimodmlp_params"]["num_layers"] = 3
        raw_config["unimodmlp_params"]["d_token"] = 8
        raw_config["unimodmlp_params"]["dim_t"] = 256
        raw_config["unimodmlp_params"]["factor"] = 24
        raw_config["unimodmlp_params"]["dropout"] = 0.1
    else:
        raw_config["train"]["main"]["lr"] = 0.001
        raw_config["train"]["main"]["closs_weight_schedule"] = "anneal"
        raw_config["unimodmlp_params"]["dropout"] = 0.15

    return raw_config


# ============================================================
# 2. 加载预计算的因果掩码
# ============================================================

def load_precomputed_causal_masks(data_dir: str, device: str, mask_subdir: str = "causal_masks"):
    """加载 prepare_dataset.py 预计算的因果掩码"""
    mask_dir = os.path.join(data_dir, mask_subdir)
    num_path = os.path.join(mask_dir, "num_causal_mask.npy")
    cat_path = os.path.join(mask_dir, "cat_causal_mask.npy")

    if os.path.exists(num_path):
        num_mask = np.load(num_path)
        print(f"[mask] Loaded num_causal_mask: {num_mask.shape}, edges={int(num_mask.sum())}")
    else:
        num_mask = None
        print("[mask] No num_causal_mask found, causal regularization disabled for numerical")

    if os.path.exists(cat_path):
        cat_mask = np.load(cat_path)
        if cat_mask.size == 0:
            cat_mask = None
            print("[mask] cat_causal_mask is empty (no categorical features)")
        else:
            print(f"[mask] Loaded cat_causal_mask: {cat_mask.shape}, edges={int(cat_mask.sum())}")
    else:
        cat_mask = None
        print("[mask] No cat_causal_mask found, causal regularization disabled for categorical")

    return num_mask, cat_mask


# ============================================================
# 3. 修补 Trainer 以使用预计算掩码
# ============================================================

class HierarchicalTrainer(Trainer):
    """
    继承 CausalDiffTab 的 Trainer，
    覆盖 extract_and_set_causal_mask 以加载预计算的 NOTEARS 掩码。
    支持 CosineAnnealingLR + Early Stopping。
    """

    def __init__(self, *args, precomputed_masks=None, early_stop_patience=200, **kwargs):
        self._precomputed_masks = precomputed_masks
        self.early_stop_patience = early_stop_patience
        super().__init__(*args, **kwargs)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.steps, eta_min=1e-6)
        self.lr_scheduler = "cosine"

    def extract_and_set_causal_mask(self):
        """覆盖: 直接加载预计算掩码，不重新跑 NOTEARS"""
        if self._precomputed_masks is None:
            print("[mask] No precomputed masks, skipping causal regularization")
            return

        num_mask, cat_mask = self._precomputed_masks

        if num_mask is not None:
            # 验证维度匹配
            expected_num_dim = self.diffusion.num_numerical_features
            if num_mask.shape[0] != expected_num_dim:
                print(f"[mask] WARN: num_mask {num_mask.shape} != expected ({expected_num_dim},{expected_num_dim})")
                print(f"[mask] Resizing num_causal_mask to ({expected_num_dim},{expected_num_dim})")
                new_mask = np.zeros((expected_num_dim, expected_num_dim), dtype=np.float32)
                min_d = min(num_mask.shape[0], expected_num_dim)
                new_mask[:min_d, :min_d] = num_mask[:min_d, :min_d]
                num_mask = new_mask
        else:
            expected_num_dim = self.diffusion.num_numerical_features
            num_mask = np.zeros((expected_num_dim, expected_num_dim), dtype=np.float32)

        if cat_mask is None:
            # 如果没有分类特征的掩码，创建全零掩码
            if len(self.diffusion.num_classes) > 0:
                total_oh = sum(self.diffusion.num_classes + 1)
                cat_mask = np.zeros((total_oh, total_oh), dtype=np.float32)
            else:
                cat_mask = np.zeros((0, 0), dtype=np.float32)
        else:
            # 验证维度
            if len(self.diffusion.num_classes) > 0:
                expected_cat_dim = sum(self.diffusion.num_classes + 1)
                if cat_mask.shape[0] != expected_cat_dim:
                    print(f"[mask] WARN: cat_mask {cat_mask.shape} != expected ({expected_cat_dim},{expected_cat_dim})")
                    new_mask = np.zeros((expected_cat_dim, expected_cat_dim), dtype=np.float32)
                    min_d = min(cat_mask.shape[0], expected_cat_dim)
                    new_mask[:min_d, :min_d] = cat_mask[:min_d, :min_d]
                    cat_mask = new_mask

        self.diffusion.set_causal_masks(num_mask, cat_mask)
        n_num_edges = int(num_mask.sum()) if num_mask is not None else 0
        n_cat_edges = int(cat_mask.sum()) if cat_mask is not None else 0
        print(f"[mask] Set causal masks: num_edges={n_num_edges}, cat_edges={n_cat_edges}")

    def evaluate_generation(self, save_metric_details=False, plot_density=False, ema=False):
        """健壮版评估: 跳过不可用的 metrics 组件"""
        self.diffusion.eval()

        num_samples = (
            self.num_samples_to_generate
            if self.num_samples_to_generate
            else getattr(self.metrics, "real_data_size", 1000)
        )
        num_samples = min(num_samples, 5000)

        try:
            syn_df = self.sample_synthetic(num_samples, ema=ema)
        except Exception as e:
            print(f"[eval] Sampling failed: {e}")
            return {}, None, None

        save_path = os.path.join(
            self.result_save_path, str(self.curr_epoch), "ema" if ema else ""
        )
        os.makedirs(save_path, exist_ok=True)
        path = os.path.join(save_path, "samples.csv")
        syn_df.to_csv(path, index=False)
        print(f"[eval] Samples saved at {path} ({len(syn_df)} rows)")

        out_metrics = {}
        extras = {}
        try:
            syn_df_loaded = pd.read_csv(path)
            result = self.metrics.evaluate(syn_df_loaded)
            if isinstance(result, tuple) and len(result) == 2:
                out_metrics, extras = result
            elif isinstance(result, dict):
                out_metrics = result
        except Exception as e:
            print(f"[eval] Metrics evaluation skipped: {e}")

        return out_metrics, extras, syn_df

    def run_loop(self):
        """覆盖 run_loop: CosineAnnealingLR + Early Stopping + 因果Warmup"""
        import glob as _glob
        from tabdiff.trainer import print_with_bar
        from utils_train import update_ema

        es_patience_counter = 0
        closs_weight, dloss_weight = self.c_lambda, self.d_lambda
        best_loss = np.inf
        best_ema_loss = np.inf
        start_time = time.time()
        min_save_epoch = max(1, self.steps // 10)

        print_with_bar(
            f"Starting Training Loop, total epochs = {self.steps}, "
            f"save after epoch {min_save_epoch}, "
            f"early_stop_patience = {self.early_stop_patience}, "
            f"lr_scheduler = {self.lr_scheduler}"
        )
        self.logger.define_metric("epoch")
        self.logger.define_metric("*", step_metric="epoch")

        start_epoch = self.curr_epoch
        for epoch in range(start_epoch, self.steps):
            self.curr_epoch = epoch + 1

            pbar = tqdm(self.train_iter, total=len(self.train_iter))
            pbar.set_description(f"Epoch {epoch+1}/{self.steps}")

            if self.closs_weight_schedule == "anneal":
                frac_done = epoch / self.steps
                closs_weight = self.c_lambda * (1 - frac_done)

            curr_dloss, curr_closs, curr_count = 0.0, 0.0, 0
            curr_lr = self.optimizer.param_groups[0]["lr"]
            for batch in pbar:
                x = batch.float().to(self.device)
                batch_dloss, batch_closs = self._run_step(x, closs_weight, dloss_weight)
                curr_dloss += batch_dloss.item() * len(x)
                curr_closs += batch_closs.item() * len(x)
                curr_count += len(x)
                pbar.set_postfix({
                    "lr": f"{curr_lr:.2e}",
                    "DLoss": np.around(curr_dloss / curr_count, 4),
                    "CLoss": np.around(curr_closs / curr_count, 4),
                })

            mloss = np.around(curr_dloss / curr_count, 4)
            gloss = np.around(curr_closs / curr_count, 4)
            total_loss = mloss + gloss

            if np.isnan(gloss):
                print("[ERROR] NaN in loss, stopping")
                break

            log_dict = {
                "epoch": epoch + 1,
                "lr": curr_lr,
                "loss/d_loss": mloss,
                "loss/c_loss": gloss,
                "loss/total_loss": total_loss,
            }

            # CosineAnnealingLR: step per epoch (no loss argument needed)
            self.scheduler.step()

            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters(), rate=self.ema_decay)
            update_ema(self.ema_num_schedule.parameters(), self.diffusion.num_schedule.parameters(), rate=self.ema_decay)
            update_ema(self.ema_cat_schedule.parameters(), self.diffusion.cat_schedule.parameters(), rate=self.ema_decay)

            # Best model checkpoint + Early Stopping patience tracking
            if total_loss < best_loss and self.curr_epoch >= min_save_epoch:
                best_loss = total_loss
                es_patience_counter = 0
                to_rm = _glob.glob(os.path.join(self.model_save_path, "best_model_*"))
                for f in to_rm:
                    if not _safe_remove_file(f):
                        print_with_bar(f"[warn] skip removing locked checkpoint: {f}")
                state_dicts = {
                    "denoise_fn": self.diffusion._denoise_fn.state_dict(),
                    "num_schedule": self.diffusion.num_schedule.state_dict(),
                    "cat_schedule": self.diffusion.cat_schedule.state_dict(),
                }
                torch.save(state_dicts, os.path.join(
                    self.model_save_path, f"best_model_{np.round(total_loss,4)}_{epoch+1}.pt"
                ))
            elif self.curr_epoch >= min_save_epoch:
                es_patience_counter += 1

            curr_model, curr_num_schedule, curr_cat_schedule = self.to_ema_model()
            ema_mloss, ema_gloss = self.compute_loss()
            self.to_model(curr_model, curr_num_schedule, curr_cat_schedule)
            ema_total_loss = ema_mloss + ema_gloss

            if ema_total_loss < best_ema_loss and self.curr_epoch >= min_save_epoch:
                best_ema_loss = ema_total_loss
                to_rm = _glob.glob(os.path.join(self.model_save_path, "best_ema_model_*"))
                for f in to_rm:
                    if not _safe_remove_file(f):
                        print_with_bar(f"[warn] skip removing locked checkpoint: {f}")
                state_dicts = {
                    "denoise_fn": self.ema_model.state_dict(),
                    "num_schedule": self.ema_num_schedule.state_dict(),
                    "cat_schedule": self.ema_cat_schedule.state_dict(),
                }
                torch.save(state_dicts, os.path.join(
                    self.model_save_path, f"best_ema_model_{np.round(ema_total_loss,4)}_{epoch+1}.pt"
                ))

            if (epoch + 1) % self.check_val_every == 0:
                state_dicts = {
                    "denoise_fn": self.diffusion._denoise_fn.state_dict(),
                    "num_schedule": self.diffusion.num_schedule.state_dict(),
                    "cat_schedule": self.diffusion.cat_schedule.state_dict(),
                }
                torch.save(state_dicts, os.path.join(self.model_save_path, f"model_{epoch+1}.pt"))
                print_with_bar(f"Evaluation at epoch #{epoch+1}, total_loss={total_loss}")
                out_metrics, _, _ = self.evaluate_generation(save_metric_details=True, plot_density=False)
                log_dict.update(out_metrics if out_metrics else {})

            try:
                self.logger.log(log_dict)
            except Exception:
                pass

            # Early Stopping check
            if es_patience_counter >= self.early_stop_patience:
                print_with_bar(
                    f"Early Stopping triggered at epoch {epoch+1}: "
                    f"no improvement for {self.early_stop_patience} epochs. "
                    f"best_loss={best_loss:.4f}"
                )
                break

        elapsed = time.time() - start_time
        print_with_bar(f"Training finished in {elapsed:.0f}s, best_loss={best_loss:.4f}")


# ============================================================
# 4. 主训练流程
# ============================================================

def train_stage(
    stage: int,
    device: str = "cuda:0",
    tier: str = "quick",
    no_wandb: bool = True,
    deterministic: bool = False,
    experiment_id: Optional[str] = None,
    lambda_causal: float = 1.0,
    use_causal_masks: bool = True,
    mask_subdir: str = "causal_masks",
    dataname: Optional[str] = None,
):
    """训练指定 Stage 的扩散模型

    experiment_id: 非空时 checkpoint 目录加后缀，避免多组实验互相覆盖，例如 stage3_full_balanced_ours_full_model
    lambda_causal: 因果正则强度，对应 UnifiedCtimeDiffusion.causal_weight_max（0 即关闭因果项）
    use_causal_masks: False 时不加载 NOTEARS 掩码（与 TabDDPM 式无因果先验一致）
    """

    np.set_printoptions(suppress=True)
    torch.set_printoptions(sci_mode=False)

    tier_cfg = TIER_SETTINGS[tier]
    sample_size = tier_cfg["sample_size"]

    _exp_suffix = f"_{experiment_id}" if experiment_id else ""
    if stage == 1:
        if dataname and dataname.startswith("nyc_crash_"):
            dataname = dataname.replace("nyc_crash", "nyc_stage1", 1)
        else:
            dataname = "nyc_stage1"
        exp_name = f"stage1_spatial_{tier}{_exp_suffix}"
    elif stage == 2:
        if dataname and dataname.startswith("nyc_crash_"):
            dataname = dataname.replace("nyc_crash", "nyc_stage2", 1)
        elif not dataname:
            dataname = "nyc_stage2"
        exp_name = f"stage2_context_{tier}{_exp_suffix}"
    else:
        dataname = dataname or "nyc_crash"
        exp_name = f"stage3_full_{tier}{_exp_suffix}"

    data_dir = str(CDT_ROOT / "data" / dataname)
    info_path = os.path.join(data_dir, "info.json")

    if not os.path.exists(info_path):
        print(f"[ERROR] Dataset not found at {data_dir}")
        print("  Run `python src/prepare_dataset.py` first!")
        return

    with open(info_path, "r") as f:
        info = json.load(f)

    print("=" * 60)
    print(f"Hierarchical CausalDiffTab - Stage {stage} [{tier}]")
    print(f"  Dataset: {dataname}")
    print(f"  Num features: {info['n_num_features']}, Cat features: {info['n_cat_features']}")
    print(f"  Tier: {tier} (samples={sample_size or 'ALL'}, epochs={tier_cfg['steps']})")
    print(f"  Device: {device}")
    print("=" * 60)

    raw_config = build_config(stage, device, tier)

    if deterministic:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

    batch_size = raw_config["train"]["main"]["batch_size"]

    train_data = TabDiffDataset(
        dataname, data_dir, info,
        y_only=False, isTrain=True,
        dequant_dist=raw_config["data"]["dequant_dist"],
        int_dequant_factor=raw_config["data"]["int_dequant_factor"],
    )

    if sample_size is not None and sample_size < len(train_data):
        rng = np.random.RandomState(42)
        subset_idx = rng.choice(len(train_data), size=sample_size, replace=False).tolist()
        train_subset = Subset(train_data, subset_idx)
        print(f"[data] Subsampled {sample_size}/{len(train_data)} training rows for tier '{tier}'")
    else:
        train_subset = train_data

    train_loader = DataLoader(
        train_subset, batch_size=batch_size,
        shuffle=True, num_workers=0,
    )

    val_data = TabDiffDataset(
        dataname, data_dir, info,
        y_only=False, isTrain=False,
        dequant_dist=raw_config["data"]["dequant_dist"],
        int_dequant_factor=raw_config["data"]["int_dequant_factor"],
    )

    d_numerical = train_data.d_numerical
    categories = train_data.categories

    print(f"[data] d_numerical={d_numerical}, categories={categories}")
    print(f"[data] Train size={len(train_data)}, Test size={len(val_data)}")

    # ---- 构建模型 ----
    raw_config["unimodmlp_params"]["d_numerical"] = d_numerical
    raw_config["unimodmlp_params"]["categories"] = (categories + 1).tolist() if len(categories) > 0 else []

    backbone = UniModMLP(**raw_config["unimodmlp_params"])
    model = Model(backbone, **raw_config["diffusion_params"]["edm_params"])
    model.to(device)

    # Causal warmup: first 20% of epochs have linearly increasing causal penalty
    total_epochs = raw_config["train"]["main"]["steps"]
    causal_warmup_steps = int(total_epochs * 0.2)

    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=model,
        y_only_model=None,
        **raw_config["diffusion_params"],
        device=torch.device(device),
        causal_weight_max=float(lambda_causal),
        causal_warmup_steps=causal_warmup_steps,
    )

    num_params = sum(p.numel() for p in diffusion.parameters())
    print(f"[model] Parameters: {num_params:,}")
    diffusion.to(device)
    diffusion.train()

    # ---- 加载预计算因果掩码 ----
    if use_causal_masks:
        num_mask, cat_mask = load_precomputed_causal_masks(
            data_dir, device, mask_subdir=mask_subdir
        )
    else:
        num_mask, cat_mask = None, None
        print("[mask] use_causal_masks=False, skipping NOTEARS mask files")

    # ---- 保存路径 ----
    model_save_path = str(CDT_ROOT / "ckpt" / dataname / exp_name)
    result_save_path = str(CDT_ROOT / "result" / dataname / exp_name)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(result_save_path, exist_ok=True)
    raw_config["model_save_path"] = model_save_path
    raw_config["result_save_path"] = result_save_path

    # ---- Wandb (默认禁用) ----
    try:
        import wandb  # type: ignore[import-untyped]
        logger = wandb.init(
            project=f"hierarchical_cdt_stage{stage}",
            name=exp_name,
            config=raw_config,
            mode="disabled" if no_wandb else "online",
        )
    except ImportError:
        class DummyLogger:
            def log(self, *a, **kw): pass
            def define_metric(self, *a, **kw): pass
        logger = DummyLogger()

    # ---- 创建 Metrics (简化版，只在训练时用 density) ----
    real_data_path = str(CDT_ROOT / "synthetic" / dataname / "real.csv")
    test_data_path = str(CDT_ROOT / "synthetic" / dataname / "test.csv")

    try:
        from tabdiff.metrics import TabMetrics
        metrics = TabMetrics(
            real_data_path, test_data_path, None, info, device,
            metric_list=["density"]
        )
    except Exception as e:
        print(f"[warn] Metrics init failed: {e}, using dummy metrics")
        class DummyMetrics:
            def __init__(self, n, info_dict):
                self.real_data_size = n
                self.info = info_dict
            def evaluate(self, *a, **kw):
                return {}, {}
            def plot_density(self, *a, **kw):
                return None
        metrics = DummyMetrics(len(train_data), info)

    # ---- 创建 Trainer ----
    sample_batch_size = raw_config["sample"]["batch_size"]
    pre_masks = (num_mask, cat_mask) if use_causal_masks else None

    trainer = HierarchicalTrainer(
        diffusion,
        train_loader,
        train_data,
        val_data,
        metrics,
        logger,
        **raw_config["train"]["main"],
        sample_batch_size=sample_batch_size,
        model_save_path=model_save_path,
        result_save_path=result_save_path,
        device=device,
        ckpt_path=None,
        y_only=False,
        precomputed_masks=pre_masks,
    )

    # ---- 保存配置 ----
    with open(os.path.join(model_save_path, "config.pkl"), "wb") as f:
        pickle.dump(raw_config, f)

    # ---- 开始训练 ----
    n_epochs = raw_config["train"]["main"]["steps"]
    n_data = sample_size or len(train_data)
    print(f"\n[train] Stage {stage} [{tier}]: {n_epochs} epochs x {n_data} samples")
    trainer.run_loop()

    print(f"\n[done] Stage {stage} [{tier}] training complete!")
    print(f"  Checkpoints: {model_save_path}")
    print(f"  Results: {result_save_path}")


# ============================================================
# 5. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Hierarchical CausalDiffTab Training")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3],
                        help="Training stage: 1=spatiotemporal, 2=context, 3=full conditional")
    parser.add_argument("--tier", type=str, default="quick",
                        choices=["quick", "balanced", "full"],
                        help="Training tier: quick(500/10) balanced(2000/200) full(all/4000)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no_wandb", action="store_true", default=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--experiment_id",
        type=str,
        default=None,
        help="实验子目录后缀，用于区分不同 YAML 管线（默认不加后缀，与旧路径兼容）",
    )
    parser.add_argument(
        "--lambda_causal",
        type=float,
        default=1.0,
        help="因果正则权重 causal_weight_max，0 关闭",
    )
    parser.add_argument(
        "--no_causal_masks",
        action="store_true",
        help="不加载预计算 NOTEARS 掩码（无因果结构先验）",
    )
    parser.add_argument(
        "--mask_subdir",
        type=str,
        default="causal_masks",
        help="掩码子目录名 (默认 causal_masks=binary; soft 实验传 causal_masks_soft)",
    )
    parser.add_argument(
        "--dataname",
        type=str,
        default=None,
        help="Stage 3 数据目录名（位于 data/ 下），例如 nyc_crash_2024；Stage 1/2 会映射为 nyc_stage1_2024/nyc_stage2_2024",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        args.device = "cpu"

    train_stage(
        stage=args.stage,
        device=args.device,
        tier=args.tier,
        no_wandb=args.no_wandb,
        deterministic=args.deterministic,
        experiment_id=args.experiment_id,
        lambda_causal=args.lambda_causal,
        use_causal_masks=not args.no_causal_masks,
        mask_subdir=args.mask_subdir,
        dataname=args.dataname,
    )


if __name__ == "__main__":
    main()
