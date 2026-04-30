"""
诊断脚本：追踪反向扩散过程中的 z_norm 统计量。
目的：找出 z_norm 从 N(0,1) 发散到 std=949 的具体位置。
"""
import sys, os, json
import numpy as np
import torch
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tab_ddpm.modules import MLPDiffusion
from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tab_dir = os.path.dirname(os.path.abspath(__file__))

# --- 加载元信息 ---
meta = joblib.load(os.path.join(tab_dir, "causal_scaler.pkl"))
topo = meta["topological_order"]
cat_indices = list(meta["cat_indices"])
num_indices = list(meta["num_indices"])
encoders = meta["encoders"]
cat_cols_topo = [topo[i] for i in cat_indices]
num_cols_topo = [topo[i] for i in num_indices]
num_classes_array = np.array(
    [len(encoders[col].categories_[0]) for col in cat_cols_topo], dtype=int
)
d_in_total = len(num_indices) + int(num_classes_array.sum())

# --- 找最新 checkpoint ---
runs_dir = os.path.join(tab_dir, "runs")
best_run = None
for d in sorted(os.listdir(runs_dir), reverse=True):
    ckpt = os.path.join(runs_dir, d, "causal_ddpm_best.pt")
    if os.path.exists(ckpt):
        best_run = os.path.join(runs_dir, d)
        break

print(f"Run: {best_run}")
checkpoint_path = os.path.join(best_run, "causal_ddpm_best.pt")
state = torch.load(checkpoint_path, map_location=device, weights_only=False)

# 推断 d_layers
inferred_layers = []
for key, value in state.items():
    if key.startswith("_denoise_fn.mlp.blocks.") and key.endswith(".linear.weight"):
        try:
            block_idx = int(key.split(".")[3])
        except Exception:
            continue
        inferred_layers.append((block_idx, int(value.shape[0])))
d_layers = [dim for _, dim in sorted(inferred_layers)]
print(f"d_layers: {d_layers}")

model = MLPDiffusion(
    d_in=d_in_total, num_classes=0, is_y_cond=False,
    rtdl_params={"d_layers": d_layers, "dropout": 0.0},
).to(device)

diffusion = GaussianMultinomialDiffusion(
    num_classes=num_classes_array,
    num_numerical_features=len(num_indices),
    denoise_fn=model,
    num_timesteps=1000,
    device=device,
).to(device)
diffusion.load_state_dict(state, strict=True)
diffusion.eval()

# --- 诊断 1: 检查噪声调度参数 ---
print("\n=== 噪声调度参数 ===")
ac = diffusion.alphas_cumprod.cpu().numpy()
print(f"alphas_cumprod[0]   = {ac[0]:.6f}  (t=0: 几乎无噪声)")
print(f"alphas_cumprod[499] = {ac[499]:.6f}  (t=500: 中间)")
print(f"alphas_cumprod[999] = {ac[999]:.6f}  (t=999: 几乎全噪声)")
print(f"sqrt_alphas_cumprod range: [{diffusion.sqrt_alphas_cumprod.min():.6f}, {diffusion.sqrt_alphas_cumprod.max():.6f}]")
print(f"sqrt_one_minus_alphas_cumprod range: [{diffusion.sqrt_one_minus_alphas_cumprod.min():.6f}, {diffusion.sqrt_one_minus_alphas_cumprod.max():.6f}]")

betas = (1. - diffusion.alphas).cpu().numpy()
print(f"betas range: [{betas.min():.6f}, {betas.max():.6f}]")
pv = diffusion.posterior_variance.cpu().numpy()
print(f"posterior_variance range: [{pv.min():.6f}, {pv.max():.6f}]")

# --- 诊断 2: 模型输出检查 ---
print("\n=== 模型输出诊断 (随机输入) ===")
B = 64
z_test = torch.randn(B, len(num_indices), device=device)  # N(0,1)

has_cat = num_classes_array[0] != 0
if has_cat:
    uniform_logits = torch.zeros((B, int(num_classes_array.sum())), device=device)
    log_z_test = diffusion.log_sample_categorical(uniform_logits)
else:
    log_z_test = torch.zeros((B, 0), device=device)

for t_val in [999, 500, 100, 10, 0]:
    t = torch.full((B,), t_val, device=device, dtype=torch.long)
    with torch.no_grad():
        x_in = torch.cat([z_test, log_z_test], dim=1).float()
        model_out = model(x_in, t)
    
    out_num = model_out[:, :len(num_indices)]
    print(f"\nt={t_val:4d}:")
    print(f"  model input  z_norm: mean={z_test.mean():.3f}, std={z_test.std():.3f}")
    print(f"  model output (eps) : mean={out_num.mean():.4f}, std={out_num.std():.4f}, "
          f"range=[{out_num.min():.2f}, {out_num.max():.2f}]")

# --- 诊断 3: 单步反向扩散追踪 ---
print("\n\n=== 反向扩散追踪 (10 样本) ===")
B = 10
z_norm = torch.randn((B, len(num_indices)), device=device)
if has_cat:
    uniform_logits = torch.zeros((B, int(num_classes_array.sum())), device=device)
    log_z = diffusion.log_sample_categorical(uniform_logits)
else:
    log_z = torch.zeros((B, 0), device=device)

y_dist = torch.tensor([1.0], device=device)
y = torch.multinomial(y_dist, num_samples=B, replacement=True)
out_dict = {'y': y.long().to(device)}

check_steps = [999, 998, 997, 990, 950, 900, 800, 500, 200, 100, 50, 10, 5, 1, 0]
with torch.no_grad():
    for i in reversed(range(0, 1000)):
        t = torch.full((B,), i, device=device, dtype=torch.long)
        model_out = diffusion._denoise_fn(
            torch.cat([z_norm, log_z], dim=1).float(),
            t,
            **out_dict
        )
        model_out_num = model_out[:, :diffusion.num_numerical_features]
        model_out_cat = model_out[:, diffusion.num_numerical_features:]
        
        old_z = z_norm.clone()
        result = diffusion.gaussian_p_sample(model_out_num, z_norm, t, clip_denoised=True)
        z_norm = result['sample']
        pred_x0 = result['pred_xstart']
        
        if has_cat:
            log_z = diffusion.p_sample(model_out_cat, log_z, t, out_dict)
        
        if i in check_steps:
            z_np = z_norm.cpu().numpy()
            pred_np = pred_x0.cpu().numpy()
            eps_np = model_out_num.cpu().numpy()
            print(f"\nt={i:4d}:")
            print(f"  z_norm     : mean={z_np.mean():.3f}, std={z_np.std():.3f}, "
                  f"range=[{z_np.min():.2f}, {z_np.max():.2f}]")
            print(f"  pred_x0    : mean={pred_np.mean():.3f}, std={pred_np.std():.3f}, "
                  f"range=[{pred_np.min():.2f}, {pred_np.max():.2f}]")
            print(f"  eps (model): mean={eps_np.mean():.3f}, std={eps_np.std():.3f}, "
                  f"range=[{eps_np.min():.2f}, {eps_np.max():.2f}]")
            # Check per-feature stats for z_norm
            for fi in range(min(3, len(num_indices))):
                col_vals = z_np[:, fi]
                print(f"    Feature {fi} ({num_cols_topo[fi]}): "
                      f"mean={col_vals.mean():.3f}, std={col_vals.std():.3f}")

print("\n=== 最终输出 ===")
z_final = z_norm.cpu().numpy()
for fi in range(len(num_indices)):
    col_vals = z_final[:, fi]
    print(f"  {num_cols_topo[fi]:25s}: mean={col_vals.mean():.3f}, std={col_vals.std():.3f}, "
          f"range=[{col_vals.min():.2f}, {col_vals.max():.2f}]")
