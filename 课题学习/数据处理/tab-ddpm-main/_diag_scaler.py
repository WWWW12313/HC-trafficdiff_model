"""诊断: 对比生成样本与真实数据在高斯空间的分布差异"""
import joblib, json, numpy as np, torch, warnings, sys
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from tab_ddpm.modules import MLPDiffusion
from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion

meta = joblib.load('causal_scaler.pkl')
scaler = meta['scaler']
num_indices = list(meta['num_indices'])
cat_indices = list(meta['cat_indices'])
topo = meta['topological_order']
encoders = meta['encoders']
cat_cols_topo = [topo[i] for i in cat_indices]
num_cols_topo = [topo[i] for i in num_indices]
nca = np.array([len(encoders[c].categories_[0]) for c in cat_cols_topo], dtype=int)

d_in = len(num_indices) + int(nca.sum())
state = torch.load('runs/M4_CausalDDPM_v4_s3000_cuda_20260309_133006/causal_ddpm_best.pt',
                    map_location='cpu', weights_only=False)
model = MLPDiffusion(d_in=d_in, num_classes=0, is_y_cond=False,
                     rtdl_params={'d_layers': [512, 512, 512], 'dropout': 0.0})
diff = GaussianMultinomialDiffusion(
    num_classes=nca, num_numerical_features=len(num_indices),
    denoise_fn=model, num_timesteps=1000)
diff.load_state_dict(state)
diff.eval()

x_gen, _ = diff.sample_all(num_samples=5000, batch_size=256,
                            y_dist=torch.tensor([1.0]), ddim=False)
x_num_norm = x_gen.cpu().numpy()[:, :len(num_indices)]

print('=== 生成样本 高斯空间(Quantile变换后) ===')
for i, c in enumerate(num_cols_topo):
    v = x_num_norm[:, i]
    print('  %-22s mean=%7.3f std=%7.3f  [%7.2f, %7.2f]  |>3|=%.1f%%'
          % (c, v.mean(), v.std(), v.min(), v.max(), 100*np.mean(np.abs(v) > 3)))

# 注意: scaler 是在 CausalTabularDataset 里 fit 的, 列顺序是 topo 排序的 num_cols
# 而 npy 里的列顺序是 info['num_columns']
# 需要确认两者对不对得上
info = json.load(open('data/nyc_crash_c4/info.json', encoding='utf-8'))
print('\nscaler fitted on topo num cols:', num_cols_topo)
print('npy num cols (info.json):', info['num_columns'])

x_raw = np.load('data/nyc_crash_c4/X_num_train.npy')
print('\nX_num_train shape:', x_raw.shape)

# scaler.n_features_in_ 看看
print('scaler n_features_in_:', scaler.n_features_in_)

# 如果列顺序不一致，scaler.transform 会产生错误的逆变换
# 在 eval 脚本中: scaler.inverse_transform(x_num_norm) 也可能列序不匹配!
# 对比: 真实数据经 scaler 变换到高斯空间 (注意: 这里列顺序是 info['num_columns'])
try:
    x_t = scaler.transform(x_raw)
    print('\n=== 真实数据 高斯空间 (scaler.transform, 列序=info) ===')
    for i, c in enumerate(info['num_columns']):
        v = x_t[:, i]
        print('  %-22s mean=%7.3f std=%7.3f  [%7.2f, %7.2f]  |>3|=%.1f%%'
              % (c, v.mean(), v.std(), v.min(), v.max(), 100*np.mean(np.abs(v) > 3)))
except Exception as e:
    print('scaler.transform failed:', e)

# 检查 inverse_transform 是否匹配
print('\n=== 验证: scaler.inverse_transform(生成高斯值) 结果范围 ===')
x_inv = scaler.inverse_transform(x_num_norm)
for i, c in enumerate(num_cols_topo):
    v = x_inv[:, i]
    print('  %-22s mean=%10.3f std=%10.3f  [%10.2f, %10.2f]'
          % (c, v.mean(), v.std(), v.min(), v.max()))

# 真实原始数据范围
print('\n=== 真实数据 原始值范围 (X_num_train.npy) ===')
for i, c in enumerate(info['num_columns']):
    v = x_raw[:, i]
    print('  %-22s mean=%10.3f std=%10.3f  [%10.2f, %10.2f]'
          % (c, v.mean(), v.std(), v.min(), v.max()))
