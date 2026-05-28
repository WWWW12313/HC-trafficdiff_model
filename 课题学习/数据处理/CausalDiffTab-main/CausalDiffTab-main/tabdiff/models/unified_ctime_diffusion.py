import torch.nn.functional as F
import torch
import math
import numpy as np
from tabdiff.models.noise_schedule import *
from tqdm import tqdm
from itertools import chain

"""
“Our implementation of the continuous-time masked diffusion is inspired by https://arxiv.org/abs/2406.07524's implementation at [https://github.com/kuleshov-group/mdlm], with modifications to support data distributions that include categorical dimensions of different sizes.”
"""

S_churn= 1
S_min=0
S_max=float('inf')
S_noise=1

class UnifiedCtimeDiffusion(torch.nn.Module):
    def __init__(
            self,
            num_classes: np.array,
            num_numerical_features: int,
            denoise_fn,
            y_only_model,
            num_timesteps=1000,
            scheduler='power_mean',
            cat_scheduler='log_linear',
            noise_dist='uniform',
            edm_params={},
            noise_dist_params={},
            noise_schedule_params={},
            sampler_params={},
            device=torch.device('cpu'),
            causal_weight_max: float = 1.0,    # 新增参数
            causal_warmup_steps: int = 4000,   # 新增参数
            causal_start_step: int = 0,
            causal_mask_mode: str = "allowed_penalty",
            causal_guidance_scale: float = 0.0,
            cg_num_mask = None,
            cg_cat_mask = None,
            macro_relation_weight: float = 0.0,
            macro_injury_idx: int = None,
            macro_group_indices = None,
            cond_cat_indices = None,
            macro_guidance_scale: float = 0.0,
            macro_guidance_start_step: int = 10,
            macro_guidance_group_means: dict = None,
            macro_guidance_mode: str = "absolute",  # "absolute" | "relative" | "adaptive"
            macro_guidance_adaptive_drift_threshold: float = 2.0,
            **kwargs
        ):

        super(UnifiedCtimeDiffusion, self).__init__()

        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes # it as a vector [K1, K2, ..., Km]
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate([num_classes[i].repeat(num_classes[i]) for i in range(len(num_classes))])
        ).to(device) if len(num_classes)>0 else torch.tensor([]).to(device).int()
        self.mask_index = torch.tensor(self.num_classes).long().to(device)
        self.neg_infinity = -1000000.0 
        self.num_classes_w_mask = tuple(self.num_classes + 1)

        offsets = np.cumsum(self.num_classes)
        offsets = np.append([0], offsets)
        self.slices_for_classes = []
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        self.offsets = torch.from_numpy(offsets).to(device)
        
        offsets = np.cumsum(self.num_classes) + np.arange(1, len(self.num_classes)+1)
        offsets = np.append([0], offsets)
        self.slices_for_classes_with_mask = []
        for i in range(1, len(offsets)):
            self.slices_for_classes_with_mask.append(np.arange(offsets[i - 1], offsets[i]))

        self._denoise_fn = denoise_fn
        self.y_only_model = y_only_model
        self.num_timesteps = num_timesteps
        self.scheduler = scheduler
        self.cat_scheduler = cat_scheduler
        self.noise_dist = noise_dist
        self.edm_params = edm_params
        self.noise_dist_params = noise_dist_params
        self.sampler_params = sampler_params
        self.causal_weight_max = causal_weight_max
        self.causal_warmup_steps = causal_warmup_steps
        self.causal_start_step = causal_start_step
        self.causal_mask_mode = causal_mask_mode
        self.causal_guidance_scale = causal_guidance_scale
        self.cg_num_mask = cg_num_mask
        self.cg_cat_mask = cg_cat_mask
        self.macro_relation_weight = macro_relation_weight
        self.macro_injury_idx = macro_injury_idx
        self.macro_group_indices = macro_group_indices
        self.cond_cat_indices = cond_cat_indices
        self.macro_guidance_scale = macro_guidance_scale
        self.macro_guidance_start_step = macro_guidance_start_step
        self.macro_guidance_group_means = macro_guidance_group_means or {}
        self.macro_guidance_mode = macro_guidance_mode
        self.macro_guidance_adaptive_drift_threshold = macro_guidance_adaptive_drift_threshold
        self.ema_loss = None
        self.w_num = 0.0
        self.w_cat = 0.0
        self.num_mask_idx = []
        self.cat_mask_idx = []

        self.device = device
        self.register_buffer('num_causal_mask', None)
        self.register_buffer('cat_causal_mask', None)
        if self.scheduler == 'power_mean':
            self.num_schedule = PowerMeanNoise(**noise_schedule_params)
        elif self.scheduler == 'power_mean_per_column':
            self.num_schedule = PowerMeanNoise_PerColumn(num_numerical = num_numerical_features, **noise_schedule_params)
        else:
            raise NotImplementedError(f"The noise schedule--{self.scheduler}-- is not implemented for contiuous data at CTIME ")
        
        if self.cat_scheduler == 'log_linear':
            self.cat_schedule = LogLinearNoise(**noise_schedule_params)
        elif self.cat_scheduler == 'log_linear_per_column':
            self.cat_schedule = LogLinearNoise_PerColumn(num_categories = len(num_classes), **noise_schedule_params)
        else:
            raise NotImplementedError(f"The noise schedule--{self.cat_scheduler}-- is not implemented for discrete data at CTIME ")
        
    def set_causal_masks(self, num_causal_mask, cat_causal_mask):
        """设置因果掩码（支持 numpy 数组或 PyTorch 张量）"""
        if isinstance(num_causal_mask, np.ndarray):
            num_causal_mask = torch.from_numpy(num_causal_mask).float()  # 先转为 Tensor
        else:
            num_causal_mask = num_causal_mask.float()       
        if isinstance(cat_causal_mask, np.ndarray):
            cat_causal_mask = torch.from_numpy(cat_causal_mask).float()
        else:
            cat_causal_mask = cat_causal_mask.float()
            
    # def set_causal_masks(self, num_causal_mask):
    #     """设置因果掩码（支持 numpy 数组或 PyTorch 张量）"""
    #     if isinstance(num_causal_mask, np.ndarray):
    #         num_causal_mask = torch.from_numpy(num_causal_mask).float()  # 先转为 Tensor
    #     else:
    #         num_causal_mask = num_causal_mask.float()       
        # 注册为 buffer 并移动到设备
        self.register_buffer('num_causal_mask', num_causal_mask.to(self.device))
        self.register_buffer('cat_causal_mask', cat_causal_mask.to(self.device))
    
    # def _causal_regularization_loss(self, pred_x, x_t, noise, sigma, is_categorical=False, current_step=0):
    # # """
    # # 通用因果正则化损失（支持数值型和类别型）
    # # Args:
    # #     pred_x: 模型预测的干净数据 (batch, features)
    # #     x_t: 带噪声的数据 (batch, features)
    # #     noise: 添加的噪声（数值型为高斯噪声，类别型为转移概率）
    # #     sigma: 噪声水平 (batch, features)
    # #     is_categorical: 是否为类别型特征
    # #     current_step: 当前训练步数（用于动态权重计算）
    # # """
    #     batch_size = pred_x.size(0)
    #     num_features = pred_x.size(1)

    # # 动态权重计算（课程学习策略）
    # # 参数建议：在类初始化时设置 self.causal_weight_max, self.causal_warmup_steps
    #     if current_step < self.causal_warmup_steps:
    #         weight = self.causal_weight_max * (1 - math.cos(math.pi * current_step / self.causal_warmup_steps)) / 2
    #     else:
    #         weight = self.causal_weight_max

    #     if is_categorical:
    #     # 扩展 sigma 到 one-hot 维度
    #         sigma_expanded = sigma.repeat_interleave(
    #             torch.tensor(self.num_classes, device=sigma.device),
    #             dim=1
    #         )
        
    #     # 计算类别型噪声误差
    #         log_probs = F.log_softmax(pred_x, dim=-1)
    #         target = x_t.argmax(dim=-1)
    #         # noise_loss = F.nll_loss(log_probs, target)
        
    #     # 计算因果方向导数
    #         probs = torch.exp(log_probs)
    #         grad_matrix = torch.einsum('bi,bj->bij', probs, probs)

    #     else:
    #     # 数值型噪声误差
    #         noise_pred = (x_t - pred_x) / sigma
    #         # noise_loss = F.mse_loss(noise_pred, noise)
        
    #     # 计算因果方向导数
    #         grad_matrix = torch.einsum('bi,bj->bij', noise_pred, noise_pred)
    
    # # 应用因果掩码
    #     mask = self.cat_causal_mask.unsqueeze(0) if is_categorical else self.num_causal_mask.unsqueeze(0)
    #     masked_grad = grad_matrix * mask
    
    # # 因果一致性损失
    #     causal_consistency = masked_grad.mean()
    # # 总损失 = 基础噪声损失 + 加权因果约束
    #     total_loss = causal_consistency * weight
    
    #     return total_loss
    def _effective_causal_mask(self, is_categorical=False):
        mask = self.cat_causal_mask if is_categorical else self.num_causal_mask
        if mask is None:
            return None
        if self.causal_mask_mode == "allowed_penalty":
            return mask
        if self.causal_mask_mode != "forbidden_penalty":
            raise ValueError(f"Unsupported causal_mask_mode: {self.causal_mask_mode}")

        valid = torch.ones_like(mask)
        if is_categorical:
            for slice_i in self.slices_for_classes_with_mask:
                idx = torch.as_tensor(slice_i, dtype=torch.long, device=mask.device)
                valid[idx[:, None], idx[None, :]] = 0.0
        else:
            valid.fill_diagonal_(0.0)
        allowed = (mask > 0).to(mask.dtype)
        return valid * (1.0 - allowed)

    def _causal_regularization_loss(self, pred_x, x_t, noise, sigma, is_categorical=False, current_step=0):
    # """
    # 通用因果正则化损失（支持数值型和类别型）
    # Args:
    #     pred_x: 模型预测的干净数据 (batch, features)
    #     x_t: 带噪声的数据 (batch, features)
    #     noise: 添加的噪声（数值型为高斯噪声，类别型为转移概率）
    #     sigma: 噪声水平 (batch, features)
    #     is_categorical: 是否为类别型特征
    #     current_step: 当前训练步数（用于动态权重计算）
    # """
        # batch_size = pred_x.size(0)
        # num_features = pred_x.size(1)

    # 计算因果方向导数和基础损失（原有逻辑保持不变）
        if is_categorical:
        # 扩展 sigma 到 one-hot 维度
            # sigma_expanded = sigma.repeat_interleave(
            #     torch.tensor(self.num_classes, device=sigma.device),
            #     dim=1
            # )
        # 计算类别型噪声误差
            log_probs = F.log_softmax(pred_x, dim=-1)
            target = x_t.argmax(dim=-1)
        # 计算因果方向导数
            probs = torch.exp(log_probs)
            grad_matrix = torch.einsum('bi,bj->bij', probs, probs)
        else:
        # 数值型噪声误差
            noise_pred = (x_t - pred_x) / sigma
        # 计算因果方向导数
            grad_matrix = torch.einsum('bi,bj->bij', noise_pred, noise_pred)
    
    # 应用因果掩码；默认保留旧语义，新实验可改为只惩罚未保留的跨特征边。
        mask = self._effective_causal_mask(is_categorical=is_categorical)
        if mask is None or torch.sum(mask) <= 0:
            return torch.tensor(0.0, device=pred_x.device)
        masked_grad = grad_matrix * mask.unsqueeze(0)
        causal_consistency = masked_grad.sum() / (mask.sum() * grad_matrix.shape[0]).clamp_min(1.0)
        total_loss = causal_consistency * 1
        return total_loss
    
    def _macro_relation_loss(self, real_x_num, pred_x_num, real_x_cat):
        """
        Macro Relation Consistency Loss (v2):
        - 多维 group 组合（细粒度分组，减少记忆空间）
        - Global mean 正则化（防止 group mean 系统性偏离）
        - Relative error 支持（对大/小 group 更公平）
        """
        if self.macro_injury_idx is None or not self.macro_group_indices:
            return torch.tensor(0.0, device=real_x_num.device)
        
        injury_real = real_x_num[:, self.macro_injury_idx]
        injury_pred = pred_x_num[:, self.macro_injury_idx]
        
        # 多维 group 组合：多个 categorical 列拼接为唯一 group_id
        group_id = real_x_cat[:, self.macro_group_indices[0]].long().flatten()
        for gidx in self.macro_group_indices[1:]:
            # 混合基数编码：group_id = group_id * n_categories + next_col
            max_val = int(real_x_cat[:, gidx].max().item()) + 1
            group_id = group_id * max(2, max_val) + real_x_cat[:, gidx].long().flatten()
        
        # 确保 group_id 非负且有效
        if group_id.min() < 0:
            return F.mse_loss(injury_pred.mean(), injury_real.mean().detach())
        
        num_groups = int(group_id.max().item()) + 1
        if num_groups <= 1:
            return F.mse_loss(injury_pred.mean(), injury_real.mean().detach())
        
        # scatter_add 计算每组和与计数
        real_sum = torch.zeros(num_groups, device=real_x_num.device).scatter_add(0, group_id, injury_real)
        pred_sum = torch.zeros(num_groups, device=real_x_num.device).scatter_add(0, group_id, injury_pred)
        counts = torch.zeros(num_groups, device=real_x_num.device).scatter_add(0, group_id, torch.ones_like(injury_real))
        
        mask = counts > 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=real_x_num.device)
        
        real_mean = real_sum[mask] / counts[mask]
        pred_mean = pred_sum[mask] / counts[mask]
        
        # 加权 MSE（大组权重更高，避免小组噪声主导）
        weights = counts[mask] / counts[mask].sum()
        loss = (weights * (pred_mean - real_mean.detach()) ** 2).sum()
        return loss
    
    def mixed_loss(self, x, current_training_step=None):
        b = x.shape[0]
        device = x.device

        x_num = x[:, :self.num_numerical_features]
        x_cat = x[:, self.num_numerical_features:].long()
        # Sample noise level
        if self.noise_dist == "uniform_t":
            t = torch.rand(b, device=device, dtype=x_num.dtype)
            t = t[:, None]
            sigma_num = self.num_schedule.total_noise(t)
            sigma_cat = self.cat_schedule.total_noise(t)
            dsigma_cat = self.cat_schedule.rate_noise(t)
        else:
            sigma_num = self.sample_ctime_noise(x)       
            t = self.num_schedule.inverse_to_t(sigma_num)
            while torch.any((t < 0) + (t > 1)):     
                # restrict t to [0,1]
                # this iterative approach is equivalent to sampling from a truncated version of the orignal noise distribution
                invalid_idx = ((t < 0) + (t > 1)).nonzero().squeeze(-1)
                sigma_num[invalid_idx] = self.sample_ctime_noise(x[:len(invalid_idx)])
                t = self.num_schedule.inverse_to_t(sigma_num)
            assert not torch.any((t < 0) + (t > 1))
            sigma_cat = self.cat_schedule.total_noise(t)
        # Convert sigma_cat to the corresponding alpha and move_chance
        alpha = torch.exp(-sigma_cat)
        move_chance = -torch.expm1(-sigma_cat)      # torch.expm1 gives better numertical stability
            
        # Continuous forward diff
        x_num_t = x_num
        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            x_num_t = x_num + noise * sigma_num
        
        # Discrete forward diff
        x_cat_t = x_cat
        x_cat_t_soft = x_cat # in the case where x_cat is empty, x_cat_t_soft will have the same shape as x_cat
        if x_cat.shape[1] > 0:
            is_learnable = self.cat_scheduler == 'log_linear_per_column'
            strategy = 'soft'if is_learnable else 'hard'
            x_cat_t, x_cat_t_soft = self.q_xt(x_cat, move_chance, strategy=strategy)

        # Predict orignal data (distribution)
        model_out_num, model_out_cat = self._denoise_fn(   
            x_num_t, x_cat_t_soft,
            t.squeeze(), sigma=sigma_num
        )
        d_loss = torch.zeros((1,)).float()
        c_loss = torch.zeros((1,)).float()

        if x_num.shape[1] > 0:
            c_loss = self._edm_loss(model_out_num, x_num, sigma_num)
        if x_cat.shape[1] > 0:
            logits = self._subs_parameterization(model_out_cat, x_cat_t)    # log normalized probabilities, with the entry mask category being set to -inf
            d_loss = self._absorbed_closs(logits, x_cat, sigma_cat, dsigma_cat)
            # 排除条件 cat 列的扩散 loss（proxy 列改为条件输入）
            if self.cond_cat_indices is not None and d_loss.ndim > 1:
                mask = torch.ones(d_loss.shape[1], device=d_loss.device)
                mask[self.cond_cat_indices] = 0.0
                d_loss = d_loss * mask
        cat_causal_loss = torch.tensor(0.0, device=x.device)
        causal_loss = torch.tensor(0.0, device=x.device)
        if self.cat_causal_mask is not None and x_cat.shape[1] > 0:
            cat_causal_loss = self._causal_regularization_loss(
                model_out_cat, x_cat_t_soft, noise=None, sigma=sigma_cat, is_categorical=True, current_step=current_training_step
                )
        if self.num_causal_mask is not None and x_num.shape[1] > 0:
            causal_loss = self._causal_regularization_loss(
                model_out_num, x_num_t, noise, sigma_num, current_step=current_training_step
            )

        # Causal penalty schedule: optional delay, then linear ramp to causal_weight_max.
        if current_training_step is not None:
            step_after_start = float(current_training_step) - float(self.causal_start_step)
            if step_after_start <= 0:
                causal_lambda = 0.0
            elif self.causal_warmup_steps > 0:
                causal_lambda = self.causal_weight_max * min(
                    1.0, step_after_start / float(self.causal_warmup_steps)
                )
            else:
                causal_lambda = self.causal_weight_max
        else:
            causal_lambda = self.causal_weight_max

        macro_loss = torch.tensor(0.0, device=x.device)
        if self.macro_relation_weight > 0 and x_num.shape[1] > 0:
            macro_loss = self._macro_relation_loss(x_num, model_out_num, x_cat)

        return (d_loss.mean() + causal_lambda * cat_causal_loss,
                c_loss.mean() + causal_lambda * causal_loss + self.macro_relation_weight * macro_loss)

    @torch.no_grad()
    def sample(self, num_samples):
        b = num_samples
        device = self.device
        dtype = torch.float32
        
        # Create the chain of t
        t = torch.linspace(0,1,self.num_timesteps, dtype=dtype, device=device)      # times = 0.0,...,1.0
        t = t[:, None]
        
        # Compute the chains of sigma
        sigma_num_cur = self.num_schedule.total_noise(t)
        sigma_cat_cur = self.cat_schedule.total_noise(t)
        sigma_num_next = torch.zeros_like(sigma_num_cur)
        sigma_num_next[1:] = sigma_num_cur[0:-1]
        sigma_cat_next = torch.zeros_like(sigma_cat_cur)
        sigma_cat_next[1:] = sigma_cat_cur[0:-1]
        
        # Prepare sigma_hat for stochastic sampling mode
        if self.sampler_params['stochastic_sampler']:
            gamma = min(S_churn / self.num_timesteps, np.sqrt(2) - 1) * (S_min <= sigma_num_cur) * (sigma_num_cur <= S_max)
            sigma_num_hat = sigma_num_cur + gamma * sigma_num_cur
            t_hat = self.num_schedule.inverse_to_t(sigma_num_hat)
            t_hat = torch.min(t_hat, dim=-1, keepdim=True).values    # take the samllest t_hat induced by sigma_num
            zero_gamma = (gamma==0).any()
            t_hat[zero_gamma] = t[zero_gamma]
            out_of_bound = (t_hat > 1).squeeze()
            sigma_num_hat[out_of_bound] = sigma_num_cur[out_of_bound]
            t_hat[out_of_bound] = t[out_of_bound]
            sigma_cat_hat = self.cat_schedule.total_noise(t_hat)
        else:
            t_hat = t
            sigma_num_hat = sigma_num_cur
            sigma_cat_hat = sigma_cat_cur
                
        # Sample priors for the continuous dimensions
        z_norm = torch.randn((b, self.num_numerical_features), device=device) * sigma_num_cur[-1] 
            
        # Sample priors for the discrete dimensions
        has_cat = len(self.num_classes) > 0
        z_cat = torch.zeros((b, 0), device=device).float()      # the default values for categorical sample if the dataset has no categorical entry
        if has_cat:
            z_cat = self._sample_masked_prior(
                b,
                len(self.num_classes),
            )
        
        pbar = tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps)
        pbar.set_description(f"Sampling Progress")
        for i in pbar:                  
            z_norm, z_cat, q_xs = self.edm_update(
                z_norm, z_cat, i, 
                t[i], t[i-1] if i > 0 else None, t_hat[i],
                sigma_num_cur[i], sigma_num_next[i], sigma_num_hat[i], 
                sigma_cat_cur[i], sigma_cat_next[i], sigma_cat_hat[i],
            )
        
        if not torch.all(z_cat < self.mask_index):      # catch any update result in the mask class or the dummy classes
            error_index = torch.any(z_cat >= self.mask_index, dim=-1).nonzero()
            error_z_cat = z_cat[error_index]
            error_q_xs = q_xs[error_index]
            print(error_index)
            print(error_z_cat)
            print(error_q_xs)
            pdb.set_trace()
        assert torch.all(z_cat < self.mask_index)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample
    
    def sample_all(self, num_samples, batch_size, keep_nan_samples=False):        
        b = batch_size

        all_samples = []
        num_generated = 0
        while num_generated < num_samples:
            print(f"Samples left to generate: {num_samples-num_generated}")
            sample = self.sample(b)
            mask_nan = torch.any(sample.isnan(), dim=1)
            if keep_nan_samples:
                # If the sample instances that contains Nan are decided to be kept, the row with Nan will be foreced to all zeros
                sample = sample * (~mask_nan)[:, None]
            else:
                # Otherwise the instances with Nan will be eliminated
                sample = sample[~mask_nan]

            all_samples.append(sample)
            num_generated += sample.shape[0]

        x_gen = torch.cat(all_samples, dim=0)[:num_samples]

        return x_gen
    
    def q_xt(self, x, move_chance, strategy='hard'):
        """Computes the noisy sample xt.

        Args:
        x: int torch.Tensor with shape (batch_size,
            diffusion_model_input_length), input. 
        move_chance: float torch.Tensor with shape (batch_size, 1).
        """
        if strategy == 'hard':
            move_indices = torch.rand(
            * x.shape, device=x.device) < move_chance
            xt = torch.where(move_indices, self.mask_index, x)
            xt_soft = self.to_one_hot(xt).to(move_chance.dtype)
            return xt, xt_soft
        elif strategy == 'soft':
            bs = x.shape[0]
            xt_soft = torch.zeros(bs, torch.sum(self.mask_index+1), device=x.device)
            xt = torch.zeros_like(x)
            for i in range(len(self.num_classes)):
                slice_i = self.slices_for_classes_with_mask[i]
                # set the bernoulli probabilities, which determines the "coin flip" transition to the mask class
                prob_i = torch.zeros(bs, 2, device=x.device)
                prob_i[:,0] = 1-move_chance[:,i]
                prob_i[:,-1] = move_chance[:,i]
                log_prob_i = torch.log(prob_i)
                # draw soft samples and place them back to the corresponding columns
                soft_sample_i = F.gumbel_softmax(log_prob_i, tau=0.01, hard=True)
                idx = torch.stack((x[:,i]+slice_i[0], torch.ones_like(x[:,i])*slice_i[-1]), dim=-1)
                xt_soft[torch.arange(len(idx)).unsqueeze(1), idx] = soft_sample_i
                # retrieve the hard samples
                xt[:, i] = torch.where(soft_sample_i[:,1] > soft_sample_i[:,0], self.mask_index[i], x[:,i])
            return xt, xt_soft
    
    
    def _subs_parameterization(self, unormalized_prob, xt):
        # log prob at the mask index = - infinity
        unormalized_prob = self.pad(unormalized_prob, self.neg_infinity)
        
        unormalized_prob[:, range(unormalized_prob.shape[1]), self.mask_index] += self.neg_infinity
        
        # Take log softmax on the unnormalized probabilities to the logits
        logits = unormalized_prob - torch.logsumexp(unormalized_prob, dim=-1,
                                        keepdim=True)
        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = (xt != self.mask_index)    # (bs, K)
        logits[unmasked_indices] = self.neg_infinity 
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits
    
    def pad(self, x, pad_value):
        """
        Converts a concatenated tensor of class probabilities into a padded matrix, 
        where each sub-tensor is padded along the last dimension to match the largest 
        category size (max number of classes).

        Args:
            x (Tensor): The input tensor containing concatenated probabilities for all the categories in x_cat. 
                        [bs, sum(num_classes_w_mask)]
            pad_value (float): The value filled into the dummy entries, which are padded to ensure all sub-tensors have equal size 
                            along the last dimension.

        Returns:
            Tensor: A new tensorwith
                    [bs, len(num_classes_w_mask), max(num_classes_w_mask)), num_categories]
        """
        splited = torch.split(x, self.num_classes_w_mask, dim=-1)
        max_K = max(self.num_classes_w_mask)
        padded_ = [
            torch.cat((
                t, 
                pad_value*torch.ones(*(t.shape[:-1]), max_K-t.shape[-1], dtype=t.dtype, device=t.device)
            ), dim=-1) 
        for t in splited]
        out = torch.stack(padded_, dim=-2)
        return out
    
    def to_one_hot(self, x_cat):
        x_cat_oh = torch.cat(
            [F.one_hot(x_cat[:, i], num_classes=self.num_classes[i]+1,) for i in range(len(self.num_classes))], 
            dim=-1
        )
        return x_cat_oh
    
    def _absorbed_closs(self, model_output, x0, sigma, dsigma):
        """
            alpha: (bs,)
        """
        log_p_theta = torch.gather(
            model_output, -1, x0[:, :, None]
        ).squeeze(-1)
        alpha = torch.exp(-sigma)
        if self.cat_scheduler in ['log_linear_unified', 'log_linear_per_column']:
            elbo_weight = - dsigma / torch.expm1(sigma)
        else:
            elbo_weight = -1/(1-alpha)
        
        loss = elbo_weight * log_p_theta
        return loss
    
    def _sample_masked_prior(self, *batch_dims):
        return self.mask_index[None,:] * torch.ones(    
        * batch_dims, dtype=torch.int64, device=self.mask_index.device)
        
    def _mdlm_update(self, log_p_x0, x, alpha_t, alpha_s):
        """
            # t: (bs,)
            log_p_x0: (bs, K, K_max)
            # alpha_t: (bs,)
            # alpha_s: (bs,)
            alpha_t: (bs, 1/K_cat)
            alpha_s: (bs,1/K_cat)
        """
        move_chance_t = 1 - alpha_t
        move_chance_s = 1 - alpha_s     
        move_chance_t = move_chance_t.unsqueeze(-1)
        move_chance_s = move_chance_s.unsqueeze(-1)
        assert move_chance_t.ndim == log_p_x0.ndim
        # Technically, this isn't q_xs since there's a division
        # term that is missing. This division term doesn't affect
        # the samples.
        # There is a noremalizing term is (1-\alpha_t) who's responsility is to ensure q_xs is normalized. 
        # However, omiting it won't make a difference for the Gumbel-max sampling trick in  _sample_categorical()
        q_xs = log_p_x0.exp() * (move_chance_t
                                - move_chance_s)
        q_xs[:, range(q_xs.shape[1]), self.mask_index] = move_chance_s[:, :, 0]
        
        # Important: make sure that prob of dummy classes are exactly 0
        dummy_mask = torch.tensor([[(1 if i <= mask_idx else 0) for i in range(max(self.mask_index+1))] for mask_idx in self.mask_index], device=q_xs.device)
        dummy_mask = torch.ones_like(q_xs) * dummy_mask
        q_xs *= dummy_mask
        
        _x = self._sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        
        z_cat = copy_flag * x + (1 - copy_flag) * _x
        if not torch.all(z_cat <= self.mask_index):     # catch any update result in the dummy classes
            error_index = torch.any(z_cat > self.mask_index, dim=-1).nonzero()
            error_z_cat = z_cat[error_index]
            error_q_xs = q_xs[error_index]
            print(error_index)
            print(error_z_cat)
            print(error_q_xs)
            pdb.set_trace()
        return copy_flag * x + (1 - copy_flag) * _x, q_xs

    def _sample_categorical(self, categorical_probs):
        gumbel_norm = (
            1e-10
            - (torch.rand_like(categorical_probs) + 1e-10).log())
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    
    def sample_ctime_noise(self, batch):
        if self.noise_dist == 'log_norm':
            rnd_normal = torch.randn(batch.shape[0], device=batch.device)
            sigma = (rnd_normal * self.noise_dist_params['P_std'] + self.noise_dist_params['P_mean']).exp()
        else:
            raise NotImplementedError(f"The noise distribution--{self.noise_dist}-- is not implemented for CTIME ")
        return sigma

    def _edm_loss(self, D_yn, y, sigma):
        weight = (sigma ** 2 + self.edm_params['sigma_data'] ** 2) / (sigma * self.edm_params['sigma_data']) ** 2
    
        target = y
        loss = weight * ((D_yn - target) ** 2)

        return loss
    
    def edm_update(
            self, x_num_cur, x_cat_cur, i, 
            t_cur, t_next, t_hat,
            sigma_num_cur, sigma_num_next, sigma_num_hat, 
            sigma_cat_cur, sigma_cat_next, sigma_cat_hat, 
        ):
        """
        i = T-1,...,0
        """
        cfg = self.y_only_model is not None
        
        b = x_num_cur.shape[0]
        has_cat = len(self.num_classes) > 0
        
        # Get x_num_hat by move towards the noise by a small step
        x_num_hat = x_num_cur + (sigma_num_hat ** 2 - sigma_num_cur ** 2).sqrt() * S_noise * torch.randn_like(x_num_cur)
        # Get x_cat_hat
        move_chance = -torch.expm1(sigma_cat_cur - sigma_cat_hat)    # the incremental move change is 1 - alpha_t/alpha_s = 1 - exp(sigma_s - sigma_t)
        x_cat_hat, _ = self.q_xt(x_cat_cur, move_chance) if has_cat else (x_cat_cur, x_cat_cur)

        # Get predictions
        x_cat_hat_oh = self.to_one_hot(x_cat_hat).to(x_num_hat.dtype) if has_cat else x_cat_hat
        denoised, raw_logits = self._denoise_fn(
            x_num_hat.float(), x_cat_hat_oh,
            t_hat.squeeze().repeat(b), sigma=sigma_num_hat.unsqueeze(0).repeat(b,1)  # sigma accepts (bs, K_num)
        )
        
        # Apply cfg updates, if is in cfg mode
        is_bin_class = len(self.num_mask_idx) == 0
        is_learnable = self.scheduler=="power_mean_per_column"
        if cfg:
            if not is_learnable:
                sigma_cond = sigma_num_hat
            else:
                if is_bin_class:
                    sigma_cond = (0.002 ** (1/7) + t_hat * (80 ** (1/7) - 0.002 ** (1/7))).pow(7)
                else:
                    sigma_cond = sigma_num_hat[self.num_mask_idx]
            y_num_hat = x_num_hat.float()[:, self.num_mask_idx]
            idx = list(chain(*[self.slices_for_classes_with_mask[i] for i in self.cat_mask_idx]))
            y_cat_hat = x_cat_hat_oh[:,idx]
            y_only_denoised, y_only_raw_logits = self.y_only_model(
                y_num_hat, 
                y_cat_hat,
                t_hat.squeeze().repeat(b), sigma=sigma_cond.unsqueeze(0).repeat(b,1)  # sigma accepts (bs, K_num)
            )
            
            denoised[:, self.num_mask_idx] *= 1 + self.w_num
            denoised[:, self.num_mask_idx] -= self.w_num*y_only_denoised
            
            mask_logit_idx = [self.slices_for_classes_with_mask[i] for i in self.cat_mask_idx]
            mask_logit_idx = np.concatenate(mask_logit_idx) if len(mask_logit_idx)>0 else np.array([])
            
            raw_logits[:, mask_logit_idx] *= 1 + self.w_cat
            raw_logits[:, mask_logit_idx] -= self.w_cat*y_only_raw_logits
        
        # Causal Guidance (CFG-style): mask parent features and extrapolate
        if self.causal_guidance_scale > 0 and (self.cg_num_mask is not None or self.cg_cat_mask is not None):
            x_num_hat_uncond = x_num_hat.clone()
            x_cat_hat_oh_uncond = x_cat_hat_oh.clone() if has_cat else x_cat_hat_oh
            
            if self.cg_num_mask is not None:
                x_num_hat_uncond[:, self.cg_num_mask] = 0.0
            
            if self.cg_cat_mask is not None and has_cat:
                cg_cat_slices = list(chain(*[self.slices_for_classes_with_mask[i] for i in self.cg_cat_mask]))
                x_cat_hat_oh_uncond[:, cg_cat_slices] = 0.0
            
            denoised_uncond, raw_logits_uncond = self._denoise_fn(
                x_num_hat_uncond.float(), x_cat_hat_oh_uncond,
                t_hat.squeeze().repeat(b), sigma=sigma_num_hat.unsqueeze(0).repeat(b,1)
            )
            
            denoised = denoised + self.causal_guidance_scale * (denoised - denoised_uncond)
            raw_logits = raw_logits + self.causal_guidance_scale * (raw_logits - raw_logits_uncond)
        
        # Inference-time Macro Guidance: align group-wise injury mean
        if self.macro_guidance_scale > 0 and i <= self.macro_guidance_start_step and self.macro_guidance_group_means and self.macro_injury_idx is not None and self.macro_group_indices:
            injury_pred = denoised[:, self.macro_injury_idx]
            # Build group keys from current cat predictions (use x_cat_hat as the best estimate)
            group_keys = []
            for batch_idx in range(b):
                key_parts = []
                for gidx in self.macro_group_indices:
                    if gidx < x_cat_hat.shape[1]:
                        key_parts.append(str(int(x_cat_hat[batch_idx, gidx].item())))
                    else:
                        key_parts.append("0")
                group_keys.append("|".join(key_parts))
            
            # Compute current group means and apply guidance offset
            # IMPORTANT: target_mean is in raw space, but denoised is in standardized space
            g_mean = getattr(self, 'macro_guidance_global_mean', 0.0)
            g_std = getattr(self, 'macro_guidance_global_std', 1.0)
            if g_std < 1e-12:
                g_std = 1.0
            unique_keys = list(set(group_keys))
            for gkey in unique_keys:
                mask = [gk == gkey for gk in group_keys]
                mask_tensor = torch.tensor(mask, device=denoised.device)
                if mask_tensor.sum() == 0:
                    continue
                current_mean_std = injury_pred[mask_tensor].mean()
                target_mean_raw = self.macro_guidance_group_means.get(gkey, current_mean_std.item() * g_std + g_mean)
                # Convert target to standardized space for fair comparison
                target_mean_std = (target_mean_raw - g_mean) / g_std
                
                # --- Robust Guidance: relative offset + adaptive scaling ---
                raw_diff = target_mean_std - current_mean_std.item()
                
                if self.macro_guidance_mode == "relative":
                    # Relative offset: scale by current magnitude to avoid over-correction
                    denom = abs(current_mean_std.item()) + 1.0
                    compressed_diff = math.tanh(raw_diff / denom)
                    offset = compressed_diff * self.macro_guidance_scale
                elif self.macro_guidance_mode == "adaptive":
                    # Adaptive: reduce scale when drift is large
                    drift_indicator = abs(raw_diff) / (abs(current_mean_std.item()) + 1e-8)
                    adaptive_scale = self.macro_guidance_scale * math.exp(-drift_indicator / self.macro_guidance_adaptive_drift_threshold)
                    offset = raw_diff * adaptive_scale
                elif self.macro_guidance_mode == "annealed":
                    # Annealed: reduce scale as denoising progresses (i from T down to 0)
                    # i is the current step index (smaller = closer to clean data)
                    # Use a temperature that increases as we get closer to clean data
                    temperature = max(0.0, min(1.0, i / max(1, self.macro_guidance_start_step)))
                    annealed_scale = self.macro_guidance_scale * temperature
                    offset = raw_diff * annealed_scale
                else:
                    # Default absolute offset (original behavior)
                    offset = raw_diff * self.macro_guidance_scale
                
                denoised[mask_tensor, self.macro_injury_idx] += offset
        
        # Euler step
        d_cur = (x_num_hat - denoised) / sigma_num_hat
        x_num_next = x_num_hat + (sigma_num_next - sigma_num_hat) * d_cur
        
        # Unmasking
        x_cat_next = x_cat_cur
        q_xs = torch.zeros_like(x_cat_cur).float()
        if has_cat:
            logits = self._subs_parameterization(raw_logits, x_cat_hat)
            alpha_t = torch.exp(-sigma_cat_hat).unsqueeze(0).repeat(b,1)
            alpha_s = torch.exp(-sigma_cat_next).unsqueeze(0).repeat(b,1)
            x_cat_next, q_xs = self._mdlm_update(logits, x_cat_hat, alpha_t, alpha_s)
        
        # Apply 2nd order correction.
        if self.sampler_params['second_order_correction']:
            if i > 0:
                x_cat_hat_oh = self.to_one_hot(x_cat_hat).to(x_num_next.dtype) if has_cat else x_cat_hat
                denoised, raw_logits = self._denoise_fn(
                    x_num_next.float(), x_cat_hat_oh,
                    t_next.squeeze().repeat(b), sigma=sigma_num_next.unsqueeze(0).repeat(b,1)
                )
                if cfg:
                    if not is_learnable:
                        sigma_cond = sigma_num_next
                    else:
                        if is_bin_class:
                            sigma_cond = (0.002 ** (1/7) + t_next * (80 ** (1/7) - 0.002 ** (1/7))).pow(7)
                        else:
                            sigma_cond = sigma_num_next[self.num_mask_idx]
                    y_num_next = x_num_next.float()[:, self.num_mask_idx]
                    idx = list(chain(*[self.slices_for_classes_with_mask[i] for i in self.cat_mask_idx]))
                    y_cat_hat = x_cat_hat_oh[:, idx]
                    y_only_denoised, y_only_raw_logits = self.y_only_model(
                        y_num_next,
                        y_cat_hat,
                        t_next.squeeze().repeat(b), sigma=sigma_cond.unsqueeze(0).repeat(b,1)  # sigma accepts (bs, K_num)
                    )
                    denoised[:, self.num_mask_idx] *= 1 + self.w_num
                    denoised[:, self.num_mask_idx] -= self.w_num*y_only_denoised
                
                # Causal Guidance (2nd order correction)
                if self.causal_guidance_scale > 0 and (self.cg_num_mask is not None or self.cg_cat_mask is not None):
                    x_num_next_uncond = x_num_next.clone()
                    x_cat_hat_oh_uncond = x_cat_hat_oh.clone() if has_cat else x_cat_hat_oh
                    
                    if self.cg_num_mask is not None:
                        x_num_next_uncond[:, self.cg_num_mask] = 0.0
                    
                    if self.cg_cat_mask is not None and has_cat:
                        cg_cat_slices = list(chain(*[self.slices_for_classes_with_mask[i] for i in self.cg_cat_mask]))
                        x_cat_hat_oh_uncond[:, cg_cat_slices] = 0.0
                    
                    denoised_uncond2, raw_logits_uncond2 = self._denoise_fn(
                        x_num_next_uncond.float(), x_cat_hat_oh_uncond,
                        t_next.squeeze().repeat(b), sigma=sigma_num_next.unsqueeze(0).repeat(b,1)
                    )
                    
                    denoised = denoised + self.causal_guidance_scale * (denoised - denoised_uncond2)
                    raw_logits = raw_logits + self.causal_guidance_scale * (raw_logits - raw_logits_uncond2)
                
                d_prime = (x_num_next - denoised) / sigma_num_next
                x_num_next = x_num_hat + (sigma_num_next - sigma_num_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        return x_num_next, x_cat_next, q_xs


    def sample_impute(self, x_num, x_cat, num_mask_idx, cat_mask_idx, resample_rounds, impute_condition, w_num, w_cat):
        self.w_num = w_num
        self.w_cat = w_cat
        self.num_mask_idx = num_mask_idx
        self.cat_mask_idx = cat_mask_idx
        
        b = x_num.size(0)
        device = self.device
        dtype = torch.float32

        # Create masks, true for the missing columns
        num_mask = [i in num_mask_idx for i in range(self.num_numerical_features)]
        cat_mask = [i in cat_mask_idx for i in range(len(self.num_classes))]
        num_mask = torch.tensor(num_mask).to(x_num.device).to(x_num.dtype)
        cat_mask = torch.tensor(cat_mask).to(x_cat.device).to(x_cat.dtype)

        # Create the chain of t
        t = torch.linspace(0,1,self.num_timesteps, dtype=dtype, device=device)      # times = 0.0,...,1.0
        t = t[:, None]
        
        # Compute the chains of sigma
        sigma_num_cur = self.num_schedule.total_noise(t)
        sigma_cat_cur = self.cat_schedule.total_noise(t)
        sigma_num_next = torch.zeros_like(sigma_num_cur)
        sigma_num_next[1:] = sigma_num_cur[0:-1]
        sigma_cat_next = torch.zeros_like(sigma_cat_cur)
        sigma_cat_next[1:] = sigma_cat_cur[0:-1]
        
        # Prepare sigma_hat for stochastic sampling mode
        if self.sampler_params['stochastic_sampler']:
            gamma = min(S_churn / self.num_timesteps, np.sqrt(2) - 1) * (S_min <= sigma_num_cur) * (sigma_num_cur <= S_max)
            sigma_num_hat = sigma_num_cur + gamma * sigma_num_cur
            t_hat = self.num_schedule.inverse_to_t(sigma_num_hat)
            t_hat = torch.min(t_hat, dim=-1, keepdim=True).values    # take the samllest t_hat induced by sigma_num
            zero_gamma = (gamma==0).any()
            t_hat[zero_gamma] = t[zero_gamma]
            out_of_bound = (t_hat > 1).squeeze()
            sigma_num_hat[out_of_bound] = sigma_num_cur[out_of_bound]
            t_hat[out_of_bound] = t[out_of_bound]
            sigma_cat_hat = self.cat_schedule.total_noise(t_hat)
        else:
            t_hat = t
            sigma_num_hat = sigma_num_cur
            sigma_cat_hat = sigma_cat_cur

        # Sample priors for the continuous dimensions
        if impute_condition == "x_t":
            z_norm = x_num + torch.randn((b, self.num_numerical_features), device=device) * sigma_num_cur[-1]   # z_{t_max} = x_0(masked) + sigma_max*epsilon
        elif impute_condition == "x_0":
            z_norm = x_num
            
        # Sample priors for the discrete dimensions
        has_cat = len(self.num_classes) > 0
        z_cat = torch.zeros((b, 0), device=device).float()      # the default values for categorical sample if the dataset has no categorical entry
        if has_cat:
            if impute_condition == "x_t":
                z_cat = self._sample_masked_prior(
                    b,
                    len(self.num_classes),
                )   # z_{t_max} is still all pushed to [MASK]
            elif impute_condition == "x_0":
                z_cat = x_cat
        
        pbar = tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps)
        pbar.set_description(f"Sampling Progress")
        for i in pbar:
            for u in range (resample_rounds):
                # Get known parts by Forward Flow
                if impute_condition == "x_t":
                    z_norm_known = x_num + torch.randn((b, self.num_numerical_features), device=device) * sigma_num_next[i]
                    move_chance = 1 - torch.exp(-sigma_cat_next[i]) if i < (self.num_timesteps-1) else torch.ones_like(sigma_cat_next[i])     # force move_chance to be 1 for the first iteration
                    z_cat_known, _ = self.q_xt(x_cat, move_chance)
                elif impute_condition == "x_0":
                    z_norm_known = x_num
                    z_cat_known = x_cat
                
                # Get unknown by Reverse Step
                z_norm_unknown, z_cat_unknown, q_xs = self.edm_update(
                    z_norm, z_cat, i, 
                    t[i], t[i-1] if i > 0 else None, t_hat[i],
                    sigma_num_cur[i], sigma_num_next[i], sigma_num_hat[i], 
                    sigma_cat_cur[i], sigma_cat_next[i], sigma_cat_hat[i],
                )
                z_norm = (1 - num_mask)  * z_norm_known + num_mask * z_norm_unknown
                z_cat = (1 - cat_mask) * z_cat_known + cat_mask * z_cat_unknown

                # Resample x_t from x_{t-1} by Foward Step
                if u < resample_rounds-1:
                    z_norm = z_norm + (sigma_num_cur[i] ** 2 - sigma_num_next[i] ** 2).sqrt() * S_noise * torch.randn_like(z_norm)
                    move_chance = -torch.expm1(sigma_cat_next[i] - sigma_cat_cur[i])
                    z_cat, _ = self.q_xt(z_cat, move_chance)
        
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample
    