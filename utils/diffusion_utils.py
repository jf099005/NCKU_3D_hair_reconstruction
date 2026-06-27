import torch
import k_diffusion as K
import torch.nn.functional as F
from tqdm import tqdm

@torch.no_grad()
def sample_images(nr_images, model_ema, model_config, nr_iters=100, extra_args={}, callback=None):
    model_ema.eval()
    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']
    size = model_config['input_size']
    n_per_proc = nr_images
    x = torch.randn([1, n_per_proc, model_config['input_channels'], size[0], size[1]]).cuda()
    x = x[0] * sigma_max
    model_fn = model_ema
    sigmas = K.sampling.get_sigmas_karras(nr_iters, sigma_min, sigma_max, rho=7., device="cuda")
    x_0 = K.sampling.sample_dpmpp_2m_sde(model_fn, x, sigmas, extra_args=extra_args, eta=0.0, solver_type='heun', disable=False, callback=callback)
    return x_0

@torch.no_grad()
#samples using classifier free guidance and only enables the cfg_val when the sigma is within the interval.
#idead from this paper: https://arxiv.org/pdf/2404.07724
def sample_images_cfg(nr_images, cfg_val, cfg_interval, model_ema, model_config, nr_iters=100, extra_args={}, callback=None):
    model_ema.eval()
    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']
    size = model_config['input_size']
    n_per_proc = nr_images
    x = torch.randn([1, n_per_proc, model_config['input_channels'], size[0], size[1]]).cuda()
    x = x[0] * sigma_max
    model_fn = model_ema
    sigmas = K.sampling.get_sigmas_karras(nr_iters, sigma_min, sigma_max, rho=7., device="cuda")
    x_0 = K.sampling.sample_dpmpp_2m_sde_cfg(model_fn, x, sigmas, cfg_val, cfg_interval, extra_args=extra_args,  eta=0.0, solver_type='heun', disable=False, callback=callback)
    return x_0


# ============================================================================================================================
@torch.no_grad()
def multi_diffusion(model_ema, model_config, multi_extra_args, masks, nr_iters=50):
    """
    multi_extra_args: 列表，包含多組圖像的 extra_args (例如 [img1_args, img2_args])
    masks: 列表，包含對應的權重遮罩 (例如 [mask1, mask2])，形狀需為 (1, 1, H, W)
    """
    model_ema.eval()
    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']
    size = model_config['input_size']
    device = "cuda"

    # 1. 初始化單一潛在空間 (Latent Space)
    # 所有區域共享同一個噪點圖，這保證了邊界的初始連續性
    x = torch.randn([1, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max
    
    # 2. 設定步數 (Sigmas)
    sigmas = K.sampling.get_sigmas_karras(nr_iters, sigma_min, sigma_max, rho=7., device=device)
    pbar = tqdm(range(len(sigmas) - 1))

    # 3. 自定義去噪循環 (使用簡單的 Euler 採樣為例)
    for i in pbar:
            sigma = sigmas[i]
            next_sigma = sigmas[i+1]
            
            combined_noise_pred = torch.zeros_like(x)
            total_weight = torch.zeros_like(x[:, :1, :, :])

            for args, mask in zip(multi_extra_args, masks):
                noise_pred = model_ema(x, sigma.expand(x.shape[0]), **args)
                combined_noise_pred += noise_pred * mask
                total_weight += mask

            combined_noise_pred /= (total_weight + 1e-6)
            
            # 3. (選配) 在進度條右側動態顯示資訊，例如目前的 sigma
            pbar.set_postfix({"sigma": f"{sigma.item():.2f}"})

            d = (x - combined_noise_pred) / sigma
            dt = next_sigma - sigma
            x = x + d * dt
        
    return x


@torch.no_grad()
def multi_diffusion_cfg(model_ema, model_config, multi_extra_args, masks, cfg_val=1.0, cfg_interval=[0.0, 5.0], nr_iters=50):
    model_ema.eval()
    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']
    size = model_config['input_size']
    device = "cuda"

    x = torch.randn([1, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max
    sigmas = K.sampling.get_sigmas_karras(nr_iters, sigma_min, sigma_max, rho=7., device=device)
    pbar = tqdm(range(len(sigmas) - 1))

    for i in pbar:
        sigma = sigmas[i]
        next_sigma = sigmas[i+1]
        
        # 判斷當前 sigma 是否在 CFG 作用區間內
        is_cfg_step = cfg_interval[0] <= sigma <= cfg_interval[1]
        
        combined_noise_pred = torch.zeros_like(x)
        total_weight = torch.zeros_like(x[:, :1, :, :])

        for args, mask in zip(multi_extra_args, masks):
            # 1. 計算「有條件」的噪點預測 (Conditional)
            cond_noise = model_ema(x, sigma.expand(x.shape[0]), **args)
            
            if is_cfg_step:
                # 2. 準備「無條件」參數 (通常是空字典或特定 dropout)
                # 根據 DiffLocks 邏輯，uncond 可能是將 latents 設為 None 或 0
                uncond_args = args.copy()
                uncond_args['latents_dict'] = None # 或者根據模型實作傳入對應空值
                
                # 3. 計算「無條件」噪點預測 (Unconditional)
                uncond_noise = model_ema(x, sigma.expand(x.shape[0]), **uncond_args)
                
                # 4. 執行 CFG 公式：uncond + cfg_val * (cond - uncond)
                noise_step = uncond_noise + cfg_val * (cond_noise - uncond_noise)
            else:
                noise_step = cond_noise

            # 根據遮罩進行權重融合
            combined_noise_pred += noise_step * mask
            total_weight += mask

        combined_noise_pred /= (total_weight + 1e-6)
        
        pbar.set_postfix({"sigma": f"{sigma.item():.2f}", "CFG": is_cfg_step})

        # Euler 步進
        d = (x - combined_noise_pred) / sigma
        dt = next_sigma - sigma
        x = x + d * dt
        
    return x


@torch.no_grad()
def multi_diffusion_cfg_v2(model_ema, model_config, multi_extra_args, masks, cfg_val=1.0, cfg_interval=[0.0, 5.0], nr_iters=50):
    model_ema.eval()
    sigma_min, sigma_max = model_config['sigma_min'], model_config['sigma_max']
    size = model_config['input_size']
    device = "cuda"

    # 1. 初始化共享潛在空間
    x = torch.randn([1, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max
    sigmas = K.sampling.get_sigmas_karras(nr_iters, sigma_min, sigma_max, rho=7., device=device)
    pbar = tqdm(range(len(sigmas) - 1), desc="MultiDiffusion-CFG-V2")

    for i in pbar:
        sigma = sigmas[i]
        next_sigma = sigmas[i+1]
        is_cfg_step = cfg_interval[0] <= sigma <= cfg_interval[1]
        
        combined_noise_pred = torch.zeros_like(x)
        total_weight = torch.zeros_like(x[:, :1, :, :])

        # --- 2. 核心改進：計算全局無條件引導 (Shared Uncond) ---
        # 這樣做能讓所有區域共享同一個底色，減少破碎感
        global_uncond_noise = None
        if is_cfg_step:
            uncond_args = multi_extra_args[0].copy()
            uncond_args['latents_dict'] = None # 觸發模型內部的 dropout 邏輯
            global_uncond_noise = model_ema(x, sigma.expand(x.shape[0]), **uncond_args)

        # 3. 區域化採樣
        for args, mask in zip(multi_extra_args, masks):
            cond_noise = model_ema(x, sigma.expand(x.shape[0]), **args)
            
            if is_cfg_step and global_uncond_noise is not None:
                # 使用全域基準進行 CFG 放大
                noise_step = global_uncond_noise + cfg_val * (cond_noise - global_uncond_noise)
            else:
                noise_step = cond_noise

            combined_noise_pred += noise_step * mask
            total_weight += mask

        combined_noise_pred /= (total_weight + 1e-6)
        
        # 4. 執行 Euler 步進
        d = (x - combined_noise_pred) / sigma
        dt = next_sigma - sigma
        x = x + d * dt
        
    return x