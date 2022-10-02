"""Noise-conditional iterative refinement networks."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from .helpers import (
    cosine_beta_schedule,
    extract
)
from tqdm import tqdm
import random
    
class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, repr_dim, skeleton, n_timesteps=1000, clip_denoised=False, predict_epsilon=True, unc_token=None, guidance_weight=3
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = repr_dim
        self.model = model
        self.UNC = unc_token
        
        self.self_condition = False

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.skeleton = skeleton

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        self.guidance_weight = guidance_weight

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))


    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            if len(x_t.shape) == 4:
                x_t = x_t[:,0]
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, y, t, self_cond=None, cond=None):
        input_dict = {"x": x.cpu().numpy(), "y": y.cpu().numpy(), "t": t.cpu().numpy(), "self_cond": self_cond, "cond": cond}
        noise = torch.Tensor(self.model.run(["out"], input_dict)[0]).cuda()
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)
        
        # after the halfway mark is unconditional. Blend.
        half = len(x_recon) // 2
        conditioned, unconditioned = x_recon[:half], x_recon[half:]

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()
            
        x_recon = unconditioned + self.guidance_weight * (conditioned - unconditioned)
        x_t = x[:half].squeeze()
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x_t, t=t[:half])
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, y, t, self_cond=None, cond=None):
        b, *_, device = *x.shape, x.device
        doubled_x = torch.cat((x,x), dim=0)
        doubled_t = torch.cat((t,t), dim=0)
        doubled_self_cond = torch.cat((self_cond, self_cond), dim=0) if self_cond is not None else None
        doubled_cond = torch.cat((cond, cond), dim=0) if cond is not None else None
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=doubled_x, y=y, t=doubled_t, self_cond=doubled_self_cond, cond=doubled_cond)
        noise = torch.randn_like(model_mean)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(noise.shape) - 1)))
        x_out = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        x_out, x_start = x_out.unsqueeze(1), x_start.unsqueeze(1)
        return x_out, x_start
        
    @torch.no_grad()
    def p_sample_loop(self, 
                      shape, 
                      y,
                      noise=None, 
                      return_diffusion=False, 
                      start_point=None
                     ):
        device = self.betas.device
        
        # default to diffusion over whole timescale
        start_point = self.n_timesteps if start_point is None else start_point
        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        y = y.to(device)
        
        # double y along the batch dimension for blending
        # set y to UNC after the halfway mark
        y_unc = torch.full_like(y, self.UNC)
        y = torch.cat((y,y_unc), dim=0)

        if return_diffusion: diffusion = [x]
            
        x_start = None
        
        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            # resample multiple times
            self_cond = x_start if self.self_condition else None
            # sample x from step i to step i-1
            x, x_start = self.p_sample(x, y, timesteps, self_cond=self_cond)

            if return_diffusion: diffusion.append(x)
        
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, shape, y, *args, horizon=None, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        horizon = horizon or self.horizon

        return self.p_sample_loop(shape, y, *args, **kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def forward(self, shape, y, *args, **kwargs):
        return self.conditional_sample(shape, y, *args, **kwargs)
    
    def partial_denoise(self, x, y, t):
        x_noisy = self.noise_to_t(x, t)
        return self.p_sample_loop(x.shape, y, noise=x_noisy, start_point=t)
    
    def noise_to_t(self, x, timestep):
        batch_size = len(x)
        t = torch.full((batch_size,), timestep, device=x.device).long()
        return self.q_sample(x, t)