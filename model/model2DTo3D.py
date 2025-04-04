import inspect
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from tqdm import tqdm

from .model_utils import get_custom_betas
from .pvn_model import PVNModel
from .VesselDiffusion_model import VesselDiffusionModel


class Two2ThreeDiffusionModel(VesselDiffusionModel):
    
    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
        point_cloud_model_embed_dim: int,
        **kwargs,  # projection arguments
    ):
        super().__init__(**kwargs)

        # Create diffusion model schedulers which define the sampling timesteps
        scheduler_kwargs = {}
        if beta_schedule == 'custom':
            scheduler_kwargs.update(dict(trained_betas=get_custom_betas(beta_start=beta_start, beta_end=beta_end)))
        else:
            scheduler_kwargs.update(dict(beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule))
        self.schedulers_map = {
            'ddpm': DDPMScheduler(**scheduler_kwargs, clip_sample=False),
            'ddim': DDIMScheduler(**scheduler_kwargs, clip_sample=False), 
            'pndm': PNDMScheduler(**scheduler_kwargs), 
        }
        self.scheduler = self.schedulers_map['ddpm']  # this can be changed for inference

        # Create point cloud model for processing point cloud at each diffusion step
        self.point_cloud_model = PVNModel(
            embed_dim=point_cloud_model_embed_dim,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
        )

    def compute_loss(self, pc, mip_img, mean, std, hf_size, adj, geo):
        x_0 = pc
        B, N, D = x_0.shape
        # Sample random noise
        noise = torch.randn_like(x_0)

        timestep = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=self.device, dtype=torch.long)
        x_t = self.scheduler.add_noise(x_0, noise, timestep)
        x_t_input = self.extend_feature(x_t, mip_img, mean, std, hf_size, t=timestep, adj=adj, geo=geo)
        noise_pred = self.point_cloud_model(x_t_input, timestep)
        if not noise_pred.shape == noise.shape:
            raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')
        loss = F.mse_loss(noise_pred, noise)

        return loss
    
    @torch.no_grad()
    def generate(self, num_points, mip_img, mean, std, hf_size, adj, geo,
        scheduler: Optional[str] = 'ddpm',
        num_inference_steps: Optional[int] = 1000,
        eta: Optional[float] = 0.0, 
        return_sample_every_n_steps: int = -1,
        disable_tqdm: bool = False,
    ):

        # Get scheduler from mapping, or use self.scheduler if None
        scheduler = self.scheduler if scheduler is None else self.schedulers_map[scheduler]

        # Get the size of the noise
        N = num_points
        B = 1 if mip_img is None else mip_img.shape[0]
        D = 3
        device = self.device if mip_img is None else mip_img.device
        
        # Sample noise
        x_t = torch.randn(B, N, D, device=device)

        # Set timesteps
        accepts_offset = "offset" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {"offset": 1} if accepts_offset else {}
        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {"eta": eta} if accepts_eta else {}

        # Loop over timesteps
        all_outputs = [x_t]
        return_all_outputs = (return_sample_every_n_steps > 0)
        progress_bar = tqdm(scheduler.timesteps.to(device), desc=f'Sampling ({x_t.shape})', disable=disable_tqdm)
        for i, t in enumerate(progress_bar):
            x_t_input = self.extend_feature(x_t, mip_img, mean, std, hf_size, t=t, adj=adj, geo=geo)
            
            # Forward
            noise_pred = self.point_cloud_model(x_t_input, t.reshape(1).expand(B))

            # Step
            x_t = scheduler.step(noise_pred, t, x_t, **extra_step_kwargs).prev_sample

            # Append to output list if desired
            if (return_all_outputs and (i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1)):
                all_outputs.append(x_t)
        output = x_t
        if return_all_outputs:
            all_outputs = torch.stack(all_outputs, dim=1)  # (B, sample_steps, N, D)
        
        return (output, all_outputs) if return_all_outputs else output
    
    def forward(self, pc, mip_img, mean, std, hf_size, mode: str = 'train', adj=None, geo=None, **kwargs):
        if mode == 'train':
            return self.compute_loss(pc, mip_img, mean, std, hf_size, adj, geo) 
        elif mode == 'sample':
            return self.generate(4096, mip_img, mean, std, hf_size, adj, geo, **kwargs) 
        else:
            raise NotImplementedError()
