import torch
import argparse
from flags import set_flags

from functools import partial
from loss import huber_loss

from ultralytics import YOLO
import json
import numpy as np

import time
import os
import glob
import tqdm
import math
import imageio
import random
import warnings
import tensorboardX
import h5py
import json
import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver
import lpips
from torchmetrics.functional import structural_similarity_index_measure
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO
import math
import time
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from PIL import Image
import numpy as np
import os
import torch
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

import raymarching
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lpips_metric = lpips.LPIPS(net='alex').to(device)


if __name__ == '__main__':
    parser= set_flags()
    
    opt = parser.parse_args()

    
    """"
    Object-Wise Scene Reconstruction
    """
    
    from nerf.provider import NeRFDataset
    from nerf.gui import NeRFGUI
    from nerf.utils import *
    from nerf.network_tcnn import NeRFNetwork
    


    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        # density_thresh=opt.density_thresh,
        density_thresh = 10,
        bg_radius=opt.bg_radius,
        n_chunks=opt.n_chunks,
        bound_inf=opt.bound_inf,
    )
    
    print(model)



    class MultiTrainer(Trainer):
        def __init__(self, ngp, opt, model, device, workspace, optimizer, criterion, ema_decay, fp16,lr_scheduler,scheduler_update_every_step, metrics, use_checkpoint, eval_interval,ckpt_list):
            super().__init__(ngp, opt, model, device=device, workspace=workspace, optimizer=optimizer, criterion=criterion, ema_decay=ema_decay, fp16=fp16,lr_scheduler=lr_scheduler, scheduler_update_every_step=scheduler_update_every_step, metrics=metrics, use_checkpoint=use_checkpoint, eval_interval=eval_interval)
            self.ckpt_list = ckpt_list
            

        def batch_image_depth_generation(self, data, densities, rgbs, background_color, max_batch_size=4096):
            pred_rgb = torch.empty((self.B, self.H * self.W, 4), device=device)
            preds_depth = torch.empty((self.B, self.H * self.W), device=device)
            

            data['rays_o'] = data['rays_o']
            data['rays_d'] = data['rays_d']

            
        
            for b in range(self.B):
                head = 0
                while head < self.N:
                  tail = min(head + max_batch_size, self.N)

                  results_ = self.image_depth_generation(
                      {
                          'rays_o': data['rays_o'][b:b+1, head:tail],
                          'rays_d': data['rays_d'][b:b+1, head:tail],
                      },
                      densities[:, head:tail, :],
                      rgbs[:, head:tail, ...], background_color
                  )
                  pred_rgb[b:b+1, head:tail], preds_depth[b:b+1, head:tail] = results_
                  head += max_batch_size
            
            outputs_image = pred_rgb
            outputs_depth = preds_depth
            return outputs_image, outputs_depth
        
        def image_depth_generation(self, data, densities, rgbs,bg_color):
            prefix = data['rays_o'].shape[:-1]
            rays_o = data['rays_o'].contiguous().view(-1, 3)  # masked [num_rays,3]  [177,3]
            rays_d = data['rays_d'].contiguous().view(-1, 3)
            N = data['rays_o'].shape[0]
            device = rays_o.device
            aabb = self.model.aabb_train if self.model.training else self.model.aabb_infer
            nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.model.min_near)
            nears.unsqueeze_(-1)
            fars.unsqueeze_(-1)
            z_vals = torch.linspace(0.0, 1.0, opt.num_steps, device=device).unsqueeze(0) # [1, T]
            z_vals = z_vals.expand((N, opt.num_steps)) # [N, T]
            z_vals = nears + (fars - nears) * z_vals
            sample_dist = (fars - nears) / opt.num_steps # [177,1]
            perturb=False
            density_scale=1
            # bg_color=1
            if perturb:
                z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
            deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
            alphas = 1 - torch.exp(-deltas * density_scale * densities.squeeze(-1)) # [N, T+t]
            alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+t+1]
            weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]
            
            rgbs= rgbs.squeeze(0)
            weights = weights.squeeze(0)
            
            weights_sum = weights.sum(dim=-1)
            densities= densities.squeeze(0). unsqueeze(-1)
            
            ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)   # [num_rays, num_bins] 177,512]
            depth = torch.sum(weights * ori_z_vals, dim=-1)     # 177
            image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)
            alpha_channel = torch.sum(weights.unsqueeze(-1) * densities, dim=-2)
           
            image = torch.cat((image, alpha_channel), dim=-1) 
            
            
        
            if bg_color == 'white':
                bg_color = torch.ones(N, 1).to(device)
            elif bg_color == 'black':
                bg_color = torch.zeros(N, 1).to(device)
            

            

            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
  

            image = image.clamp(0, 1)
            

         

            pred_rgb = image
            pred_depth = depth
            
            return pred_rgb, pred_depth
        

        def batch_run (self, rays_o, rays_d, yolo_details, staged=True, max_ray_batch= 4096, **kwargs):
            self.cuda_ray=False
            

            if self.cuda_ray:
                
                self.run_cuda
            else:
                self.run
          
            if staged:
                densities = torch.empty((self.B, self.N, opt.num_steps), device=device)
                rgbs = torch.empty((self.B, self.N, opt.num_steps, 3), device=device)
                
            
                for b in range(self.B):
                    head = 0
                    while head < self.N:
                      tail = min(head + max_ray_batch, self.N)
                      rays_o = rays_o.reshape(1, self.H * self.W, 3)
                      rays_d = rays_d.reshape(1, self.H * self.W, 3)
                      
                      results_ = self.run(model, rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], yolo_details, **kwargs)
                      densities[b:b+1, head:tail] = results_['densities'].permute(2,0,1)
                      rgbs[b:b+1, head:tail] = results_['rgbs']
                      
                      

                      head += max_ray_batch
                
                results = {}
                results['densities'] = densities
                results['rgbs'] = rgbs
                
            else:
                results = self.run(rays_o, rays_d, yolo_details, **kwargs)
                
            return results

        



        
        def best_densities_and_colors_v3(self, densities, max_densities, rgbs, best_rgbs):
            new_max_densities = torch.maximum(densities, max_densities)
            # breakpoint()
            new_best_rgbs = torch.where(densities[..., None] > max_densities[..., None], rgbs, best_rgbs)
            return new_max_densities, new_best_rgbs
        
        

        def compute_lpips(self,img1, img2):
            loss_fn = lpips.LPIPS(net='alex')
            
            if img1.shape[-1] == 4:  # RGBA
                img1 = img1[..., :3]
            if img2.shape[-1] == 4:  # RGBA
                img2 = img2[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            return loss_fn(img1, img2).item()


        def calculate_psnr(self,img1, img2):
            
            if not isinstance(img1, torch.Tensor):
                img1 = torch.tensor(img1)
            if not isinstance(img2, torch.Tensor):
                img2 = torch.tensor(img2)
            
           
            img1 = img1.float()
            img2 = img2.float()
            
            
            assert img1.shape == img2.shape, "Input images must have the same shape"
            
           
            if img1.dim() == 3:
                img1 = img1.unsqueeze(0)
                img2 = img2.unsqueeze(0)
            
            mse = F.mse_loss(img1, img2)
            if mse == 0:
                return float('inf')
            max_pixel_value = 255.0 if img1.max() > 1.0 else 1.0
            psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
            return psnr.item()

        

        def ssim_rgba(self,img1, img2, window_size=11, sigma=1.5, k1=0.01, k2=0.03, L=255):
            
            img1 = img1.astype(np.float64) / 255.0
            img2 = img2.astype(np.float64) / 255.0

           
            img1_rgb = img1[..., :3]
            img2_rgb = img2[..., :3]
            alpha1 = img1[..., 3]
            alpha2 = img2[..., 3]

            
            ssim_rgb = np.mean([self.ssim_channel(img1_rgb[..., i], img2_rgb[..., i], window_size, sigma, k1, k2, 1.0) for i in range(3)])

            
            ssim_alpha = self.ssim_channel(alpha1, alpha2, window_size, sigma, k1, k2, 1.0)

           
            return 0.8 * ssim_rgb + 0.2 * ssim_alpha

        def ssim_channel(self,ch1, ch2, window_size, sigma, k1, k2, L):
            
            mu1 = gaussian_filter(ch1, sigma)
            mu2 = gaussian_filter(ch2, sigma)

           
            sigma1_sq = gaussian_filter(ch1**2, sigma) - mu1**2
            sigma2_sq = gaussian_filter(ch2**2, sigma) - mu2**2
            sigma12 = gaussian_filter(ch1 * ch2, sigma) - mu1 * mu2

            
            C1 = (k1 * L)**2
            C2 = (k2 * L)**2
            numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
            denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
            ssim_map = numerator / denominator

            return np.mean(ssim_map)
        
        
        def process_images_with_background(self, trainer, data, max_densities, max_rgbs, H, W, background_color):
            
            pred_rgb, preds_depth = trainer.batch_image_depth_generation(data, max_densities, max_rgbs,background_color)
            img = pred_rgb.view(1, H, W, 4)[0].detach().cpu().numpy()
            depth = preds_depth.view(1, H, W)[0].detach().cpu().numpy()
            gt_uint8 = data['images'].cpu().numpy().squeeze(0)
            
           
            if background_color == 'white':
                background = np.ones((H, W, 4), dtype=np.float32)
                background[:, :] = [1.0, 1.0, 1.0, 1.0]
            elif background_color == 'black':
                background = np.zeros((H, W, 4), dtype=np.float32)
                background[:, :] = [0.0, 0.0, 0.0, 1.0]
            else:
                raise ValueError("background_color must be 'white' or 'black'")
            
            
            mask = img[:, :, 3] < 0.5
            mask1 = gt_uint8[:, :, 3] < 0.5
            
           
            img[mask] = background[mask]
            gt_uint8[mask1] = background[mask1]
            
            
            img[:, :, 3] = 1.0
            gt_uint8[:, :, 3] = 1.0


            img_uint8 = (img * 255).astype(np.uint8)
            gt_uint8 = (gt_uint8 * 255).astype(np.uint8)
            rgb_img = Image.fromarray(img_uint8, 'RGBA')
            depth_img = Image.fromarray((depth * 255).astype(np.uint8), 'L')
            gt_img = Image.fromarray(gt_uint8, 'RGBA')
            
            psnr= self.calculate_psnr(img_uint8, gt_uint8)
            

            lpips_value = self.compute_lpips(img_uint8, gt_uint8)
            # print(f"LPIPS: {lpips_value}")
            ssim_value = self.ssim_rgba(img_uint8, gt_uint8)
            
            return rgb_img, depth_img, gt_img, psnr, ssim_value, lpips_value
        def save_image(self,img, path):
            try:
                img.save(path)
            except Exception as e:
                print(f"Error saving image: {e}")
        
        def compute_metrics_both_backgrounds(self, trainer, data, max_densities, max_rgbs, H, W, opt, i):
           
            white_rgb_img, white_depth_img, white_gt_img, white_psnr , white_ssim, white_lpips= self.process_images_with_background(
                trainer, data, max_densities, max_rgbs, H, W, 'white'
            )
            
           
            black_rgb_img, black_depth_img, black_gt_img, black_psnr, black_ssim, black_lpips = self.process_images_with_background(
                trainer, data, max_densities, max_rgbs, H, W, 'black'
            )
            
           
            avg_psnr = (white_psnr + black_psnr) / 2
            avg_ssim = (white_ssim + black_ssim) / 2
            avg_lpips = (white_lpips + black_lpips) / 2



            rgb_path_w = os.path.join(opt.workspace, 'rgbs_w', f'{i+1}.png')
            depth_path_w = os.path.join(opt.workspace, 'depth_w', f'{i+1}.png')
            rgb_path_b = os.path.join(opt.workspace, 'rgbs_b', f'{i+1}.png')
            depth_path_b = os.path.join(opt.workspace, 'depth_b', f'{i+1}.png')
            gt_path_w = os.path.join(opt.workspace, 'ground_truth_w', f'{i+1}.png')
            gt_path_b = os.path.join(opt.workspace, 'ground_truth_b', f'{i+1}.png')

            os.makedirs(os.path.dirname(rgb_path_w), exist_ok=True)
            os.makedirs(os.path.dirname(rgb_path_b), exist_ok=True)
            os.makedirs(os.path.dirname(depth_path_w), exist_ok=True)
            os.makedirs(os.path.dirname(depth_path_b), exist_ok=True)
            os.makedirs(os.path.dirname(gt_path_w), exist_ok=True)
            os.makedirs(os.path.dirname(gt_path_b), exist_ok=True)
            
            
            self.save_image(white_rgb_img, rgb_path_w)
            self.save_image(white_depth_img, depth_path_w)
            self.save_image(black_rgb_img, rgb_path_b)
            self.save_image(white_depth_img, depth_path_b)
            self.save_image(white_gt_img, gt_path_w)
            self.save_image(black_gt_img, gt_path_b)
            
            return {
                'white': {
                    'rgb_img': white_rgb_img,
                    'depth_img': white_depth_img,
                    'gt_img': white_gt_img,
                    'psnr': white_psnr,
                    'ssim': white_ssim,
                    'lpips': white_lpips
                },
                'black': {
                    'rgb_img': black_rgb_img,
                    'depth_img': black_depth_img,
                    'gt_img': black_gt_img,
                    'psnr': black_psnr,
                    'ssim': black_ssim,
                    'lpips': black_lpips
                },
                'average_psnr': avg_psnr,
                'average_ssim': avg_ssim,
                'average_lpips': avg_lpips
            }

       

           

        def run(self, model, rays_o, rays_d, yolo_details, staged=True, max_ray_batch= 4096, **kwargs):
            prefix = rays_o.shape[:-1]    # masked 4096
            rays_o = rays_o.contiguous().view(-1, 3)  # masked [num_rays,3]  [177,3]
            rays_d = rays_d.contiguous().view(-1, 3) 
            N = rays_o.shape[0] # N = B * N, in fact   # masked 177
            device = rays_o.device

            aabb = self.model.aabb_train if self.model.training else self.model.aabb_infer
            
            nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.model.min_near)
            nears.unsqueeze_(-1)
            fars.unsqueeze_(-1)

            z_vals = torch.linspace(0.0, 1.0, opt.num_steps, device=device).unsqueeze(0) # [1, T]
            z_vals = z_vals.expand((N, opt.num_steps)) 
            z_vals = nears + (fars - nears) * z_vals
            sample_dist = (fars - nears) / opt.num_steps # [177,1], T], in [nears, fars]
            self.density_scale=1
            perturb=False
            if perturb:
                z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist   # [num_rays,num_bins]  177,512

            xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [N, T, 3] [ 177,512,3] [masked_rays, num_bins,3]
            xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:]) # a manual clip.n

            density_outputs = model.density(xyzs.reshape(-1, 3))
       
            for k, v in density_outputs.items():
                    density_outputs[k] = v.view(N, opt.num_steps, -1)

            if opt.upsample_steps > 0:
                with torch.no_grad():

                    deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T-1]  # 177, 511]
                    deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)  # [177, 512]

                    alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs['sigma'].squeeze(-1)) 
                    alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) 
                    weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] 

                    z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1]) 
                    new_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1], opt.upsample_steps, det=not self.training).detach() 

                    new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1) 
                    new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]), aabb[3:]) 

                new_density_outputs = model.density(new_xyzs.reshape(-1, 3))

                for k, v in new_density_outputs.items():
                    new_density_outputs[k] = v.view(N, opt.upsample_steps, -1)

                z_vals = torch.cat([z_vals, new_z_vals], dim=1) 
                z_vals, z_index = torch.sort(z_vals, dim=1)

                xyzs = torch.cat([xyzs, new_xyzs], dim=1) 
                xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))

                for k in density_outputs:
                    tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                    density_outputs[k] = torch.gather(tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))

            densities = density_outputs['sigma']
            self.z_vals, self.sample_dist, self.nears, self.fars = z_vals, sample_dist, nears, fars

            deltas = z_vals[..., 1:] - z_vals[..., :-1] 
            deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
            alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs['sigma'].squeeze(-1)) 
            alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) 
            weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] 

            dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
            for k, v in density_outputs.items():
                density_outputs[k] = v.view(-1, v.shape[-1])

            mask = weights > 1e-10 
            rgbs = model.color(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), yolo_details, mask=mask.reshape(-1), **density_outputs)      
            
            rgbs = rgbs.view(N, -1, 3) 
            

            return {
                'densities': densities,
                'rgbs': rgbs,
            }
        
        def evaluate_one_epoch(self, loader, name=None, post=False, grab_loss=False):
        
            self.log(f"++> Evaluate at epoch {self.epoch} ...")

            if name is None:
                name = f'{self.name}_ep{self.epoch:04d}'

            total_loss = 0
            if self.local_rank == 0:
                for metric in self.metrics:
                    metric.clear()

            self.model.eval()

            if self.ema is not None:
                self.ema.store()
                self.ema.copy_to()

            log_file_path = os.path.join(opt.workspace, 'log.txt')

            if self.local_rank == 0:
                pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

            psnr_values_W = []
            psnr_values_B = []
            ssim_values_W = []
            ssim_values_B = []
            lpips_values_W = []
            lpips_values_B = []
            GPU_TIME = []

            with torch.no_grad(), open(log_file_path, 'a') as log_file:
                self.local_step = 0
                

                for i, data in tqdm.tqdm(enumerate(loader), desc=f"Inference with {len(self.ckpt_list)} models"):    
                    self.local_step += 1
                    # self.N = data['rays_o'].shape[1]
                    self.B = data['rays_o'].shape[0]
                    self.H = data['H']
                    self.W = data['W']
                    self.N = self.W * self.H
                    self.images = data['images'] 
                    self.yolo_details = data['yolo_details']
                    self.rays_o = data['rays_o'].squeeze(0).squeeze(0)
                    self.rays_d = data['rays_d'].squeeze(0).squeeze(0)
                    self.max_ray_batch = 4096
                    self.T = opt.num_steps + opt.upsample_steps

                    all_rgbs = []
                    all_densities = []
                    max_densities = None


                    

                    for ckpts in self.ckpt_list:
                        self.ckpt = ckpts
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)

                        self.load_checkpoint(self.ckpt)

                        with torch.cuda.amp.autocast(enabled=self.fp16):
                            start.record()
                            results = self.batch_run(self.rays_o, self.rays_d, self.yolo_details, **vars(opt))

                            rgbs = results['rgbs']
                            densities = results['densities']

                            all_rgbs.append(rgbs)
                            all_densities.append(densities)

                            if len(ckpt_list) > 1:
                                if max_densities is None:
                                    max_densities = densities
                                    max_rgbs = rgbs
                                else:
                                    max_densities, max_rgbs = self.best_densities_and_colors_v3(densities, max_densities, rgbs, max_rgbs)
                                   

                            else:
                                max_densities, max_rgbs = densities, rgbs
                    
                    results = self.compute_metrics_both_backgrounds(trainer, data, max_densities, max_rgbs, self.H, self.W, opt, i)


                    end.record()

                   
                    torch.cuda.synchronize()
                    psnr_values_W.append(results['white']['psnr'])
                    psnr_values_B.append(results['black']['psnr'])
                    ssim_values_W.append(results['white']['ssim'])
                    ssim_values_B.append(results['black']['ssim'])
                    lpips_values_B.append(results['black']['lpips'])
                    lpips_values_W.append(results['white']['lpips'])

                    Inference_time = start.elapsed_time(end)
                    GPU_TIME.append(Inference_time)
                    log_file.write(f"Inference Time: {Inference_time} ms, SSIM_W: {results['white']['ssim']}, PSNR_W: {results['white']['psnr']}, LPIPS_W: {results['white']['lpips']}, SSIM_B: {results['black']['ssim']}, LPIPS_B: {results['black']['lpips']}, PSNR_B: {results['black']['psnr']}\n")
               
                max_psnr_w = np.max(psnr_values_W)
                max_psnr_b = np.max(psnr_values_B)
                max_ssim_w = np.max(ssim_values_W)
                max_ssim_b = np.max(ssim_values_B)
                max_lpips_w = np.min(lpips_values_W)
                max_lpips_b = np.min(lpips_values_B)
                avg_psnr_w = np.mean(psnr_values_W)
                avg_psnr_b = np.mean(psnr_values_B)
                avg_ssim_w = np.mean(ssim_values_W)
                avg_ssim_b = np.mean(ssim_values_B)
                avg_lpips_b = np.mean(lpips_values_B)
                avg_lpips_w = np.mean(lpips_values_W)
                final_psnr = (avg_psnr_b + avg_psnr_w)/2
                final_ssim = (avg_ssim_b + avg_ssim_w)/2
                final_lpips = (avg_lpips_b + avg_lpips_w)/2
                AVG_GPU_TIME = np.mean(GPU_TIME)
                
                log_file.write(f"Average PSNR White: {avg_psnr_w:.2f}\n")
                log_file.write(f"Average SSIM White: {avg_ssim_w:.2f}\n")
                log_file.write(f"Average PSNR Black: {avg_psnr_b:.2f}\n")
                log_file.write(f"Average SSIM Black: {avg_ssim_b:.2f}\n")
                log_file.write(f"Average LPIPS BLACK: {avg_lpips_b:.4f}\n")
                log_file.write(f"Average LPIPS White: {avg_lpips_w:.4f}\n")
                log_file.write(f"FINAL Average PSNR: {final_psnr:.2f}\n")
                log_file.write(f"FINAL Average SSIM: {final_ssim:.2f}\n")
                log_file.write(f"FINAL Average LPIPS: {final_lpips:.4f}\n")
                log_file.write(f"AVERAGE GPU TIME: {AVG_GPU_TIME:.4f}\n")
                log_file.write(f" BEST SSIM_BLACK: {max_ssim_b}, BEST SSIM_WHITE: {max_ssim_w}, BEST PSNR_BLACK: {max_psnr_b}, BEST PSNR_WHITE: {max_psnr_w}, BEST LPIPS_BLACK: {max_lpips_b}, BEST LPIPS_WHITE: {max_lpips_w}\n")


            if self.ema is not None:
                self.ema.restore()


    seed_everything(opt.seed)

    ckpt_list = gather_checkpoints(opt.ckpt_dir)
    obj_feats_paths = gather_obj_feats(opt.ckpt_dir)
    
    ckpt_list = [checkpoint for checkpoint in ckpt_list if any(obj in checkpoint[len(opt.ckpt_dir):] for obj in opt.objects_of_interest)]
    obj_feats_paths = {obj_name: path for obj_name, path in obj_feats_paths.items() if obj_name in opt.objects_of_interest}


criterion = torch.nn.MSELoss(reduction='none')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)


load_1_start = time.time()

load_1_end = time.time()

scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
metrics = [PSNRMeter(), LPIPSMeter(device=device), SSIMMeter()]

trainer = MultiTrainer('ngp', opt, model, device=device, workspace=f'{opt.workspace}', optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=50,ckpt_list=ckpt_list)

mask_details_test = []
opt.detected_object = None
test_dataset = NeRFDataset(opt, mask_details=None, device=device, type='val')

test_dataset = calculate_avg_feats_inference(test_dataset, obj_feats_paths)

test_loader = test_dataset.dataloader()

if test_loader.has_gt:
        
        trainer.evaluate_one_epoch(test_loader) 


    
   