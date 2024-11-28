import math
import time
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching


def sample_pdf(bins, weights, n_samples, det=False):
  

   
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
  
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def plot_pointcloud(pc, color=None):
   
    print('[visualize points]', pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
   
    axes = trimesh.creation.axis(axis_length=4)
   
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()


class NeRFRenderer(nn.Module):
    def __init__(self,
                 bound=1,
                 cuda_ray=False,
                 density_scale=1,
                 min_near=0.2,
                 density_thresh=0.01,
                 bg_radius=-1,
                 bound_inf=None,
                 ):
        super().__init__()

        self.bound = bound
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.grid_size = 128
        self.density_scale = density_scale
        self.min_near = min_near
        self.density_thresh = density_thresh
        self.bg_radius = bg_radius # radius of the background sphere.
        self.bound_inf = bound_inf

       
        aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])

        aabb_infer = torch.FloatTensor(bound_inf) if bound_inf is not None else aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        # extra state for cuda raymarching
        self.cuda_ray = cuda_ray
        if cuda_ray:
            # density grid
            density_grid = torch.zeros([self.cascade, self.grid_size ** 3]) # [CAS, H * H * H]
            density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
            self.register_buffer('density_grid', density_grid)
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0
            # step counter
            step_counter = torch.zeros(16, 2, dtype=torch.int32) # 16 is hardcoded for averaging...
            self.register_buffer('step_counter', step_counter)
            self.mean_count = 0
            self.local_step = 0
    
    def forward(self, x, d):
        raise NotImplementedError()

   
    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def reset_extra_state(self):
        if not self.cuda_ray:
            return 
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0


    
    def run(self, rays_o, rays_d, yolo_details, num_steps=128, upsample_steps=128, bg_color=None, perturb=False, **kwargs):
        
    
        prefix = rays_o.shape[:-1]    
        rays_o = rays_o.contiguous().view(-1, 3)  
        rays_d = rays_d.contiguous().view(-1, 3)  
        
        N = rays_o.shape[0] 
        device = rays_o.device

       
        aabb = self.aabb_train if self.training else self.aabb_infer
        aabb = aabb.to(device)        
        
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)
        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)
        
        
        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0) # [1, T]
        z_vals = z_vals.expand((N, num_steps)) # [N, T]
        z_vals = nears + (fars - nears) * z_vals # [N, T], in [nears, fars]
   	
        
        sample_dist = (fars - nears) / num_steps # [177,1]
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist   
       
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [N, T, 3] [ 177,512,3] [masked_rays, num_bins,3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:]) # a manual clip.n


        stage_1_start_time = time.time()
        
        
        
        density_outputs = self.density(xyzs.reshape(-1, 3)) # [N * T, 1]  
        criterion_outside_mask = None
        if self.training:
            criterion_outside_mask = torch.norm(density_outputs['sigma'].view(N, num_steps)[~yolo_details[0].squeeze(0)] - 0)
        
        
        for k, v in density_outputs.items():
                density_outputs[k] = v.view(N, num_steps, -1)

       
        densities = density_outputs['sigma']
       
        deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs['sigma'].squeeze(-1)) # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]
        

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])
       
        mask = weights > 1e-10 

        rgbs = self.color(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), yolo_details, mask=mask.reshape(-1), **density_outputs)      # [177*512 , 3]

      
        
        stage_1_end_time = time.time()

        stage_1_time_taken = stage_1_end_time - stage_1_start_time 


        stage_2_start_time = time.time()

       
        rgbs = rgbs.view(N, -1, 3) # [N, T+t, 3]   177, 512,3
        
       
        weights_sum = weights.sum(dim=-1) # [N]
        
       
        ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)   # [num_rays, num_bins] 177,512]
        depth = torch.sum(weights * ori_z_vals, dim=-1)     # 177

        
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) # [N, 3], in [0, 1]     #177, 3
        
        if self.bg_radius > 0:
            
            sph = raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius) # [N, 2] in [-1, 1]
            bg_color = self.background(sph, rays_d.reshape(-1, 3)) # [N, 3]
        elif bg_color is None:
            bg_color = 1

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        image = image.view(*prefix, 3)   # [177,3 ]
        depth = depth.view(*prefix)      # [177]

       

        stage_2_end_time = time.time()

        stage_2_time_taken = stage_2_end_time - stage_2_start_time

       
        return {
            'depth': depth,
            'image': image,
            'weights_sum': weights_sum,
            'criterion_outside_mask': criterion_outside_mask,
            'densities': densities,
            'rgbs':rgbs,
            'timing': [stage_1_time_taken, stage_2_time_taken],
        }



    
    def run_cuda(self, rays_o, rays_d, dt_gamma=0, bg_color=None, perturb=False, force_all_rays=False, max_steps=1024, T_thresh=1e-4, **kwargs):
        

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)

      
        if self.bg_radius > 0:
            
            sph = raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius) # [N, 2] in [-1, 1]
            bg_color = self.background(sph, rays_d) # [N, 3]
        elif bg_color is None:
            bg_color = 1

        results = {}

        if self.training:
            
            counter = self.step_counter[self.local_step % 16]
            counter.zero_() # set to 0
            self.local_step += 1

            xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, counter, self.mean_count, perturb, 128, force_all_rays, dt_gamma, max_steps)

            
            
            sigmas, rgbs = self(xyzs, dirs)
            
            sigmas = self.density_scale * sigmas

           
            if len(sigmas.shape) == 2:
                K = sigmas.shape[0]
                depths = []
                images = []
                for k in range(K):
                    weights_sum, depth, image = raymarching.composite_rays_train(sigmas[k], rgbs[k], deltas, rays, T_thresh)
                    image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
                    depth = torch.clamp(depth - nears, min=0) / (fars - nears)
                    images.append(image.view(*prefix, 3))
                    depths.append(depth.view(*prefix))
            
                depth = torch.stack(depths, axis=0) # [K, B, N]
                image = torch.stack(images, axis=0) # [K, B, N, 3]

            else:

                weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, deltas, rays, T_thresh)
                image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
                depth = torch.clamp(depth - nears, min=0) / (fars - nears)
                image = image.view(*prefix, 3)
                depth = depth.view(*prefix)
            
            results['weights_sum'] = weights_sum

        else:
           
            
            dtype = torch.float32
            
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            
            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = nears.clone() # [N]

            step = 0
            
            while step < max_steps:

                # count alive rays 
                n_alive = rays_alive.shape[0]
                
                # exit loop
                if n_alive <= 0:
                    break

              
                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, 128, perturb if step == 0 else False, dt_gamma, max_steps)

                sigmas, rgbs = self(xyzs, dirs)
                
                sigmas = self.density_scale * sigmas

                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh)

                rays_alive = rays_alive[rays_alive >= 0]

                

                step += n_step

            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth = torch.clamp(depth - nears, min=0) / (fars - nears)
            image = image.view(*prefix, 3)
            depth = depth.view(*prefix)
        
        results['depth'] = depth
        results['image'] = image

        return results

    @torch.no_grad()
    def mark_untrained_grid(self, poses, intrinsic, S=64):
        
        if not self.cuda_ray:
            return
        
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)

        B = poses.shape[0]
        
        fx, fy, cx, cy = intrinsic
        
        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

        count = torch.zeros_like(self.density_grid)
        poses = poses.to(count.device)

        # 5-level loop, forgive me...

        for xs in X:
            for ys in Y:
                for zs in Z:
                    
                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    world_xyzs = (2 * coords.float() / (self.grid_size - 1) - 1).unsqueeze(0) # [1, N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_world_xyzs = world_xyzs * (bound - half_grid_size)

                        # split batch to avoid OOM
                        head = 0
                        while head < B:
                            tail = min(head + S, B)

                            # world2cam transform (poses is c2w, so we need to transpose it. Another transpose is needed for batched matmul, so the final form is without transpose.)
                            cam_xyzs = cas_world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
                            cam_xyzs = cam_xyzs @ poses[head:tail, :3, :3] # [S, N, 3]
                            
                            # query if point is covered by any camera
                            mask_z = cam_xyzs[:, :, 2] > 0 # [S, N]
                            mask_x = torch.abs(cam_xyzs[:, :, 0]) < cx / fx * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask_y = torch.abs(cam_xyzs[:, :, 1]) < cy / fy * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask = (mask_z & mask_x & mask_y).sum(0).reshape(-1) # [N]

                            # update count 
                            count[cas, indices] += mask
                            head += S
    
        # mark untrained grid as -1
        self.density_grid[count == 0] = -1

        print(f'[mark untrained grid] {(count == 0).sum()} from {self.grid_size ** 3 * self.cascade}')

    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        # call before each epoch to update extra states.

        if not self.cuda_ray:
            return 
        
        ### update density grid

        tmp_grid = - torch.ones_like(self.density_grid)
        
        # full update.
        if self.iter_density < 16:
        #if True:
            X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

            for xs in X:
                for ys in Y:
                    for zs in Z:
                        
                       
                        xx, yy, zz = custom_meshgrid(xs, ys, zs)
                        coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                        indices = raymarching.morton3D(coords).long() # [N]
                        xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                       
                        for cas in range(self.cascade):
                            bound = min(2 ** cas, self.bound)
                            half_grid_size = bound / self.grid_size
                           
                            cas_xyzs = xyzs * (bound - half_grid_size)
                            
                            cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                            
                            sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                            sigmas *= self.density_scale
                           
                            tmp_grid[cas, indices] = sigmas

        
        else:
            N = self.grid_size ** 3 // 4 # H * H * H / 4
            for cas in range(self.cascade):
               
                coords = torch.randint(0, self.grid_size, (N, 3), device=self.density_bitfield.device) # [N, 3], in [0, 128)
                indices = raymarching.morton3D(coords).long() # [N]
                
                occ_indices = torch.nonzero(self.density_grid[cas] > 0).squeeze(-1) # [Nz]
                rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long, device=self.density_bitfield.device)
                occ_indices = occ_indices[rand_mask] 
                occ_coords = raymarching.morton3D_invert(occ_indices) 
                
                indices = torch.cat([indices, occ_indices], dim=0)
                coords = torch.cat([coords, occ_coords], dim=0)
                
                xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 
                bound = min(2 ** cas, self.bound)
                half_grid_size = bound / self.grid_size
                
                cas_xyzs = xyzs * (bound - half_grid_size)
                
                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
               
                sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                sigmas *= self.density_scale
                 
                tmp_grid[cas, indices] = sigmas

        

        
        valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item() 
       
        self.iter_density += 1

        
        density_thresh = min(self.mean_density, self.density_thresh)
        self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)

       
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

      

    def render(self, rays_o, rays_d, yolo_details, staged=False, max_ray_batch=4096, **kwargs):
       

        if self.cuda_ray:
            _run = self.run_cuda
        else:
            _run = self.run

        B, N = rays_o.shape[:2]
        
        device = rays_o.device

        
        if staged and not self.cuda_ray:
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)
            T = kwargs['num_steps'] + kwargs['upsample_steps']
            densities = torch.empty((B, N, T), device=device)
            
            rgbs = torch.empty((B, N, T, 3), device=device)
            
            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], yolo_details, **kwargs)
                    depth[b:b+1, head:tail] = results_['depth']
                    image[b:b+1, head:tail] = results_['image']
                   
                    densities[b:b+1, head:tail] = results_['densities'].permute(2,0,1)
                    
                   
                    rgbs[b:b+1, head:tail] = results_['rgbs']
                    
                    

                    head += max_ray_batch
            
            results = {}
            results['depth'] = depth
            results['image'] = image
            results['densities'] = densities
            results['rgbs'] = rgbs
            results['timing'] = results_['timing']

        else:
            results = _run(rays_o, rays_d, yolo_details, **kwargs)
            
        return results
    

    
   