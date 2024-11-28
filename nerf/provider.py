import os
import cv2
import glob
import json
from cv2 import transform
import tqdm
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
from ultralytics import YOLO

import trimesh

import torch
from torch.utils.data import DataLoader

from .utils import get_rays

from PIL import Image
import cv2
from nerf.dilations import increase_dilation_percentage

# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def visualize_poses(poses, size=0.1):
   

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


class NeRFDataset:
    def __init__(self, opt, mask_details, device, type='train', downscale=1, n_test=10):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose
        self.mask_details = mask_details
        self.detected_object = opt.detected_object
        self.yolo_idx = None
        


       
        self.yolo_model = YOLO('yolov8x-seg.pt')
        self.yolo_model = YOLO(opt.yolo_model)
        self.yolo_torch_model = self.yolo_model.model
        self.yolo_torch_model.eval()
       

        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

        
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            
            if type == 'all':
                transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            
            elif type == 'trainval':
                with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                    transform = json.load(f)
                with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                    transform_val = json.load(f)
                transform['frames'].extend(transform_val['frames'])
            
            else:
                with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)

        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            
            self.H = self.W = None
        
       
        frames = transform["frames"]
        frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        
        
        if self.mode == 'colmap' and type == 'test':
            print('Colmap - Test')
            
            
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.images = None
            
            self.torch_images = []
            
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
               
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)

        else:
            
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
               

            self.poses = []
            self.images = []
            self.yolo_details = []

           
            for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                    f_path += '.png' # so silly...

               
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                if opt.edit_x != 0:
                    pose[0, 3] = pose[0, 3] + opt.edit_x
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)
                
                if self.detected_object == 'background':
                    image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                else:

                       
                    image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                  
                    results = self.yolo_model(image_rgb)
                    
                    
                    masks = results[0].masks
                    masks = masks.data.cpu().numpy()  
                    height, width, _ = image.shape
                    final_mask = np.zeros((height, width), dtype=np.uint8)
                    
                  
                    for mask in masks:
                        final_mask = np.maximum(final_mask, mask)
                    
                    
                    alpha_channel = (final_mask * 255).astype(np.uint8)
                    
                   
                    image = cv2.merge((image[:, :, 0], image[:, :, 1], image[:, :, 2], alpha_channel))

                
               
                
                if self.H is None or self.W is None:
                    self.H = image.shape[0] // downscale
                    self.W = image.shape[1] // downscale

                
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                self.W = 640; self.H = 640
                if image.shape[0] != self.H or image.shape[1] != self.W:
                  image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                   
                
                image = image.astype(np.float32) / 255 # [H, W, 3/4] 
                n_image = image
                image_copy = image.copy()
                
                image_torch = torch.from_numpy(image_copy)
                image_torch = image_torch.permute(2, 0, 1).unsqueeze(0) # image shape -> 1, 3, 640, 640
                
               
                if len(image_torch.shape) == 4: image_torch = image_torch[:, :3, :, :]

                if mask_details is not None:

                  self.yolo_torch_model.to(device)
                  with torch.no_grad():
                      nn_result = self.yolo_torch_model(image_torch.to(device))[1][0][2] # shape -> 1, 144, 20, 20 i.e last activations
                                                                            # 1, 84, 84000
                                                                            # 3 items --> 1, 144, 20, 20 | 1, 144, 40, 40 | 1, 144, 80, 80

                                                                        # 3 items --> 1, 144, 20, 20 | 1, 144, 40, 40 | 1, 144, 80, 80
                  
                  nn_result = nn_result.squeeze(0).permute(1, 2, 0).cpu().numpy() 

                
                  yolo_res = self.yolo_model.predict(image[..., :3] * 255., verbose = False,   conf=0.1)[0] # YOLO inference
                  
                

                  mask = None

                  # retrieve appropriate mask, if not present, skip..
                  for d in mask_details:
                      if d[0] == f_path:
                          mask = d[1]
                      else: 
                          continue
                  
                  
                  
                  if (mask is None) or (mask.sum() == 0):
                     
                      mask = np.zeros_like(n_image[:,:,0]).astype(np.bool8)
                    
                      mask[mask.shape[0]//2, mask.shape[1]//2] = 1
                  
                  n_image = cv2.bitwise_and(n_image, n_image, mask = mask.astype(np.uint8))  
                  
                  
                #   n_image = n_image * mask.astype(np.uint8)[...,np.newaxis]
                  
                  if yolo_res.boxes.data.shape[0] != 0:
                    bbox = yolo_res.boxes[0].xyxy.tolist()[0]
                    bbox = list(map(int, bbox))
                  else: bbox = None
                  
                  resized_mask = cv2.resize(mask.squeeze().astype(np.uint8), (self.W // 32, self.H // 32), interpolation=cv2.INTER_NEAREST) # shape -> 20, 20
                  resized_mask = resized_mask.astype(np.uint8)

                 
                  nn_result = cv2.bitwise_and(nn_result, nn_result, mask = resized_mask) # shape -> 20, 20, 144
                
                 
                  obj_feats = np.mean(nn_result, axis = (0, 1)) 
              
                  obj_feat_shape = obj_feats.shape
                  self.yolo_idx = obj_feat_shape
 
                  
                  self.yolo_details.append((mask.astype(np.uint8), bbox, obj_feats))
                  
                  self.poses.append(pose)
                  self.images.append(n_image)

                  if self.opt.save_gt:
                      save_processed_gt_path = f"{self.opt.workspace}/ground_truths/{self.detected_object}"
                      os.makedirs(save_processed_gt_path, exist_ok=True)
                      cv2.imwrite(os.path.join(save_processed_gt_path, Path(f_path).name), cv2.cvtColor((255.0 * n_image), cv2.COLOR_BGR2RGB).astype(np.uint8))
                
                else:

                  self.poses.append(pose)
                  self.images.append(image)

        
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        
       
        
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

    def collate(self, index):


        B = len(index) 
       
        if self.rand_pose == 0 or index[0] >= len(self.poses):

            poses = rand_poses(B, self.device, radius=self.radius)

           
            s = np.sqrt(self.H * self.W / self.num_rays) 
            rH, rW = int(self.H / s), int(self.W / s)
            
            rays = get_rays(poses, self.yolo_details[index[0]], self.intrinsics / s, rH, rW, -1)
            
            return {
                'H': rH,
                'W': rW,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],    
            }

        poses = self.poses[index].to(self.device) # [B, 4, 4]

        error_map = None if self.error_map is None else self.error_map[index]
        rays = get_rays(poses, self.yolo_details[index[0]], self.intrinsics, self.H, self.W, self.num_rays, error_map, self.opt.patch_size)
       
        
        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],  
            'rays_d': rays['rays_d'],	
        }

        if self.images is not None:
            images = self.images[index].to(self.device) 
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) 
            results['images'] = images
        
        
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']
	     
        results['yolo_details'] = rays['yolo_details']
        
        return results
    

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose # index >= size means we use random pose
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self 
        loader.has_gt = self.images is not None
        loader.detected_object = self.detected_object
        loader.yolo_feats_encoder_dim = self.yolo_idx[0]
        return loader
