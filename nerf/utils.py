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
from ultralytics import YOLO


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, yolo_details, intrinsics, H, W, N=-1, error_map=None, patch_size=1):

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics
    mask, bbox, obj_feats = yolo_details

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device)) # float    
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

   

    results = {}

    if N > 0:
        N = min(N, H*W)

        if patch_size > 1:

            
            num_patch = N // (patch_size ** 2)
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            # inds_x = (63 * i) // W - 1
            # inds_y = (63 * j) // H - 1
            inds = torch.stack([inds_x, inds_y], dim=-1) # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1) # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0) # [np, p^2, 2]
            inds = inds.view(-1, 2) # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1] # [N], flatten

            inds = inds.expand([B, N])

        elif error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) 
            inds = inds.expand([B, N])
        else:

            
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) 

            
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
           
            
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse 

        
        
      
        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)
        

        results['inds'] = inds
        
        mask = cv2.resize(mask, (64, 64), interpolation = cv2.INTER_NEAREST) # during training     

        
    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

        mask = cv2.resize(mask, (H, W), interpolation = cv2.INTER_NEAREST) if mask is not None else None # during evaluation                         

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    if mask is not None:
            
      mask = mask.flatten() # 4096 <==> 2073600
      mask = torch.from_numpy(mask.astype(np.bool8))
      mask = mask[None, ...] # 1, 4096

    
      if mask.sum() == 0:
          mask[:, mask.numel()//2] = True


    results['rays_o'] = rays_o
    results['rays_d'] = rays_d
    results['yolo_details'] = (mask, bbox, obj_feats)
   

    return results



def gather_checkpoints(base_dir):
    base_dir = os.path.dirname(base_dir)
    ckpt_list = []
    for entry in os.listdir(base_dir):
        entry_path = os.path.join(base_dir, entry)
        if os.path.isdir(entry_path) and '_' in entry:
            for root, _, files in os.walk(entry_path):
                final_checkpoint = ""
                for file in files:
                    if file.endswith('.pth'):
                        files.sort()
                        file = files[-1]
                        ckpt_list.append(os.path.join(root, file))
                        break
    return ckpt_list

def gather_obj_feats(base_dir):
    obj_feats = {}
    target_dir = os.path.join(base_dir, "obj_feats")
    
    if os.path.isdir(target_dir):
        for file in os.listdir(target_dir):
            if file.endswith('.pt'):
                object_name = file.removesuffix('.pt')
                obj_feats[object_name] = os.path.join(target_dir, file)
                
    return obj_feats

def calculate_feats(test_dataset, train_dataset):
    # yolo_feats = torch.tensor([item[-1] for item in train_dataset['yolo_details']])
    yolo_feats = np.stack([item[-1] for item in train_dataset['yolo_details']])
    yolo_feats = torch.from_numpy(yolo_feats)

    n_yolo_feats = [
        torch.sum(1 / torch.norm(test_pose[:3, :4][:, -1] - train_pose[:3, :4][:, -1], p='fro'))
        for test_pose in test_dataset.poses
        for train_pose in train_dataset['poses']
    ]

    n_yolo_feats = [
        (1 / sum(deltas)) * deltas.matmul(yolo_feats)
        for deltas in torch.split(torch.tensor(n_yolo_feats), len(train_dataset['poses']))
    ]

    if test_dataset.yolo_details == []:
        n_yolo_feats_mean = torch.stack(n_yolo_feats).mean(0)
        test_dataset.yolo_details = [
            (None, None, n_yolo_feats_mean)
            for i in range(len(test_dataset.images))
        ]

    else:

      test_dataset.yolo_details = [
          (item[0], item[1], n_yolo_feats[i])
          for i, item in enumerate(test_dataset.yolo_details)
      ]

    return test_dataset 




def calculate_feats2(test_dataset, obj_feats_paths):
    # Load object features for each object type
    obj_feats_dict = {}
    for obj_type, path in obj_feats_paths.items():
        obj_feats_dict[obj_type] = torch.load(path)

    n_yolo_feats_dict = {}

    for obj_type, obj_feats in obj_feats_dict.items():
        yolo_feats = torch.tensor([item[-1] for item in obj_feats['yolo_details']])

        n_yolo_feats = [
            torch.sum(1 / torch.norm(test_pose[:3, :4][:, -1] - train_pose[:3, :4][:, -1], p='fro'))
            for test_pose in test_dataset.poses
            for train_pose in obj_feats['poses']
        ]

        n_yolo_feats = [
            (1 / sum(deltas)) * deltas.matmul(yolo_feats)
            for deltas in torch.split(torch.tensor(n_yolo_feats), len(obj_feats['poses']))
        ]

        n_yolo_feats_dict[obj_type] = n_yolo_feats

    if test_dataset.yolo_details == []:
        for obj_type in obj_feats_dict.keys():
            n_yolo_feats_mean = torch.stack(n_yolo_feats_dict[obj_type]).mean(0)
            test_dataset.yolo_details = [
                (None, None, n_yolo_feats_mean)
                for i in range(len(test_dataset.images))
            ]
    else:
        for i, item in enumerate(test_dataset.yolo_details):
            obj_type = item[1]  # Assuming item[1] is the object type
            test_dataset.yolo_details[i] = (item[0], item[1], n_yolo_feats_dict[obj_type][i])

    return test_dataset



def calculate_avg_feats(test_dataset, train_dataset):
    # Calculate average YOLO features from the training dataset
    yolo_feats = np.stack([item[-1] for item in train_dataset['yolo_details']])
    yolo_feats = torch.from_numpy(yolo_feats)
    avg_yolo_feats = torch.mean(yolo_feats, dim=0)

    # Assign average features to test dataset
    if test_dataset.yolo_details == []:
        test_dataset.yolo_details = [
            (None, None, avg_yolo_feats)
            for _ in range(len(test_dataset.images))
        ]
    else:
        test_dataset.yolo_details = [
            (item[0], item[1], avg_yolo_feats)
            for item in test_dataset.yolo_details
        ]

    return test_dataset


def calculate_avg_feats_inference(test_dataset, obj_feats_paths):
    
    obj_feats_dict = {}
    for obj_type, path in obj_feats_paths.items():
        obj_feats_dict[obj_type] = torch.load(path)

    avg_yolo_feats_dict = {}
    for obj_type, obj_feats in obj_feats_dict.items():
        yolo_feats = torch.stack([torch.tensor(item[-1]) for item in obj_feats['yolo_details']])
        avg_yolo_feats_dict[obj_type] = torch.mean(yolo_feats, dim=0)

    if test_dataset.yolo_details == []:
        
        all_avg_feats = torch.stack(list(avg_yolo_feats_dict.values()))
        overall_avg_feats = torch.mean(all_avg_feats, dim=0)
        test_dataset.yolo_details = [
            (None, None, overall_avg_feats)
            for _ in range(len(test_dataset.images))
        ]
    else:
       
        for i, item in enumerate(test_dataset.yolo_details):
            obj_type = item[1]  
            if obj_type in avg_yolo_feats_dict:
                avg_feats = avg_yolo_feats_dict[obj_type]
            else:
                
                avg_feats = torch.mean(torch.stack(list(avg_yolo_feats_dict.values())), dim=0)
            test_dataset.yolo_details[i] = (item[0], item[1], avg_feats)

    test_dataset.yolo_idx = test_dataset.yolo_details[0][-1].shape

    return test_dataset


def calculate_feats2_inference(test_dataset, obj_feats_paths):
    
    obj_feats_dict = {}
    for obj_type, path in obj_feats_paths.items():
        obj_feats_dict[obj_type] = torch.load(path)

    n_yolo_feats_dict = {}

    for obj_type, obj_feats in obj_feats_dict.items():
        yolo_feats = torch.tensor([item[-1] for item in obj_feats['yolo_details']])

        n_yolo_feats = [
            torch.sum(1 / torch.norm(test_pose[:3, :4][:, -1] - train_pose[:3, :4][:, -1], p='fro'))
            for test_pose in test_dataset.poses
            for train_pose in obj_feats['poses']
        ]

        n_yolo_feats = [
            (1 / sum(deltas)) * deltas.matmul(yolo_feats)
            for deltas in torch.split(torch.tensor(n_yolo_feats), len(obj_feats['poses']))
        ]
        

        n_yolo_feats_dict[obj_type] = n_yolo_feats

    if test_dataset.yolo_details == []:
        for obj_type in obj_feats_dict.keys():
            n_yolo_feats_mean = torch.stack(n_yolo_feats_dict[obj_type]).mean(0)
            test_dataset.yolo_details = [
                (None, None, n_yolo_feats_mean)
                for i in range(len(test_dataset.images))
            ]
    else:
        for i, item in enumerate(test_dataset.yolo_details):
            obj_type = item[1]  
            test_dataset.yolo_details[i] = (item[0], item[1], n_yolo_feats_dict[obj_type][i])

    test_dataset.yolo_idx = test_dataset.yolo_details[0][-1].shape

    return test_dataset




def get_object_masks(root_path: str, type: str, disable_yolo: bool = False, model: str = 'yolov8x-seg.pt', use_full_mask: bool = False):
   
    yolo_model = YOLO(model)
    message = 'Using YOLO on' if not disable_yolo else 'Loading'
    with open(os.path.join(root_path, f'transforms_{type}.json'), 'r') as f:
                transform = json.load(f)
    frames = transform["frames"]
    
    all_images_dict = []

    if use_full_mask: 
        full_mask_details = []

        for f in tqdm.tqdm(frames, desc=f'{message} {type} data'):
            f_path = os.path.join(root_path, f['file_path'])

            if '.' not in os.path.basename(f_path):
                f_path += '.png'
               

            image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
            image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

            if not disable_yolo:

                yolo_res = yolo_model.predict(image, verbose = False)[0]
                masks = yolo_res.masks.data.cpu().numpy()

               
                full_mask = np.logical_or.reduce(masks)
            else:
                full_mask = np.ones_like(image[:, :, 0])

            full_mask_details.append((f_path, full_mask))
        return full_mask_details
        
    for f in tqdm.tqdm(frames, desc=f'{message} {type} data'):

        f_path = os.path.join(root_path, f['file_path'])
        
        if '.' not in os.path.basename(f_path):
            f_path += '.png'

        
        image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        yolo_res = yolo_model.predict(image, verbose = False)[0]
        
        
        detected_objects = set(yolo_res.boxes.cls.tolist())

        

        all_coco_classes = set(yolo_res.names.keys())
        if len(all_coco_classes)==80:
            detected_objects = set(yolo_res.boxes.cls.tolist())
            
        else:
            detected_objects = detected_objects.union(all_coco_classes)

        
        
        
        
        
        single_image_dict = []    

       
        for detected_object in detected_objects:
            
            
            if type=='test' and disable_yolo:
                
                single_object_dict = {
                    'object_type': yolo_res.names[detected_object],
                    'merged_mask': np.ones_like(image[:, :, 0])
                }

            else:
                
            
              indices = [index for index, value in enumerate(yolo_res.boxes.cls.tolist()) if value == detected_object]

            
              
              mask = yolo_res.masks.data[[indices]].cpu().numpy()
            
              merged_mask = np.logical_or.reduce(mask)

              single_object_dict = {
                  'object_type': yolo_res.names[detected_object],
                  'merged_mask': merged_mask
              }

            single_image_dict.append(single_object_dict)
        
        single_image_dict.append(
            {
                'object_type': 'background',
                'merged_mask': 1 - np.logical_or.reduce(yolo_res.masks.data.cpu().numpy()) if yolo_res.masks is not None else np.ones((640, 640)),
                
            }
        )
        
        all_images_dict.append({'f_path': f_path, 'obj_dict': single_image_dict})
        detected_objects_str = [yolo_res.names[index] for index in detected_objects]
        # breakpoint()

        detected_objects_str.append('background')

    return all_images_dict, detected_objects_str

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


def torch_vis_2d(x, renormalize=False):
   
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1,2,0).squeeze()
        x = x.detach().cpu().numpy()
        
    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')
    
    x = x.astype(np.float32)
    
    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()


def extract_fields(bound_min, bound_max, resolution, query_func, S=128):

    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]
                    u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    #print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    #print(u.shape, u.max(), u.min(), np.percentile(u, 50))
    
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, N, 3] or [B, H, W, 3], range[0, 1]
          
        self.mse = np.mean((preds - truths) ** 2)

        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        
        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def measure_mse(self):
        return self.mse

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f} \n MSE = {self.measure_mse():.6f}'


class SSIMMeter:
    def __init__(self, device=None):
        self.V = 0
        self.N = 0

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]

        ssim = structural_similarity_index_measure(preds, truths)

        self.V += ssim
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "SSIM"), self.measure(), global_step)

    def report(self):
        return f'SSIM = {self.measure():.6f}'


class LPIPSMeter:
    def __init__(self, net='alex', device=None):
        self.V = 0
        self.N = 0
        self.net = net

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs
    
    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.fn(truths, preds, normalize=True).item() # normalize=True: [0, 1] to [-1, 1]
        self.V += v
        self.N += 1
    
    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step)

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'

class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 yolo_feats_encoder_dim=144,
                 ):
        
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.depths = []
        self.densities = []
        self.colors = []

        model.to(self.device)
        model.yolo_feat_encoder = model.get_yolo_feat_encoder(yolo_feats_encoder_dim)
        
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        # optionally use LPIPS loss for patch-based training
        if self.opt.patch_size > 1:
            import lpips
            self.criterion_lpips = lpips.LPIPS(net='alex').to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
        
        # clip loss prepare
        if opt.rand_pose >= 0: # =0 means only using CLIP loss, >0 means a hybrid mode.
            from nerf.clip_utils import CLIPLoss
            self.clip_loss = CLIPLoss(self.device)
            self.clip_loss.prepare_text([self.opt.clip_text]) # only support one text prompt now...


    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    

    def train_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]

        yolo_details = data['yolo_details']
        
        # if there is no gt image, we train with CLIP loss.
        if 'images' not in data:

            B, N = rays_o.shape[:2]
            H, W = data['H'], data['W']

            outputs = self.model.render(rays_o, rays_d, yolo_details, staged=False, bg_color=None, perturb=True, force_all_rays=True, **vars(self.opt))
            pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
            
            

            loss = self.clip_loss(pred_rgb)
            
            return pred_rgb, None, loss

        images = data['images'] # [B, N, 3/4]

        B, N, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        if C == 3 or self.model.bg_radius > 0:
            bg_color = 1
        
        else:
            #bg_color = torch.ones(3, device=self.device) # [3], fixed white background
            #bg_color = torch.rand(3, device=self.device) # [3], frame-wise random.
            bg_color = torch.rand_like(images[..., :3]) # [N, 3], pixel-wise random.
            
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        outputs = self.model.render(rays_o, rays_d, yolo_details, staged=False, bg_color=bg_color, perturb=True, force_all_rays=False if self.opt.patch_size == 1 else True, **vars(self.opt))
        
        pred_rgb = outputs['image']

        
        
        
        loss = self.criterion(pred_rgb, gt_rgb).mean(-1) # [B, N, 3] --> [B, N]

       
        if self.opt.patch_size > 1:
            gt_rgb = gt_rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous()
            pred_rgb = pred_rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous()

           
            loss = loss + 1e-3 * self.criterion_lpips(pred_rgb, gt_rgb)

        
        if len(loss.shape) == 3:
            loss = loss.mean(0)
        if outputs['criterion_outside_mask'] is not None: loss = loss + 1e-08 * outputs['criterion_outside_mask']  # 1e09
        # update error_map
        if self.error_map is not None:
            index = data['index'] # [B]
            inds = data['inds_coarse'] # [B, N]

           
            error_map = self.error_map[index] # [B, H * W]

            
            error = loss.detach().to(error_map.device) # [B, N], already in [0, 1]
            
            
            ema_error = 0.1 * error_map.gather(1, inds) + 0.9 * error
            error_map.scatter_(1, inds, ema_error)

           
            self.error_map[index] = error_map

        loss = loss.mean()

        
        return pred_rgb, gt_rgb, loss

    def eval_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        images = data['images'] # [B, H, W, 3/4]
        yolo_details = data['yolo_details']
        
        B, H, W, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        outputs = self.model.render(rays_o, rays_d, yolo_details, staged=True, bg_color=bg_color, perturb=False, **vars(self.opt))
        time_taken = outputs['timing']

        
        densities = outputs['densities']
        rgbs= outputs['rgbs']
        
      

        pred_rgb = outputs['image'] #outputs_image
        pred_depth = outputs['depth'] #outputs_depth

        

        masked_pred_rgb = None #pred_rgb[0][mask].unsqueeze(0)
        masked_gt_rgb = None #gt_rgb[0][mask].unsqueeze(0)
                
        
        loss = self.criterion(pred_rgb, gt_rgb.view(1, H*W, 3)).mean()
       

        pred_rgb = pred_rgb.view(B, H, W, 3)
        pred_depth = pred_depth.view(B, H, W)
     
     
        
        return pred_rgb, pred_depth, gt_rgb, loss, masked_pred_rgb, masked_gt_rgb, densities,rgbs, time_taken
    

    
   
    def test_step(self, data, bg_color=None, perturb=False):  

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        H, W = data['H'], data['W']
        yolo_details = data['yolo_details']

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        outputs = self.model.render(rays_o, rays_d, yolo_details, staged=True, bg_color=bg_color, perturb=perturb, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        return pred_rgb, pred_depth

    def save_mesh(self, save_path=None, resolution=256, threshold=10):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))['sigma']
            return sigma

        vertices, triangles = extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=resolution, threshold=threshold, query_func=query_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

   

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.model.cuda_ray:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        # get a ref to error_map
        self.error_map = train_loader._data.error_map
        
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

        #    

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)
            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader, grab_loss=True)
                self.evaluate_one_epoch(valid_loader)
                
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None, post=False):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name, post=post)
        self.use_tensorboardX = use_tensorboardX

   

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

        with torch.no_grad():

            for i, data in enumerate(loader):
                
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth = self.test_step(data)

                if self.opt.color_space == 'linear':
                    preds = linear_to_srgb(preds)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)
                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                else:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)
        
        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)

        self.log(f"==> Finished Test.")
        
    
    # [GUI] just train for 16 steps, without any other overhead that may slow down rendering.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        
        loader = iter(train_loader)

        # mark untrained grid
        if self.global_step == 0:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        for _ in range(step):
            
            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
            
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
        
        return outputs

    
    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1):
        
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, None, intrinsics, rH, rW, -1)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }
        
        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed! (but not perturb the first sample)
                preds, preds_depth = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            
            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
                    
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")


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

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')


        total_time_stage_1 = 0
        total_time_stage_2 = 0

        with torch.no_grad():
            self.local_step = 0

            for data in loader:    
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, truths, loss, masked_preds, masked_truths, densities,rgbs, time_taken = self.eval_step(data)

                total_time_stage_1 += time_taken[0]
                total_time_stage_2 += time_taken[1]

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros._like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)

                
                loss_val = loss.item()
                
                total_loss += loss_val

                if grab_loss: 
                    average_loss = total_loss / self.local_step
                    self.stats["valid_loss"].append(average_loss)
                    return
                

               
                if self.local_rank == 0:

                    for metric in self.metrics:
                        if isinstance(metric, PSNRMeter): 
                            
                            metric.update(preds, truths)
                        else: metric.update(preds, truths)

                base_path = os.path.join(self.workspace, 'validation')
                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:    

                    for metric in self.metrics:
                      if isinstance(metric, PSNRMeter): 
                      
                        metric.update(preds, truths)
                      else: metric.update(preds, truths)
                    if(os.path.exists(base_path)):
                    # save image
                        save_path = os.path.join(base_path, f'{name}_{self.local_step:04d}_rgb.png')
                        save_path_depth = os.path.join(base_path, f'{name}_{self.local_step:04d}_depth.png')
                    else:
                        os.makedirs(base_path, exist_ok=True)
                        save_path = os.path.join(base_path, f'{name}_{self.local_step:04d}_rgb.png')
                        save_path_depth = os.path.join(base_path, f'{name}_{self.local_step:04d}_depth.png')

                    #self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    if self.opt.color_space == 'linear':
                        preds = linear_to_srgb(preds)

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()

                    if post:
                        # only save depths, densities & colors in post mode
                        self.depths.append(pred_depth) 
                        # self.densities.append(densities.cpu().numpy().tolist())
                        self.colors.append(pred) 

                    pred_depth = (pred_depth * 255).astype(np.uint8)
                    
                    cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth, pred_depth)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)
                    

        if loader.detected_object != 'all' and post == True:
          with open(f'{self.opt.workspace}_{loader.detected_object}/render_times.txt', "w") as file:
                    file.write(f"""rgbs & sigmas = {float(total_time_stage_1):.5f}s\npred_depth & pred_rgb = {float(total_time_stage_2):.5f}s""")
        else:
        
           pass


        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)
        

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                # if isinstance(metric, PSNRMeter): 
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.") 

    


    
   
    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
       
        if not best:
            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    if 'density_grid' in state['model']:
                        del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            pattern = f'{self.ckpt_path}/{self.name}_ep*.pth'
            checkpoint_list = sorted(glob.glob(pattern))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log(f"[WARN] No checkpoint found, model randomly initialized. Looked for {pattern}")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
        
        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")


def retrieve_mask(all_images_dict_test):
    masks = []
    for d in all_images_dict_test:
        if 'obj_dict' in d:
            for item in d['obj_dict']:
                masks.append((d['f_path'], np.ones((640, 640))))
    return masks
                
