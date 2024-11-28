
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import numpy as np

# import tinycudann as tcnn
# from activation import trunc_exp
# from .renderer import NeRFRenderer


# class NeRFNetwork(NeRFRenderer):
#     def __init__(self,
#                  encoding="HashGrid",
#                  encoding_dir="SphericalHarmonics",
#                  num_layers=5,  #2
#                  hidden_dim=128, # 64
#                  geo_feat_dim=15,
#                  num_layers_color=3,
#                  hidden_dim_color=128,  #64
#                  yolo_encoding_dim=16,
#                  bound=1,
#                  n_chunks=5,
#                  **kwargs
#                  ):
#         super().__init__(bound, **kwargs)

#         # sigma network
#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
#         self.yolo_encoding_dim = yolo_encoding_dim
#         self.geo_feat_dim = geo_feat_dim

#         per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))

#         self.encoder = tcnn.Encoding(
#             n_input_dims=3,
#             encoding_config={
#                 "otype": "HashGrid",
#                 "n_levels": 16,
#                 "n_features_per_level": 2,
#                 "log2_hashmap_size": 19,
#                 "base_resolution": 16,
#                 "per_level_scale": per_level_scale,
#             },
#         )

#         self.sigma_net = tcnn.Network(
#             n_input_dims=32,
#             n_output_dims=1 + self.geo_feat_dim,
#             network_config={
#                 "otype": "FullyFusedMLP",
#                 "activation": "ReLU",
#                 "output_activation": "None",
#                 "n_neurons": hidden_dim,
#                 "n_hidden_layers": num_layers - 1,
#             },
#         )

#         self.yolo_feat_encoder = tcnn.Network(
#           n_input_dims=66,
#         #   n_input_dims=1280,
#           n_output_dims=self.yolo_encoding_dim,
#           network_config={
#               "otype": "FullyFusedMLP",
#               "activation": "ReLU",
#               "output_activation": "None",
#               "n_hidden_layers": 1,
#               "n_neurons": 16,
#           },
#         )

#         # color network
#         self.num_layers_color = 5#num_layers_color     2   
#         self.hidden_dim_color = 128#hidden_dim_color     64

#         self.encoder_dir = tcnn.Encoding(
#             n_input_dims=3,
#             encoding_config={
#                 "otype": "SphericalHarmonics",
#                 "degree": 4,
#             },
#         )

#         self.in_dim_color = self.encoder_dir.n_output_dims + self.geo_feat_dim

#         # chunk size for yolo object features
#         self.n_chunks = n_chunks

#         self.color_net = tcnn.Network(
#             n_input_dims=self.in_dim_color + self.yolo_encoding_dim, # expand input dimensions to contain object features
#             n_output_dims=3,
#             network_config={
#                 "otype": "FullyFusedMLP",
#                 "activation": "ReLU",
#                 "output_activation": "None",
#                 "n_neurons": hidden_dim_color,
#                 "n_hidden_layers": num_layers_color - 1,
#             },
#         )

    
#     def forward(self, x, d,yolo_details=None):
#         # x: [N, 3], in [-bound, bound]
#         # d: [N, 3], nomalized in [-1, 1]


#         # sigma
#         x = (x + self.bound) / (2 * self.bound) # to [0, 1]
#         x = self.encoder(x)
        
#         h = self.sigma_net(x)

#         #sigma = F.relu(h[..., 0])
#         sigma = trunc_exp(h[..., 0])
#         geo_feat = h[..., 1:]
#         _, _, obj_feat = (yolo_details[0],yolo_details[1],yolo_details[2])
        
#         # color
#         d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
#         d = self.encoder_dir(d)

#         #p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        
#         h = torch.cat([d, geo_feat, obj_feat], dim=-1)
#         h = self.color_net(h)
        
#         # sigmoid activation for rgb
#         color = torch.sigmoid(h)

#         return sigma, color

#     def density(self, x, yolo_details=None):
#         # x: [N, 3], in [-bound, bound]
#         # if yolo_details is not None:
#         #     mask, _, _ = yolo_details
#         #     mask_resized = F.interpolate(mask.squeeze(0).reshape(640, 640).unsqueeze(0).unsqueeze(0).float(), size=(64, 64), mode='bilinear', align_corners=False)
#         #     mask_resized = mask_resized.flatten()
#         #     x = x.reshape(x.shape[0]//512, 512, -1)
#         #     x = x[(mask_resized > 0)]
#         x = (x + self.bound) / (2 * self.bound) # to [0, 1]

#         x = self.encoder(x)
        
#         h = self.sigma_net(x)

#         #sigma = F.relu(h[..., 0])
#         sigma = trunc_exp(h[..., 0])
#         geo_feat = h[..., 1:]
#         # _, _, obj_feat = (yolo_details[0],yolo_details[1],yolo_details[2])

#         return {
#             'sigma': sigma,
#             'geo_feat': geo_feat,
#         }

#     # allow masked inference
#     def color(self, x, d, yolo_details, mask=None, geo_feat=None, **kwargs):
#         # x: [N, 3] in [-bound, bound]
#         # mask: [N,], bool, indicates where we actually needs to compute rgb.
        
        
#         # retrieve only obj_feat from yolo_details which is mask, bbox, obj_feat
#         _, _, raw_obj_feat = (yolo_details[0],yolo_details[1],yolo_details[2])
#         raw_obj_feat = torch.tensor(raw_obj_feat, device=x.device)
        
#         # pass through the obj_feat encoder network
#         # breakpoint()
#         obj_feat = self.yolo_feat_encoder(raw_obj_feat.unsqueeze(0))


#         x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        
#         if mask is not None:
#             rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
#             # in case of empty mask
#             if not mask.any():
#                 return rgbs
#             x = x[mask]
#             d = d[mask]
#             geo_feat = geo_feat[mask]

#         # color
#         d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
#         d = self.encoder_dir(d)
      
#         obj_feat = obj_feat.squeeze(0).repeat(x.shape[0], 1)
        
#         h = torch.cat([d, geo_feat, obj_feat], dim=-1)

#         h = self.color_net(h)
        
#         # sigmoid activation for rgb
#         h = torch.sigmoid(h)
        
#         if mask is not None:
#             rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
#         else:
#             rgbs = h
        

#         return rgbs        

#     # optimizer utils
#     def get_params(self, lr):

#         params = [
#             {'params': self.encoder.parameters(), 'lr': lr},
#             {'params': self.sigma_net.parameters(), 'lr': lr},
#             {'params': self.encoder_dir.parameters(), 'lr': lr},
#             {'params': self.color_net.parameters(), 'lr': lr}, 
#         ]
#         if self.bg_radius > 0:
#             params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
#             params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
#         return params
















# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import numpy as np

# import tinycudann as tcnn
# from activation import trunc_exp
# from .renderer import NeRFRenderer


# class NeRFNetwork(NeRFRenderer):
#     def __init__(self,
#                  encoding="HashGrid",
#                  encoding_dir="SphericalHarmonics",
#                  num_layers=5,  #2
#                  hidden_dim=128, # 64
#                  geo_feat_dim=15,
#                  num_layers_color=3,
#                  hidden_dim_color=128,  #64
#                  bound=1,
#                  n_chunks=5,
#                  **kwargs
#                  ):
#         super().__init__(bound, **kwargs)

#         # sigma network
#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
#         self.geo_feat_dim = geo_feat_dim

#         per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))

#         self.encoder = tcnn.Encoding(
#             n_input_dims=3,
#             encoding_config={
#                 "otype": "HashGrid",
#                 "n_levels": 16,
#                 "n_features_per_level": 2,
#                 "log2_hashmap_size": 19,
#                 "base_resolution": 16,
#                 "per_level_scale": per_level_scale,
#             },
#         )

#         self.sigma_net = tcnn.Network(
#             n_input_dims=32,
#             n_output_dims=1 + self.geo_feat_dim,
#             network_config={
#                 "otype": "FullyFusedMLP",
#                 "activation": "ReLU",
#                 "output_activation": "None",
#                 "n_neurons": hidden_dim,
#                 "n_hidden_layers": num_layers - 1,
#             },
#         )

#         # color network
#         self.num_layers_color = 5#num_layers_color        
#         self.hidden_dim_color = 128#hidden_dim_color

#         self.encoder_dir = tcnn.Encoding(
#             n_input_dims=3,
#             encoding_config={
#                 "otype": "SphericalHarmonics",
#                 "degree": 4,
#             },
#         )

#         self.in_dim_color = self.encoder_dir.n_output_dims + self.geo_feat_dim

#         # chunk size for yolo object features
#         self.n_chunks = n_chunks

#         self.color_net = tcnn.Network(
#             n_input_dims=self.in_dim_color + self.n_chunks, # expand input dimensions to contain object features
#             n_output_dims=3,
#             network_config={
#                 "otype": "FullyFusedMLP",
#                 "activation": "ReLU",
#                 "output_activation": "None",
#                 "n_neurons": hidden_dim_color,
#                 "n_hidden_layers": num_layers_color - 1,
#             },
#         )

    
#     def forward(self, x, d):
#         # x: [N, 3], in [-bound, bound]
#         # d: [N, 3], nomalized in [-1, 1]


#         # sigma
#         x = (x + self.bound) / (2 * self.bound) # to [0, 1]
#         x = self.encoder(x)
        
#         h = self.sigma_net(x)

#         #sigma = F.relu(h[..., 0])
#         sigma = trunc_exp(h[..., 0])
#         geo_feat = h[..., 1:]

#         # color
#         d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
#         d = self.encoder_dir(d)

#         #p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        
#         h = torch.cat([d, geo_feat], dim=-1)
#         h = self.color_net(h)
        
#         # sigmoid activation for rgb
#         color = torch.sigmoid(h)

#         return sigma, color

#     def density(self, x):
#         # x: [N, 3], in [-bound, bound]

#         x = (x + self.bound) / (2 * self.bound) # to [0, 1]

#         x = self.encoder(x)
        
#         h = self.sigma_net(x)

#         #sigma = F.relu(h[..., 0])
#         sigma = trunc_exp(h[..., 0])
#         geo_feat = h[..., 1:]

#         return {
#             'sigma': sigma,
#             'geo_feat': geo_feat,
#         }

#     # allow masked inference
#     def color(self, x, d, yolo_details, mask=None, geo_feat=None, **kwargs):
#         # x: [N, 3] in [-bound, bound]
#         # mask: [N,], bool, indicates where we actually needs to compute rgb.

#         # retrieve only obj_feat from yolo_details which is mask, bbox, obj_feat
#         _, _, obj_feat = (yolo_details[0],yolo_details[1],yolo_details[2])
#         obj_feat = torch.tensor(obj_feat)

#         x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        
#         if mask is not None:
#             rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
#             # in case of empty mask
#             if not mask.any():
#                 return rgbs
#             x = x[mask]
#             d = d[mask]
#             geo_feat = geo_feat[mask]

#         # color
#         d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
#         d = self.encoder_dir(d)
        
#         if self.n_chunks != 0:

#             # chunk the obj_feat to specified number of chunks
#             divisor_chunks = obj_feat.shape[0] // self.n_chunks
#             remainder_chunks = obj_feat.shape[0] % self.n_chunks

#             chunked_obj_feat = torch.chunk(obj_feat[:divisor_chunks * self.n_chunks], self.n_chunks)

#             # take the mean of each chunk
#             avg_chunk_obj_feat = [_.mean() for _ in chunked_obj_feat]
#             avg_chunk_obj_feat = torch.tensor(avg_chunk_obj_feat)  

#             # expand to size N      
#             avg_chunk_obj_feat = avg_chunk_obj_feat.repeat(x.shape[0], 1).to(x.device) 
            
#             h = torch.cat([d, geo_feat, avg_chunk_obj_feat], dim=-1)
        
#         else:
#             h = torch.cat([d, geo_feat], dim=-1)

#         h = self.color_net(h)
        
#         # sigmoid activation for rgb
#         h = torch.sigmoid(h)
        
#         if mask is not None:
#             rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
#         else:
#             rgbs = h

#         return rgbs        

#     # optimizer utils
#     def get_params(self, lr):

#         params = [
#             {'params': self.encoder.parameters(), 'lr': lr},
#             {'params': self.sigma_net.parameters(), 'lr': lr},
#             {'params': self.encoder_dir.parameters(), 'lr': lr},
#             {'params': self.color_net.parameters(), 'lr': lr}, 
#         ]
#         if self.bg_radius > 0:
#             params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
#             params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
#         return params


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import tinycudann as tcnn
from activation import trunc_exp
from .renderer import NeRFRenderer



class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="HashGrid",
                 encoding_dir="SphericalHarmonics",
                 num_layers=2,  #5
                 hidden_dim=64, # 128
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,  #128
                 yolo_encoding_dim=16,
                 bound=1,
                 n_chunks=5,
                 yolo_feats_encoder_dim=144,
                 **kwargs
                 ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.yolo_encoding_dim = yolo_encoding_dim
        self.geo_feat_dim = geo_feat_dim
        self.yolo_feats_encoder_dim = yolo_feats_encoder_dim
        

        per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
        )

        self.sigma_net = tcnn.Network(
            n_input_dims=32,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )
        
        self.yolo_feat_encoder = tcnn.Network(
        #   n_input_dims=69,
          n_input_dims=self.yolo_feats_encoder_dim,

        #   n_input_dims=1280,
          n_output_dims=self.yolo_encoding_dim,
          network_config={
              "otype": "FullyFusedMLP",
              "activation": "ReLU",
              "output_activation": "None",
              "n_hidden_layers": 1,
              "n_neurons": 16,
          },
        )

        # color network
        self.num_layers_color = 2 #num_layers_color     5 
        self.hidden_dim_color = 64 #hidden_dim_color     128

        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.in_dim_color = self.encoder_dir.n_output_dims + self.geo_feat_dim

        # chunk size for yolo object features
        self.n_chunks = n_chunks

        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color + self.yolo_encoding_dim, # expand input dimensions to contain object features
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    
    def forward(self, x, d,yolo_details=None):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]


        # sigma
        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        x = self.encoder(x)
        
        h = self.sigma_net(x)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]
        _, _, obj_feat = (yolo_details[0],yolo_details[1],yolo_details[2])
        # breakpoint()
        
        # color
        d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        #p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        
        h = torch.cat([d, geo_feat, obj_feat], dim=-1)
        h = self.color_net(h)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color

    def density(self, x, yolo_details=None):
        # x: [N, 3], in [-bound, bound]
        # if yolo_details is not None:
        #     mask, _, _ = yolo_details
        #     mask_resized = F.interpolate(mask.squeeze(0).reshape(640, 640).unsqueeze(0).unsqueeze(0).float(), size=(64, 64), mode='bilinear', align_corners=False)
        #     mask_resized = mask_resized.flatten()
        #     x = x.reshape(x.shape[0]//512, 512, -1)
        #     x = x[(mask_resized > 0)]
        x = (x + self.bound) / (2 * self.bound) # to [0, 1]

        x = self.encoder(x)
        
        h = self.sigma_net(x)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]
        # _, _, obj_feat = (yolo_details[0],yolo_details[1],yolo_details[2])

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    # allow masked inference
    def color(self, x, d, yolo_details, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.
        
        
        # retrieve only obj_feat from yolo_details which is mask, bbox, obj_feat
        _, _, raw_obj_feat = (yolo_details[0],yolo_details[1],yolo_details[2])
        raw_obj_feat = torch.tensor(raw_obj_feat, device=x.device)
        
        # pass through the obj_feat encoder network
        # breakpoint()
        obj_feat = self.yolo_feat_encoder(raw_obj_feat.unsqueeze(0))


        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        
        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        # color
        d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)
      
        obj_feat = obj_feat.squeeze(0).repeat(x.shape[0], 1)
        
        h = torch.cat([d, geo_feat, obj_feat], dim=-1)

        h = self.color_net(h)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)
        
        if mask is not None: 
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h
        

        return rgbs        

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
            {'params': self.yolo_feat_encoder.parameters(), 'lr':lr},
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params

    def get_yolo_feat_encoder(self, yolo_feats_encoder_dim):
        return tcnn.Network(
        #   n_input_dims=69,
          n_input_dims=yolo_feats_encoder_dim,

        #   n_input_dims=1280,
          n_output_dims=self.yolo_encoding_dim,
          network_config={
              "otype": "FullyFusedMLP",
              "activation": "ReLU",
              "output_activation": "None",
              "n_hidden_layers": 1,
              "n_neurons": 16,
          },
        )



# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import numpy as np

# import tinycudann as tcnn
# from activation import trunc_exp
# from .renderer import NeRFRenderer


# class NeRFNetwork(NeRFRenderer):
#     def __init__(self,
#                  encoding="HashGrid",
#                  encoding_dir="SphericalHarmonics",
#                  num_layers=5,  #2
#                  hidden_dim=128, # 64
#                  geo_feat_dim=15,
#                  num_layers_color=3,
#                  hidden_dim_color=128,  #64
#                  bound=1,
#                  n_chunks=5,
#                  **kwargs
#                  ):
#         super().__init__(bound, **kwargs)

#         # sigma network
#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
#         self.geo_feat_dim = geo_feat_dim

#         per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))

#         self.encoder = tcnn.Encoding(
#             n_input_dims=3,
#             encoding_config={
#                 "otype": "HashGrid",
#                 "n_levels": 16,
#                 "n_features_per_level": 2,
#                 "log2_hashmap_size": 19,
#                 "base_resolution": 16,
#                 "per_level_scale": per_level_scale,
#             },
#         )

#         self.sigma_net = tcnn.Network(
#             n_input_dims=32,
#             n_output_dims=1 + self.geo_feat_dim,
#             network_config={
#                 "otype": "FullyFusedMLP",
#                 "activation": "ReLU",
#                 "output_activation": "None",
#                 "n_neurons": hidden_dim,
#                 "n_hidden_layers": num_layers - 1,
#             },
#         )

#         # color network
#         self.num_layers_color = 5#num_layers_color        
#         self.hidden_dim_color = 128#hidden_dim_color

#         self.encoder_dir = tcnn.Encoding(
#             n_input_dims=3,
#             encoding_config={
#                 "otype": "SphericalHarmonics",
#                 "degree": 4,
#             },
#         )

#         self.in_dim_color = self.encoder_dir.n_output_dims + self.geo_feat_dim

#         # chunk size for yolo object features
#         self.n_chunks = n_chunks

#         self.color_net = tcnn.Network(
#             n_input_dims=self.in_dim_color + self.n_chunks, # expand input dimensions to contain object features
#             n_output_dims=3,
#             network_config={
#                 "otype": "FullyFusedMLP",
#                 "activation": "ReLU",
#                 "output_activation": "None",
#                 "n_neurons": hidden_dim_color,
#                 "n_hidden_layers": num_layers_color - 1,
#             },
#         )

    
#     def forward(self, x, d,yolo_details=None):
#         # x: [N, 3], in [-bound, bound]
#         # d: [N, 3], nomalized in [-1, 1]


#         # sigma
#         x = (x + self.bound) / (2 * self.bound) # to [0, 1]
#         x = self.encoder(x)
        
#         h = self.sigma_net(x)

#         #sigma = F.relu(h[..., 0])
#         sigma = trunc_exp(h[..., 0])
#         geo_feat = h[..., 1:]
#         _, _, obj_feat = (yolo_details[0],yolo_details[1],yolo_details[2])
        
#         # color
#         d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
#         d = self.encoder_dir(d)

#         #p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        
#         h = torch.cat([d, geo_feat, obj_feat], dim=-1)
#         h = self.color_net(h)
        
#         # sigmoid activation for rgb
#         color = torch.sigmoid(h)

#         return sigma, color

#     def density(self, x, yolo_details=None):
#         # x: [N, 3], in [-bound, bound]
#         # if yolo_details is not None:
#         #     mask, _, _ = yolo_details
#         #     mask_resized = F.interpolate(mask.squeeze(0).reshape(640, 640).unsqueeze(0).unsqueeze(0).float(), size=(64, 64), mode='bilinear', align_corners=False)
#         #     mask_resized = mask_resized.flatten()
#         #     x = x.reshape(x.shape[0]//512, 512, -1)
#         #     x = x[(mask_resized > 0)]
#         x = (x + self.bound) / (2 * self.bound) # to [0, 1]

#         x = self.encoder(x)
        
#         h = self.sigma_net(x)

#         #sigma = F.relu(h[..., 0])
#         sigma = trunc_exp(h[..., 0])
#         geo_feat = h[..., 1:]
#         # breakpoint()
#         # _, _, obj_feat = (yolo_details[0],yolo_details[1],yolo_details[2])

#         return {
#             'sigma': sigma,
#             'geo_feat': geo_feat,
#         }

#     # allow masked inference
#     def color(self, x, d, yolo_details, mask=None, geo_feat=None, **kwargs):
#         # x: [N, 3] in [-bound, bound]
#         # mask: [N,], bool, indicates where we actually needs to compute rgb.
        
        
#         # retrieve only obj_feat from yolo_details which is mask, bbox, obj_feat
#         _, _, obj_feat = (yolo_details[0],yolo_details[1],yolo_details[2])
#         obj_feat = torch.tensor(obj_feat)

#         x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        
#         if mask is not None:
#             rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
#             # in case of empty mask
#             if not mask.any():
#                 return rgbs
#             x = x[mask]
#             d = d[mask]
#             geo_feat = geo_feat[mask]

#         # color
#         d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
#         d = self.encoder_dir(d)
        
#         if self.n_chunks != 0:

#             # chunk the obj_feat to specified number of chunks
#             divisor_chunks = obj_feat.shape[0] // self.n_chunks
#             remainder_chunks = obj_feat.shape[0] % self.n_chunks

#             chunked_obj_feat = torch.chunk(obj_feat[:divisor_chunks * self.n_chunks], self.n_chunks)

#             # take the mean of each chunk
#             avg_chunk_obj_feat = [_.mean() for _ in chunked_obj_feat]
#             avg_chunk_obj_feat = torch.tensor(avg_chunk_obj_feat)  

#             # expand to size N      
#             avg_chunk_obj_feat = avg_chunk_obj_feat.repeat(x.shape[0], 1).to(x.device) 
            
#             h = torch.cat([d, geo_feat, avg_chunk_obj_feat], dim=-1)
        
#         else:
#             h = torch.cat([d, geo_feat], dim=-1)

#         h = self.color_net(h)
        
#         # sigmoid activation for rgb
#         h = torch.sigmoid(h)
        
#         if mask is not None:
#             rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
#         else:
#             rgbs = h
        

#         return rgbs        

#     # optimizer utils
#     def get_params(self, lr):

#         params = [
#             {'params': self.encoder.parameters(), 'lr': lr},
#             {'params': self.sigma_net.parameters(), 'lr': lr},
#             {'params': self.encoder_dir.parameters(), 'lr': lr},
#             {'params': self.color_net.parameters(), 'lr': lr}, 
#         ]
#         if self.bg_radius > 0:
#             params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
#             params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
#         return params
