import tqdm
import glob
import os
import torch
from nerf.network_tcnn import NeRFNetwork
from nerf.renderer import NeRFRenderer


class MONeRFNetwork(NeRFNetwork):
  def __init__(self, ckpt_list, model_class, 
              fp16=False, # amp optimize level
               nw_args=[], nw_kwargs={}):
    print("Initializing MONeRFNetwork")
    super().__init__(*nw_args, **nw_kwargs)
    self.ckpt_list = ckpt_list
    self.fp16 = fp16
    self.model_class = model_class
    self.nw_args = nw_args
    self.nw_kwargs = nw_kwargs
    self.device = None
    # checkpoint_dict = torch.load(ckpt_list[0])
    # self._load_checkpoint_dict_to_model(self, checkpoint_dict)

  def _load_checkpoint_dict_to_model(self, model, checkpoint_dict):
    if 'model' not in checkpoint_dict:
      model.load_state_dict(checkpoint_dict)
      print("[INFO] loaded model.")
      return

    model.load_state_dict(checkpoint_dict['model'], strict=False)
    if self.device is not None:
      model.to(self.device)

  def get_model_with_checkpoint(self, ckpt):
    checkpoint_dict = torch.load(ckpt, map_location=self.device)
    model = self.model_class(*self.nw_args, **self.nw_kwargs)
    return model
  
  def color_and_densities(self, *args, **kwargs):
      with torch.no_grad():
        return self._color_and_densities(*args, **kwargs)   

  def _color_and_densities(self, x, density_only=True, color_args=[], color_kwargs={}):
      #os.makedirs(f"{opt.workspace}/combined", exist_ok=True)
      max_densities = None
      max_density_color = None
      for ckpt in self.ckpt_list:
          model = self.get_model_with_checkpoint(ckpt)
          with torch.cuda.amp.autocast(enabled=self.fp16):
              densities = model.density(x)
              densities_sigma= densities['sigma']
              densities_geo_feat = densities['geo_feat']
              # breakpoint()
              if not density_only:
                rgbs = model.color(x, *color_args, **color_kwargs)
              if max_densities  is None:
                # max_densities = densities
                max_densities= densities_sigma
                max_geo_feat = densities_geo_feat
                if not density_only:
                  max_density_color = rgbs
              else:
                print("MAX density logic called")
                
                # max_geo_feat, max_indices_g = torch.max(
                #     torch.stack([densities_geo_feat,  max_geo_feat]), dim=0, keepdim=True)
                # max_geo_feat = max_geo_feat.squeeze(0)
                # breakpoint()
                
                max_densities, max_indices = torch.max(
                    torch.stack([densities_sigma, max_densities]), dim=0, keepdim=True)
                max_densities= max_densities.squeeze(0)


                max_geo_feat = torch.take_along_dim(
                      torch.stack([densities_geo_feat, max_geo_feat]),
                      max_indices.unsqueeze(-1), dim=0)
                max_geo_feat = max_geo_feat.squeeze(0)
                # breakpoint()
                # breakpoint()
                if not density_only:
                    # breakpoint()
                    max_density_color = torch.take_along_dim(
                      torch.stack([rgbs, max_density_color]),
                      max_indices.unsqueeze(-1), dim=0)
                    max_density_color = max_density_color.squeeze(0)
                    # breakpoint()
      torch.cuda.empty_cache()
      if not density_only:
        #return self.density(x, *density_args, **density_kwargs ), self.color(x, *color_args, **color_kwargs)
        return max_densities, max_density_color         
      else:
        #return self.density(x, *density_args, **density_kwargs )
        return max_densities, max_geo_feat
        breakpoint()
      
  def to(self, device=None):
     self.device = device
     super().to(device=device)

  def color(self, x, d, yolo_details, **kwargs):
      max_densities, max_density_color = self.color_and_densities(x, density_only=False,                                                                
                                                                  color_args=[d, yolo_details], 
                                                                  color_kwargs=kwargs)
      return max_density_color
  
  def density(self, x):
       max_densities, max_geo_feat = self.color_and_densities(x, density_only=True)  
       return {'sigma': max_densities, 
               'geo_feat' : max_geo_feat
       }