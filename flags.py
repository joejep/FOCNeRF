import argparse


def set_flags():

  parser = argparse.ArgumentParser()
  parser.add_argument('path', type=str)
  parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
  parser.add_argument('--test', action='store_true', help="test mode")
  parser.add_argument('--mo-density-infer', action='store_true', help="Multi obect density inference")
  parser.add_argument('--workspace', type=str, default='workspace')
  parser.add_argument('--seed', type=int, default=0)
  ### training options
  parser.add_argument('--iters', type=int, default=30000, help="training iters")
  parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate") # 1e-3
  parser.add_argument('--ckpt', type=str, default='latest')
  parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
  parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
  parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
  parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
  parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
  parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
  parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
  parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

  ### network backbone options
  parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
  parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
  parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

  ### dataset options
  parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
  parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
  # (the default value is for the fox dataset)
  parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.") #2
  parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
  parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location") # [0,0,0]
  parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
  parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")  #0.2,0.8
  parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
  parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

  ### GUI options
  parser.add_argument('--gui', action='store_true', help="start a GUI")
  parser.add_argument('--W', type=int, default=1920, help="GUI width")
  parser.add_argument('--H', type=int, default=1080, help="GUI height")
  parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
  parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
  parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

  ### experimental
  parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
  parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
  parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

  ### yolo
  parser.add_argument('--yolo_model', type=str, default='yolov8s-seg.pt', help='Ultralytics YOLO segmentation model size to use')
  parser.add_argument('--post', action='store_true', help="combine depths to regenerate")
  parser.add_argument('--legacy', action='store_true', help='use default torch-ngp')
  parser.add_argument('--n_chunks', type=int, default=5, help='number of chunks for obj_feat, use 0 to turn off obj_feats')
  parser.add_argument('--bound_inf', nargs='+', type=float, help='xmin, ymin, zmin, xmax, ymax, zmax')
  parser.add_argument('--edit_x', type=float, default=0, help="edit x coordinate of camera location")
  parser.add_argument('--ckpt_dir', type=str, default='', help="directory to load obj_feats and trained checkpoints from")
  parser.add_argument('--objects_of_interest', nargs='+', type=str, help='List of objects to focus on', default=[])
  return parser