import torch
import argparse

from functools import partial
from loss import huber_loss

from ultralytics import YOLO
import json
import numpy as np

import time

#torch.autograd.set_detect_anomaly(True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
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
    parser.add_argument('--edit_x', type=float, default=0, help="edit x coordinate of camera location")

    parser.add_argument('--save_gt', action='store_true', help="save ground truth images")

    ### yolo
    parser.add_argument('--yolo_model', type=str, default='yolov8l-seg.pt', help='Ultralytics YOLO segmentation model size to use')
    parser.add_argument('--post', action='store_true', help="combine depths to regenerate")
    parser.add_argument('--legacy', action='store_true', help='use default torch-ngp')
    parser.add_argument('--n_chunks', type=int, default=5, help='number of chunks for obj_feat, use 0 to turn off obj_feats')
    # parser.add_argument('--bound_inf', nargs='+', type=float, help='xmin, ymin, zmin, xmax, ymax, zmax')

    opt = parser.parse_args()

    if not opt.legacy:

        """"
        Object-Wise Scene Reconstruction
        """
        
        from nerf.provider import NeRFDataset
        from nerf.gui import NeRFGUI
        from nerf.utils import *


        if opt.O:
            opt.fp16 = True
            opt.cuda_ray = True
            opt.preload = True
    
        if opt.patch_size > 1:
            opt.error_map = False # do not use error_map if use patch-based training
            # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
            assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."


        if opt.ff:
            opt.fp16 = True
            assert opt.bg_radius <= 0, "background model is not implemented for --ff"
            from nerf.network_ff import NeRFNetwork
        elif opt.tcnn:
            opt.fp16 = True
            assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
            from nerf.network_tcnn import NeRFNetwork
        else:
            from nerf.network import NeRFNetwork
        
        seed_everything(opt.seed)

        model = NeRFNetwork(
            encoding="hashgrid",
            bound=opt.bound,
            cuda_ray=opt.cuda_ray,
            density_scale=1,
            min_near=opt.min_near,
            density_thresh=opt.density_thresh,
            bg_radius=opt.bg_radius,
            n_chunks=opt.n_chunks,
            # bound_inf=opt.bound_inf,
        )
        
        print(opt.test)


        def write_timing_files(opt, detected_object, train_duration, load_2_start, load_2_end):
            """
            Safely handle reading and writing of timing files with error checking.
            
            Args:
                opt: Options object containing workspace path
                detected_object: Name of the detected object
                train_duration: Duration of training
                load_2_start: Start time of loading phase 2
                load_2_end: End time of loading phase 2
            """
            # Initialize timing strings
            eval_timing_str = ""
            train_timing_str = ""
            loading_timing_str = ""
            
            # Try to read evaluation timing, handle if file doesn't exist
            try:
                render_time_path = f'{opt.workspace}_{detected_object}/render_times.txt'
                with open(render_time_path, 'r') as f:
                    eval_duration = f.read()
                    eval_timing_str += f'\n{detected_object}\n{eval_duration}\n'
            except FileNotFoundError:
                print(f"Warning: Render times file not found for {detected_object}")
                eval_timing_str += f'\n{detected_object}\nNo render timing data available\n'
            
            # Add train and loading timing information
            train_timing_str += f'\n{detected_object} - {float(train_duration):.5f}s'
            loading_timing_str += f'\n{detected_object} = {float(load_2_end-load_2_start):.5f}s'
            
            # Ensure results directory exists
            results_dir = f'{opt.workspace}/results'
            import os
            os.makedirs(results_dir, exist_ok=True)
            
           

        criterion = torch.nn.MSELoss(reduction='none')
        #criterion = partial(huber_loss, reduction='none')
        #criterion = torch.nn.HuberLoss(reduction='none', beta=0.1) # only available after torch 1.10 ?

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if opt.test:
            
            metrics = [PSNRMeter(), LPIPSMeter(device=device), SSIMMeter()]
            
            if opt.gui:
                pass
                #gui = NeRFGUI(opt, trainer)
                #gui.render()
            
            else:
                all_images_dict_test, detected_objects_test = get_object_masks(opt.path, type='test', model=opt.yolo_model)
                
                for detected_object in detected_objects_test:
                    opt.detected_object = detected_object
                    mask_details_test = [(d['f_path'], item['merged_mask']) for d in all_images_dict_test if 'obj_dict' in d for item in d['obj_dict'] if item['object_type'] == opt.detected_object]

                    trainer = Trainer('ngp', opt, model, device=device, workspace=f'{opt.workspace}_{opt.detected_object}', criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt,eval_interval=50)

                    test_dataset = NeRFDataset(opt, mask_details_test, device=device, type='test')
                    test_loader = test_dataset.dataloader()
                    if test_loader.has_gt:
                        trainer.evaluate(test_loader) # blender has gt, so evaluate it.
                        
                    trainer.test(test_loader, write_video=True) # test and save video 
                    trainer.save_mesh(resolution=256, threshold=10)
        
        else:
             ##################using depth##########################################
            # if not os.path.exists(f'{opt.workspace}/validation'): 
            #   os.makedirs(f'{opt.workspace}/validation')
            # if not os.path.exists(f'{opt.workspace}/densities'): 
            #   os.makedirs(f'{opt.workspace}/densities')




            if not os.path.exists(f'{opt.workspace}/results'): 
              os.makedirs(f'{opt.workspace}/results')

            optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

            # load appropriate masks for train, val and test 
            all_images_dict_train, detected_objects_train = get_object_masks(opt.path, type='train', model=opt.yolo_model)
            all_images_dict_val, dect_val = get_object_masks(opt.path, type='val', model=opt.yolo_model)
            load_1_start = time.time()
            all_images_dict_test, detected_objects_test = get_object_masks(opt.path, type='test', model=opt.yolo_model, disable_yolo=True)
            load_1_end = time.time()
           
        
            ##########################using depth##########################################
            # objects_depths = {}
            # objects_colors = {}
            # objects_densities = {}
            # post = opt.post # perform post-training evaluations

            eval_timing_str = '''
            Objects render times
            ----------------------
            '''
            train_timing_str = '''
            Objects train times
            ----------------------
            '''
            loading_timing_str = '''
            Loading times
            ----------------------
            '''

            loading_timing_str += f'\nStage 1 --> YOLO Preprocess = {float(load_1_end - load_1_start):.5f}s\n'
            loading_timing_str += f'\nStage 2 --> Dataloader'

            for detected_object in detected_objects_train:
                opt.detected_object = detected_object
                
                # retrieve mask details for each object type in each image 
                mask_details_train = [(d['f_path'], item['merged_mask']) for d in all_images_dict_train if 'obj_dict' in d for item in d['obj_dict'] if item['object_type'] == opt.detected_object]

                train_dataset = NeRFDataset(opt, mask_details_train, device=device, type='train')

                
                if not os.path.exists(os.path.join(opt.workspace, 'obj_feats')):
                    os.makedirs(os.path.join(opt.workspace, 'obj_feats'))
                
                # save train dataset
                torch.save(
                    {
                        "yolo_details": train_dataset.yolo_details,
                        "poses": train_dataset.poses
                    }, 
                  f"{os.path.join(opt.workspace, 'obj_feats')}/{opt.detected_object}.pt")

                
                train_loader = train_dataset.dataloader()
                scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
                metrics = [PSNRMeter(), LPIPSMeter(device=device), SSIMMeter()]

                trainer = Trainer('ngp', opt, model, device=device, workspace=f'{opt.workspace}_{opt.detected_object}', optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=50, yolo_feats_encoder_dim=train_loader.yolo_feats_encoder_dim)
                if opt.gui:
                    gui = NeRFGUI(opt, trainer, train_loader)
                    gui.render()
                else:
                    # retrieve mask details for each object type in each image in val set
                    mask_details_val = [(d['f_path'], item['merged_mask']) for d in all_images_dict_val if 'obj_dict' in d for item in d['obj_dict'] if item['object_type'] == opt.detected_object]
                    valid_loader = NeRFDataset(opt, mask_details_val, device=device, type='val', downscale=1).dataloader()
                    max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
                    
                    train_start = time.time()
                    trainer.train(train_loader, valid_loader, max_epoch)
                    train_end = time.time()
                    train_duration = train_end - train_start
                    
                    load_2_start = time.time()
                    
                    
                    mask_details_test= []
                    test_dataset = NeRFDataset(opt, mask_details_test, device=device, type='test')
                    test_dataset = calculate_feats(test_dataset, torch.load(f"{os.path.join(opt.workspace, 'obj_feats')}/{opt.detected_object}.pt"))

                   

                    test_loader = test_dataset.dataloader()
                    load_2_end = time.time()

                    if test_loader.has_gt:
                            trainer.evaluate(test_loader) # blender has gt, so evaluate it.
                    
                    trainer.save_mesh(resolution=256, threshold=10)

                
                try:
                    write_timing_files(
                        opt=opt,
                        detected_object=detected_object,
                        train_duration=train_duration,
                        load_2_start=load_2_start,
                        load_2_end=load_2_end
                    )
                    with open(f'{opt.workspace}/results/train_times.txt', "w") as file:
                        file.write(train_timing_str)
                    with open(f'{opt.workspace}/results/render_times.txt', "w") as file:
                        file.write(eval_timing_str)
                except Exception as e:
                    print(f"Error processing timing files: {e}")
                # with open(f'{opt.workspace}/results/render_times.txt', "w") as file:
                #     file.write(eval_timing_str)

                # with open(f'{opt.workspace}/results/train_times.txt', "w") as file:
                #     file.write(train_timing_str)

                # with open(f'{opt.workspace}/results/loading_times.txt', "w") as file:
                #     file.write(loading_timing_str)
                

            ###################################USING DEPTH#####################################################################################    
            # if post: 
            #     # get np array from dict
            #     # stacked_depths = np.stack(objects_depths[obj] for obj in objects_depths.keys())
            #     # stacked_colors = np.stack(objects_colors[obj] for obj in objects_colors.keys())

            #     stacked_depths = np.array([objects_depths[obj] for obj in objects_depths.keys()])
            #     stacked_colors = np.array([objects_colors[obj] for obj in objects_colors.keys()])

            #     # get index (in other words "object") with the highest depth value
            #     indexes = np.argmax(stacked_depths, axis = 0)
            #     n_images, height, width = indexes.shape

            #     # black canvas placeholder for result
            #     colors = np.zeros((n_images, height, width, 3))
                
            #     post_lpips = LPIPSMeter(device = 'cuda')
            #     post_psnr = PSNRMeter()
            #     post_ssim = SSIMMeter()
                
            #     # retrieve mask details for dataset with combined object masks
            #     full_mask_details = get_object_masks(opt.path, type='test', use_full_mask = True, model=opt.yolo_model)
            #     full_mask_dataset = NeRFDataset(opt, full_mask_details, device=device, type='test')

            #     opt.detected_object = 'all'
            #     full_mask_dataloader = full_mask_dataset.dataloader()
            #     full_mask_dataloader.detected_object = 'all'
            #     trainer.evaluate(full_mask_dataloader, post = True)
                
            #     # eval_timing_str += f'\n all_objects - {float(eval_duration):.5f}s'

            #     objects_densities['all_objects'] = trainer.densities

            #     # loop through all pixels in each image in the current set
            #     for i in tqdm.tqdm(range(n_images), desc='Generating final images:'):
            #         for j in range(height):
            #             for k in range(width):

            #                 # select the color of the object with higest value depth
            #                 index = indexes[i, j, k] 
            #                 # write retrieved index's color to plain canvas
            #                 colors[i, j, k] = stacked_colors[index, i, j, k]
                    
            #         post_lpips.update(preds = torch.tensor(colors[i]).to(torch.float32).unsqueeze(0), truths = full_mask_dataset.images[i].unsqueeze(0))
                    
            #         # write output image
            #         cv2.imwrite(f'{opt.workspace}/validation/ngp_ep{str(max_epoch).zfill(4)}_{str(i).zfill(4)}.png', cv2.cvtColor(colors[i].astype(np.uint8), cv2.COLOR_BGR2RGB))

            #     # with open(f'{opt.workspace}/densities/object_densities.json', "w") as json_file:
            #         # json.dump(objects_densities, json_file)
                
            #     post_psnr.update(preds = colors/colors.max(), truths = full_mask_dataset.images)
            #     post_ssim.update(preds = torch.tensor(colors/colors.max()), truths = full_mask_dataset.images)
                
            #     print('PSNR: ', post_psnr.measure())
            #     print('MSE: ', post_psnr.measure_mse())
            #     print('LPIPS: ', post_lpips.measure())
            #     print('SSIM: ', post_ssim.measure().item())
            #     #trainer.test(test_loader, write_video=True) # test and save video
            #     #trainer.save_mesh(resolution=256, threshold=10)
                    
            
            # # decay to 0.1 * init_lr at last iter step   
    else:
        print('[INFO] Using pure torch-ngp')

        from legacy.nerf.provider import NeRFDataset
        from legacy.nerf.gui import NeRFGUI
        from legacy.nerf.utils import *

        if opt.O:
            opt.fp16 = True
            opt.cuda_ray = True
            opt.preload = True
    
        if opt.patch_size > 1:
            opt.error_map = False # do not use error_map if use patch-based training
            # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
            assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."


        if opt.ff:
            opt.fp16 = True
            assert opt.bg_radius <= 0, "background model is not implemented for --ff"
            from legacy.nerf.network_ff import NeRFNetwork
        elif opt.tcnn:
            opt.fp16 = True
            assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
            from legacy.nerf.network_tcnn import NeRFNetwork
        else:
            from legacy.nerf.network import NeRFNetwork

        print(opt)
        
        seed_everything(opt.seed)

        model = NeRFNetwork(
            encoding="hashgrid",
            bound=opt.bound,
            cuda_ray=opt.cuda_ray,
            density_scale=1,
            min_near=opt.min_near,
            density_thresh=opt.density_thresh,
            bg_radius=opt.bg_radius,
            # bound_inf=opt.bound_inf,
        )
        
        print(model)

        criterion = torch.nn.MSELoss(reduction='none')
        #criterion = partial(huber_loss, reduction='none')
        #criterion = torch.nn.HuberLoss(reduction='none', beta=0.1) # only available after torch 1.10 ?

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        timing_str = '''
        Torch-NGP Timing
        ----------------------
        '''
        
        if opt.test:
            
            metrics = [PSNRMeter(), LPIPSMeter(device=device), SSIMMeter()]
            trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)

            if opt.gui:
                gui = NeRFGUI(opt, trainer)
                gui.render()
            
            else:
                test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

                if test_loader.has_gt:
                    eval_start = time.time()
                    trainer.evaluate(test_loader) # blender has gt, so evaluate it.
                    eval_end = time.time()

                    eval_duration = eval_end - eval_start

                    timing_str += f'Total rendering time - {float(eval_duration):.5f}'
        
                trainer.test(test_loader, write_video=False) # test and save video
                
                trainer.save_mesh(resolution=256, threshold=10)
        
        else:

            optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

            train_loader = NeRFDataset(opt, device=device, type='train').dataloader()
            
            # decay to 0.1 * init_lr at last iter step
            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

            metrics = [PSNRMeter(), LPIPSMeter(device=device), SSIMMeter()]
            trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=50)

            if opt.gui:
                gui = NeRFGUI(opt, trainer, train_loader)
                gui.render()
            
            else:
                valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()

                max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
                
                train_start = time.time()
                trainer.train(train_loader, valid_loader, max_epoch)
                
                train_end = time.time()
                train_duration = train_end - train_start

                timing_str += f'\n Total training time - {float(train_duration):.5f}'

                # also test
                test_loader = NeRFDataset(opt, device=device, type='test').dataloader()
                
                if test_loader.has_gt:
                    eval_start = time.time()
                    trainer.evaluate(test_loader) # blender has gt, so evaluate it.
                    eval_end = time.time()

                    eval_duration = eval_end - eval_start

                    timing_str += f'\n Total rendering time - {float(eval_duration):5f}'

                trainer.test(test_loader, write_video=False) # test and save video
                
                trainer.save_mesh(resolution=256, threshold=10)
        with open(f'{opt.workspace}/timing.txt', "w") as file:
            file.write(timing_str)



