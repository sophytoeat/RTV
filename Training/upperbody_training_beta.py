"""
Training script for Extended Hybrid Representation with SMPL β Parameters

This script trains the Garment Synthesis (GS) network using:
    I_hybrid' = I_vm ⊕ I_sdp ⊕ I_β

Input channels: 7 (3 + 3 + 1)  # β[0] only
Output channels: 4 (RGB + Alpha)
"""

import os
import cv2
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from collections import OrderedDict

from Datasets.upperbody_garment.upperbody_garment_beta import UpperBodyGarmentBeta, UpperBodyGarmentBetaHybrid
from options.train_options import TrainOptions
from model.pix2pixHD.models import create_model
import util.util as util
from util.visualizer import Visualizer
import torchvision
import torch
import math
import argparse

def lcm(a, b): 
    return abs(a * b) // math.gcd(a, b) if a and b else 0

import time


def main():
    # Parse options
    opt = TrainOptions().parse()
    
    # Override for β[0]-only Hybrid Representation
    # I_vm (3) + I_sdp (3) + I_β0 (1) = 7 channels
    opt.input_nc = 7
    opt.output_nc = 4  # RGB + Alpha
    opt.model = 'pix2pixHD_RGBA'  # Use RGBA model for 4-channel output
    
    print("=" * 50)
    print("β[0]-only Hybrid Representation Training")
    print(f"Input channels: {opt.input_nc} (I_vm: 3, I_sdp: 3, I_β0: 1)")
    print(f"Output channels: {opt.output_nc}")
    print("=" * 50)
    
    if opt.dataset_path is not None:
        dataset_paths = opt.dataset_path
        path_list = dataset_paths.split(',')
        
        # Use the new dataset with β parameters
        dataset = UpperBodyGarmentBeta(path_list[0], img_size=opt.img_size, use_random_beta=True)
        
        if len(path_list) > 1:
            for i in range(1, len(path_list)):
                dataset = dataset + UpperBodyGarmentBeta(path_list[i], img_size=opt.img_size, use_random_beta=True)
    else:
        print("Please specify a dataset for training!")
        exit(0)
    
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size}")
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=opt.batchSize, 
        shuffle=True, 
        sampler=None, 
        batch_sampler=None, 
        num_workers=4, 
        collate_fn=None, 
        pin_memory=True, 
        drop_last=True, 
        timeout=0, 
        worker_init_fn=None
    )
    
    # Create model with extended input channels
    model = create_model(opt)
    visualizer = Visualizer(opt)
    
    # Load checkpoint if resuming
    start_epoch, epoch_iter = 1, 0
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    if opt.continue_train and os.path.exists(iter_path):
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        print(f'Resuming from epoch {start_epoch} at iteration {epoch_iter}')
    
    opt.print_freq = lcm(opt.print_freq, opt.batchSize)
    
    # Get optimizers
    optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D
    
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq
    
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        
        for i, data in enumerate(dataloader):
            # Unpack data: garment_img, vm_img, dp_img, beta_img, garment_mask
            garment_img, vm_img, dp_img, beta_img, garment_mask = data
            
            # Move to CUDA
            garment_img = garment_img.cuda()
            vm_img = vm_img.cuda()
            dp_img = dp_img.cuda()
            beta_img = beta_img.cuda()
            garment_mask = garment_mask.cuda()
            
            if total_steps % opt.print_freq == print_delta:
                iter_start_time = time.time()
            
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            
            # Create β[0]-only Hybrid Representation
            # I_hybrid' = I_vm ⊕ I_sdp ⊕ I_β0
            input_img = torch.cat([vm_img, dp_img, beta_img], dim=1)  # (B, 7, H, W)
            
            # Ground truth: RGB + Alpha
            gt_image = torch.cat([garment_img, garment_mask], dim=1)  # (B, 4, H, W)
            
            # Whether to collect output images
            save_fake = total_steps % opt.display_freq == display_delta
            
            # Forward pass
            losses, generated = model(input_img, gt_image, infer=save_fake)
            
            # Sum per device losses
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model.module.loss_names, losses))
            
            # Calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)
            
            ############### Backward Pass ####################
            # Update generator weights
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            
            # Update discriminator weights
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            
            ############## Display results and errors ##########
            ### Print out errors
            if total_steps % opt.print_freq == print_delta:
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.print_freq
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)
            
            ### Display output images
            if save_fake:
                # Visualize inputs and outputs
                real_list = [('garment_img', util.tensor2im((garment_img / 2.0 + 0.5)[0], rgb=True))]
                fake_list = [('fake_img', util.tensor2im((generated.data[:, [0, 1, 2], :, :] / 2.0 + 0.5)[0], rgb=True))]
                fake2_list = [('fake_mask', util.tensor2im((generated.data[:, [3, 3, 3], :, :])[0], rgb=True))]
                vm_list = [('vm_image', util.tensor2im((vm_img / 2.0 + 0.5)[0], rgb=True))]
                dp_list = [('dp_image', util.tensor2im((dp_img / 2.0 + 0.5)[0], rgb=True))]
                
                # Visualize β[0] map (repeat to RGB for visualization)
                beta_vis = beta_img[:, :1, :, :] / 2.0 + 0.5
                beta_vis = beta_vis.repeat(1, 3, 1, 1)
                beta_list = [('beta0_map', util.tensor2im(beta_vis[0], rgb=True))]
                
                visuals = OrderedDict(real_list + vm_list + dp_list + beta_list + fake_list + fake2_list)
                visualizer.display_current_results(visuals, epoch, total_steps)
            
            ### Save latest model
            if total_steps % opt.save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.module.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
            
            if epoch_iter >= dataset_size:
                break
        
        # End of epoch
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        
        ### Save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.module.save('latest')
            model.module.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')
        
        ### Update fixed params if needed
        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            model.module.update_fixed_params()
        
        ### Linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            model.module.update_learning_rate()


if __name__ == "__main__":
    main()
