"""
Dataset Generation with SMPL β Parameters

This script extends the original dataset generation to also extract and save
SMPL β parameters for training the Extended Hybrid Representation.

The generated dataset will include:
    - {frame_id}_garment.jpg: Ground truth garment image
    - {frame_id}_vm.jpg: Virtual Measurement Garment (I_vm)
    - {frame_id}_mask.png: Garment alpha mask
    - {frame_id}_iuv.npy: Raw IUV from DensePose
    - {frame_id}_trans2roi.npy: Transformation matrix
    - {frame_id}_beta.npy: SMPL β parameters (NEW)
    - beta_params.json: All β parameters in JSON format (NEW)
"""

import numpy as np
import cv2
import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

import util.util
import util.util as util
import torch
from util.multithread_video_loader import MultithreadVideoLoader
import time
from SMPL.smpl_np_base import SMPLModel
from SMPL.smpl_regressor import SMPL_Regressor
from SMPL.pose_filter import PoseFilter, OfflineFilter
import glm
from OffscreenRenderer.flat_renderer import FlatRenderer
from SMPL.amputated_smpl import AmputatedSMPL
from SMPL.upperbody_smpl.UpperBody import UpperBodySMPL
from util.image_warp import zoom_in, zoom_out, shift_image_right, shift_image_down, rotate_image
from tqdm import tqdm
from model.DensePose.densepose_extractor import DensePoseExtractor
from util.densepose_util import IUV2UpperBodyImg, IUV2TorsoLeg
from util.beta_utils import estimate_beta_from_joints
import json
from DatasetGeneration.options import BaseOptions
from util.file_io import get_file_path_list


def make_video_loader(source_path):
    source_dataset_new = MultithreadVideoLoader(source_path, max_height=3072)
    print("Length of source dataset: %d" % len(source_dataset_new))
    return source_dataset_new


def extract_beta_from_smpl_param(smpl_param, mu_slim=0.27, mu_fat=0.34):
    """
    Extract β parameters from SMPL regressor output.
    
    Args:
        smpl_param: SMPL parameter dictionary
        mu_slim: Median ratio for slim body type
        mu_fat: Median ratio for fat body type
    
    Returns:
        β parameters, shape (10,)
    """
    import torch
    
    # Try to get β directly from SMPL parameters
    # Note: SMPL regressor uses 'smpl_betas' key, not 'betas'
    if 'smpl_betas' in smpl_param:
        betas = smpl_param['smpl_betas']
        if isinstance(betas, torch.Tensor):
            betas = betas.cpu().numpy()
        # Handle batch dimension
        if betas.ndim > 1:
            betas = betas[0]  # Take first person if batch
        return betas.flatten()[:10].astype(np.float32)
    
    # Fallback: try 'betas' key (for compatibility)
    if 'betas' in smpl_param:
        betas = smpl_param['betas']
        if isinstance(betas, torch.Tensor):
            betas = betas.cpu().numpy()
        if betas.ndim > 1:
            betas = betas[0]
        return betas.flatten()[:10].astype(np.float32)
    
    # Estimate from 2D joints (fallback method)
    if 'pj2d_org' in smpl_param:
        try:
            joints = smpl_param['pj2d_org']
            cam_trans = smpl_param['cam_trans']
            depth_order = torch.sort(cam_trans[:, 2].cpu(), descending=False).indices.numpy()
            J = joints[depth_order][0].cpu().numpy()
            return estimate_beta_from_joints(J, mu_slim, mu_fat)
        except Exception as e:
            print(f"Warning: Could not estimate β from joints: {e}")
    
    return np.zeros(10, dtype=np.float32)


def gen_dataset_with_beta(source_path, mask_dir, dataset_name, save_beta=True):
    """
    Generate dataset with β parameters.
    
    Args:
        source_path: Input video path
        mask_dir: Directory containing garment masks
        dataset_name: Output dataset name
        save_beta: Whether to save β parameters
    """
    video_loader = make_video_loader(source_path)
    # NOTE:
    # For β-dataset generation we want the regressor's estimated SMPL betas.
    # `fix_body=True` forcibly overwrites parts of `smpl_betas` in `SMPL/my_bev.py`,
    # which destroys variance and makes PCA compression impossible.
    smpl_regressor = SMPL_Regressor(use_bev=True, fix_body=False)
    densepose_extractor = DensePoseExtractor()
    mask_lists = get_file_path_list(mask_dir, 'png')
    
    assert len(mask_lists) == len(video_loader), \
        f"Number of masks ({len(mask_lists)}) and video frames ({len(video_loader)}) are inconsistent!"
    
    upper_body = UpperBodySMPL()
    target_path = os.path.join('./PerGarmentDatasets', dataset_name)
    os.makedirs(target_path, exist_ok=True)
    resolution = 1024
    
    height = None
    width = None
    
    # Dictionary to store all β parameters
    beta_dict = {}
    
    for i in tqdm(range(len(video_loader)), desc="Generating dataset"):
        raw_image = video_loader.cap()
        raw_mask_path = mask_lists[i]
        
        if raw_image is None:
            break
        
        height = raw_image.shape[0]
        width = raw_image.shape[1]
        new_height = 1024
        new_width = new_height * width // height
        resized_image = cv2.resize(raw_image, (new_width, new_height))
        
        # Run SMPL regression
        result = smpl_regressor.forward(raw_image, True, size=1.45, roi_img_size=resolution)
        
        if result is None or len(result) < 3:
            continue
        
        smpl_param, trans2roi, inv_trans2roi = result
        
        if smpl_param is None:
            continue
        
        vertices = smpl_regressor.get_raw_verts(smpl_param)
        vertices = torch.from_numpy(vertices).unsqueeze(0)
        v = vertices
        
        # Render Virtual Measurement Garment
        raw_vm = upper_body.render(v[0], height=new_height, width=new_width)
        
        # Get DensePose IUV
        raw_IUV = densepose_extractor.get_IUV(resized_image, isRGB=False)
        if raw_IUV is None:
            continue
        
        # Load and resize mask (robust to 1ch/3ch/4ch)
        raw_mask_img = cv2.imread(raw_mask_path, cv2.IMREAD_UNCHANGED)
        if raw_mask_img is None:
            print(f"Warning: cannot read mask {raw_mask_path}, skip frame {i}")
            continue
        if raw_mask_img.ndim == 2:
            raw_mask = raw_mask_img
        elif raw_mask_img.shape[2] == 4:
            raw_mask = raw_mask_img[:, :, 3]
        else:
            raw_mask = cv2.cvtColor(raw_mask_img, cv2.COLOR_BGR2GRAY)
        raw_mask = cv2.resize(raw_mask, (width, height))
        
        # Create garment image
        raw_garment = raw_image.copy()
        raw_garment[raw_mask < 127] = 0
        roi_garment_img = cv2.warpAffine(raw_garment, trans2roi, (resolution, resolution),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0))
        
        # Extract β parameters
        beta = extract_beta_from_smpl_param(smpl_param)
        
        # Frame ID
        frame_id = str(i).zfill(5)
        
        # Save files
        garment_path = os.path.join(target_path, frame_id + '_garment.jpg')
        vm_path = os.path.join(target_path, frame_id + '_vm.jpg')
        mask_path = os.path.join(target_path, frame_id + '_mask.png')
        iuv_path = os.path.join(target_path, frame_id + '_iuv.npy')
        trans2roi_path = os.path.join(target_path, frame_id + '_trans2roi.npy')
        
        cv2.imwrite(garment_path, roi_garment_img)
        cv2.imwrite(vm_path, raw_vm)
        cv2.imwrite(mask_path, raw_mask)
        np.save(iuv_path, raw_IUV)
        np.save(trans2roi_path, trans2roi)
        
        # Save β parameters
        if save_beta:
            beta_path = os.path.join(target_path, frame_id + '_beta.npy')
            np.save(beta_path, beta)
            beta_dict[frame_id] = beta.tolist()
    
    # Save dataset info
    dataset_info = {
        "height": height,
        "width": width,
        "has_beta": save_beta,
        "num_frames": len(beta_dict)
    }
    
    with open(os.path.join(target_path, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    # Save all β parameters to a single JSON file
    if save_beta:
        with open(os.path.join(target_path, "beta_params.json"), "w") as f:
            json.dump(beta_dict, f, indent=2)
        print(f"Saved β parameters for {len(beta_dict)} frames")
    
    video_loader.close()
    print(f"Dataset generated: {target_path}")


def process_video(v_path, mask_dir, dataset_name, save_beta=True):
    """Main entry point for dataset generation."""
    gen_dataset_with_beta(v_path, mask_dir, dataset_name, save_beta=save_beta)


if __name__ == '__main__':
    opts = BaseOptions()
    opt = opts.parse()
    video_path = opt.video_path
    mask_dir = opt.mask_dir
    dataset_name = opt.dataset_name
    
    # Add save_beta option
    save_beta = getattr(opt, 'save_beta', True)
    
    process_video(video_path, mask_dir, dataset_name=dataset_name, save_beta=save_beta)
