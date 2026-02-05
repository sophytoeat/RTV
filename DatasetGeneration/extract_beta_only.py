"""
Extract SMPL β parameters from video using BEV

This script extracts only β parameters without generating the full dataset.
No mask files are required.

Usage:
    python DatasetGeneration/extract_beta_only.py \
        --video_path ./video/f_fatsuit_gap.MP4 \
        --output_dir ./extracted_betas/f_fatsuit_gap \
        --max_frames 100
"""

import numpy as np
import cv2
import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

import torch
from util.multithread_video_loader import MultithreadVideoLoader
from SMPL.smpl_regressor import SMPL_Regressor
from tqdm import tqdm
import json
import argparse


def extract_beta_from_smpl_param(smpl_param):
    """
    Extract β parameters from SMPL regressor output.
    
    Args:
        smpl_param: SMPL parameter dictionary from BEV
    
    Returns:
        β parameters, shape (10,) or (11,)
    """
    if 'smpl_betas' in smpl_param:
        betas = smpl_param['smpl_betas']
        if isinstance(betas, torch.Tensor):
            betas = betas.cpu().numpy()
        # Handle batch dimension - take closest person
        if betas.ndim > 1:
            cam_trans = smpl_param['cam_trans']
            depth_order = torch.sort(cam_trans[:, 2].cpu(), descending=False).indices.numpy()
            betas = betas[depth_order[0]]
        return betas.flatten().astype(np.float32)
    
    return np.zeros(10, dtype=np.float32)


def extract_betas_from_video(video_path, output_dir, max_frames=None, skip_frames=1):
    """
    Extract β parameters from all frames of a video.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted β parameters
        max_frames: Maximum number of frames to process (None = all)
        skip_frames: Process every N-th frame
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load video
    print(f"Loading video: {video_path}")
    video_loader = MultithreadVideoLoader(video_path, max_height=3072)
    total_frames = len(video_loader)
    print(f"Total frames: {total_frames}")
    
    # Initialize SMPL regressor with fix_body=False to get real β
    print("Initializing BEV with fix_body=False...")
    smpl_regressor = SMPL_Regressor(use_bev=True, fix_body=False)
    print("BEV initialized.")
    
    # Calculate frames to process
    if max_frames is None:
        max_frames = total_frames
    frames_to_process = min(max_frames, total_frames // skip_frames)
    
    # Storage for all betas
    all_betas = {}
    beta_list = []
    
    frame_idx = 0
    processed = 0
    
    pbar = tqdm(total=frames_to_process, desc="Extracting β parameters")
    
    while processed < frames_to_process and frame_idx < total_frames:
        raw_image = video_loader.cap()
        
        if raw_image is None:
            break
        
        # Process every N-th frame
        if frame_idx % skip_frames == 0:
            try:
                # Run SMPL regression
                result = smpl_regressor.forward(raw_image, roi=True, size=1.45, roi_img_size=1024)
                
                if result is not None and len(result) >= 1:
                    smpl_param = result[0] if isinstance(result, tuple) else result
                    
                    if smpl_param is not None:
                        beta = extract_beta_from_smpl_param(smpl_param)
                        
                        # Save individual β file
                        frame_id = str(frame_idx).zfill(5)
                        beta_path = os.path.join(output_dir, f"{frame_id}_beta.npy")
                        np.save(beta_path, beta[:10])  # Save first 10 dimensions
                        
                        all_betas[frame_id] = beta[:10].tolist()
                        beta_list.append(beta[:10])
                        
                        processed += 1
                        pbar.update(1)
                        
                        # Print sample every 100 frames
                        if processed % 100 == 1:
                            print(f"\nFrame {frame_idx}: β = {beta[:10]}")
                    else:
                        print(f"\nFrame {frame_idx}: No SMPL detection")
                else:
                    print(f"\nFrame {frame_idx}: BEV returned None")
                    
            except Exception as e:
                print(f"\nFrame {frame_idx}: Error - {e}")
        
        frame_idx += 1
    
    pbar.close()
    video_loader.close()
    
    # Save all betas to JSON
    with open(os.path.join(output_dir, "beta_params.json"), "w") as f:
        json.dump(all_betas, f, indent=2)
    
    # Compute and save statistics
    if len(beta_list) > 0:
        beta_array = np.array(beta_list)
        stats = {
            "num_frames": len(beta_list),
            "mean": beta_array.mean(axis=0).tolist(),
            "std": beta_array.std(axis=0).tolist(),
            "min": beta_array.min(axis=0).tolist(),
            "max": beta_array.max(axis=0).tolist(),
        }
        
        with open(os.path.join(output_dir, "beta_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Processed frames: {len(beta_list)}")
        print(f"Output directory: {output_dir}")
        print("\nβ Parameter Statistics:")
        print(f"  Mean: {np.array(stats['mean'])}")
        print(f"  Std:  {np.array(stats['std'])}")
        print(f"  Min:  {np.array(stats['min'])}")
        print(f"  Max:  {np.array(stats['max'])}")
        print("=" * 60)
    else:
        print("No frames processed!")
    
    return all_betas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SMPL β parameters from video")
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to input video")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save extracted β parameters")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum frames to process (default: all)")
    parser.add_argument("--skip_frames", type=int, default=1,
                        help="Process every N-th frame (default: 1)")
    
    args = parser.parse_args()
    
    extract_betas_from_video(
        args.video_path,
        args.output_dir,
        args.max_frames,
        args.skip_frames
    )
