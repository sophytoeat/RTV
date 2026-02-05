"""
Video Metrics Computation Script

Computes SSIM, LPIPS, and VFID between two videos.

Usage:
    python QuantitativeEvaluation/compute_video_metrics.py \
        --gt_video <ground_truth_video_path> \
        --pred_video <predicted_video_path>
"""

import argparse
import cv2
import numpy as np
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from QuantitativeEvaluation.quantitative_evaluation import SsimComputer, LpipsComputer
from QuantitativeEvaluation.vfid_computer import VfidComputerI3D


def load_video_frames(video_path, max_frames=None):
    """Load all frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    
    print(f"Loading {frame_count} frames from {video_path}...")
    
    for _ in tqdm(range(frame_count), desc="Loading frames"):
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    return frames


def resize_frames_to_match(frames1, frames2):
    """Resize frames to match dimensions."""
    if len(frames1) == 0 or len(frames2) == 0:
        raise ValueError("Empty frame list")
    
    h1, w1 = frames1[0].shape[:2]
    h2, w2 = frames2[0].shape[:2]
    
    if h1 != h2 or w1 != w2:
        print(f"Resizing frames from {w2}x{h2} to {w1}x{h1}...")
        frames2 = [cv2.resize(f, (w1, h1)) for f in tqdm(frames2, desc="Resizing")]
    
    return frames1, frames2


def main():
    parser = argparse.ArgumentParser(description='Compute video metrics (SSIM, LPIPS, VFID)')
    parser.add_argument('--gt_video', type=str, required=True, help='Ground truth video path')
    parser.add_argument('--pred_video', type=str, required=True, help='Predicted video path')
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum frames to process')
    parser.add_argument('--skip_vfid', action='store_true', help='Skip VFID computation')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.gt_video):
        print(f"Error: Ground truth video not found: {args.gt_video}")
        return
    if not os.path.exists(args.pred_video):
        print(f"Error: Predicted video not found: {args.pred_video}")
        return
    
    # Load video frames
    gt_frames = load_video_frames(args.gt_video, args.max_frames)
    pred_frames = load_video_frames(args.pred_video, args.max_frames)
    
    # Ensure same number of frames
    min_frames = min(len(gt_frames), len(pred_frames))
    if len(gt_frames) != len(pred_frames):
        print(f"Warning: Frame count mismatch (GT: {len(gt_frames)}, Pred: {len(pred_frames)})")
        print(f"Using first {min_frames} frames from each video.")
        gt_frames = gt_frames[:min_frames]
        pred_frames = pred_frames[:min_frames]
    
    # Resize if needed
    gt_frames, pred_frames = resize_frames_to_match(gt_frames, pred_frames)
    
    print(f"\nComputing metrics for {len(gt_frames)} frames...")
    print("=" * 50)
    
    # Compute SSIM
    print("\n[1/3] Computing SSIM...")
    ssim_computer = SsimComputer()
    ssim_score = ssim_computer(gt_frames, pred_frames)
    print(f"SSIM: {ssim_score:.4f}")
    
    # Compute LPIPS
    print("\n[2/3] Computing LPIPS...")
    lpips_computer = LpipsComputer()
    lpips_score = lpips_computer(gt_frames, pred_frames)
    print(f"LPIPS: {lpips_score:.4f}")
    
    # Compute VFID
    if not args.skip_vfid:
        print("\n[3/3] Computing VFID (I3D)...")
        vfid_computer = VfidComputerI3D()
        
        # VFID requires frames in multiples of clip_length (10)
        clip_length = vfid_computer.clip_length
        usable_frames = (len(gt_frames) // clip_length) * clip_length
        
        if usable_frames < clip_length:
            print(f"Warning: Not enough frames for VFID (need at least {clip_length})")
            vfid_score = float('nan')
        else:
            gt_frames_vfid = gt_frames[:usable_frames]
            pred_frames_vfid = pred_frames[:usable_frames]
            vfid_score = vfid_computer(gt_frames_vfid, pred_frames_vfid)
        print(f"VFID: {vfid_score:.4f}")
    else:
        vfid_score = None
        print("\n[3/3] Skipping VFID computation")
    
    # Summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"Ground Truth: {args.gt_video}")
    print(f"Predicted:    {args.pred_video}")
    print(f"Frames:       {len(gt_frames)}")
    print("-" * 50)
    print(f"SSIM:  {ssim_score:.4f}  (higher is better, max=1.0)")
    print(f"LPIPS: {lpips_score:.4f}  (lower is better, min=0.0)")
    if vfid_score is not None:
        print(f"VFID:  {vfid_score:.4f}  (lower is better)")
    print("=" * 50)


if __name__ == '__main__':
    main()
