"""
Minimal VFID computation - memory efficient
"""
import cv2
import numpy as np
import torch
import argparse
import gc
import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('--pred', type=str, required=True)
    parser.add_argument('--max_frames', type=int, default=500)
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--clip_length', type=int, default=16)
    args = parser.parse_args()
    
    print(f"Settings: max_frames={args.max_frames}, resize={args.resize}, clip_length={args.clip_length}")
    sys.stdout.flush()
    
    # Load I3D model
    print("Loading I3D model...")
    from QuantitativeEvaluation.vfid_computer import VfidComputerI3D
    from QuantitativeEvaluation.fid_metrics import calculate_fid
    
    vfid_computer = VfidComputerI3D()
    vfid_computer.clip_length = args.clip_length
    print("I3D model loaded.")
    sys.stdout.flush()
    
    gt_cap = cv2.VideoCapture(args.gt)
    pred_cap = cv2.VideoCapture(args.pred)
    
    if not gt_cap.isOpened():
        print(f"Error: Cannot open {args.gt}")
        return
    if not pred_cap.isOpened():
        print(f"Error: Cannot open {args.pred}")
        return
    
    gt_total = int(gt_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pred_total = int(pred_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"GT frames: {gt_total}, Pred frames: {pred_total}")
    
    # Calculate number of clips
    frame_count = min(gt_total, pred_total, args.max_frames)
    num_clips = frame_count // args.clip_length
    print(f"Processing {num_clips} clips ({num_clips * args.clip_length} frames)")
    sys.stdout.flush()
    
    gt_features = []
    pred_features = []
    
    for clip_idx in range(num_clips):
        gt_clip = []
        pred_clip = []
        
        for _ in range(args.clip_length):
            ret1, f1 = gt_cap.read()
            ret2, f2 = pred_cap.read()
            
            if not ret1 or not ret2:
                break
            
            # Resize and convert BGR to RGB
            f1_resized = cv2.resize(f1, (args.resize, args.resize))
            f2_resized = cv2.resize(f2, (args.resize, args.resize))
            f1_rgb = cv2.cvtColor(f1_resized, cv2.COLOR_BGR2RGB)
            f2_rgb = cv2.cvtColor(f2_resized, cv2.COLOR_BGR2RGB)
            
            gt_clip.append(f1_rgb)
            pred_clip.append(f2_rgb)
        
        if len(gt_clip) == args.clip_length:
            gt_feat = vfid_computer.compute_feature(gt_clip).cpu().numpy()
            pred_feat = vfid_computer.compute_feature(pred_clip).cpu().numpy()
            gt_features.append(gt_feat)
            pred_features.append(pred_feat)
            
            if (clip_idx + 1) % 10 == 0 or clip_idx == 0:
                print(f"Processed clip {clip_idx + 1}/{num_clips}")
                sys.stdout.flush()
        
        del gt_clip, pred_clip
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    gt_cap.release()
    pred_cap.release()
    
    if len(gt_features) > 1:
        gt_features = np.concatenate(gt_features, axis=0)
        pred_features = np.concatenate(pred_features, axis=0)
        
        print(f"\nCalculating FID from {len(gt_features)} feature vectors...")
        vfid_score = calculate_fid(gt_features, pred_features)
        
        print(f"\n{'='*40}")
        print(f"RESULTS")
        print(f"{'='*40}")
        print(f"Processed clips: {len(gt_features)}")
        print(f"VFID: {vfid_score:.4f}")
        print(f"{'='*40}")
    else:
        print("Not enough clips processed!")

if __name__ == '__main__':
    main()
