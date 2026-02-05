"""
Minimal SSIM computation - extremely memory efficient
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import argparse
import gc
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('--pred', type=str, required=True)
    parser.add_argument('--max_frames', type=int, default=50)
    parser.add_argument('--resize', type=int, default=256, help='Resize frames to this size')
    parser.add_argument('--skip', type=int, default=10, help='Process every N-th frame')
    args = parser.parse_args()
    
    print(f"Settings: max_frames={args.max_frames}, resize={args.resize}, skip={args.skip}")
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
    sys.stdout.flush()
    
    scores = []
    frame_idx = 0
    processed = 0
    
    while processed < args.max_frames:
        ret1, f1 = gt_cap.read()
        ret2, f2 = pred_cap.read()
        
        if not ret1 or not ret2:
            print(f"End of video at frame {frame_idx}")
            break
        
        # Only process every N-th frame
        if frame_idx % args.skip == 0:
            # Resize to reduce memory and computation
            f1_small = cv2.resize(f1, (args.resize, args.resize))
            f2_small = cv2.resize(f2, (args.resize, args.resize))
            
            # Compute SSIM
            score = ssim(f1_small, f2_small, channel_axis=-1)
            scores.append(score)
            processed += 1
            
            print(f"Frame {frame_idx}: SSIM = {score:.4f} ({processed}/{args.max_frames})")
            sys.stdout.flush()
            
            # Explicit cleanup
            del f1_small, f2_small
        
        # Always cleanup original frames
        del f1, f2
        gc.collect()
        
        frame_idx += 1
    
    gt_cap.release()
    pred_cap.release()
    
    if len(scores) > 0:
        print(f"\n{'='*40}")
        print(f"RESULTS")
        print(f"{'='*40}")
        print(f"Processed frames: {len(scores)}")
        print(f"Average SSIM: {np.mean(scores):.4f}")
        print(f"Min SSIM: {np.min(scores):.4f}")
        print(f"Max SSIM: {np.max(scores):.4f}")
        print(f"{'='*40}")
    else:
        print("No frames processed!")

if __name__ == '__main__':
    main()
