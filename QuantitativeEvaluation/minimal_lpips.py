"""
Minimal LPIPS computation - memory efficient
"""
import cv2
import numpy as np
import torch
import lpips
import argparse
import gc
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('--pred', type=str, required=True)
    parser.add_argument('--max_frames', type=int, default=100)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--skip', type=int, default=1)
    args = parser.parse_args()
    
    print(f"Settings: max_frames={args.max_frames}, resize={args.resize}, skip={args.skip}")
    sys.stdout.flush()
    
    # Load LPIPS model
    print("Loading LPIPS model...")
    lpips_model = lpips.LPIPS(net='alex')
    if torch.cuda.is_available():
        lpips_model = lpips_model.cuda()
    print("LPIPS model loaded.")
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
        
        if frame_idx % args.skip == 0:
            # Resize
            f1_small = cv2.resize(f1, (args.resize, args.resize))
            f2_small = cv2.resize(f2, (args.resize, args.resize))
            
            # Convert BGR to RGB and normalize to [-1, 1]
            f1_tensor = torch.from_numpy(f1_small[:,:,::-1].copy()).permute(2,0,1).unsqueeze(0).float() / 255.0 * 2 - 1
            f2_tensor = torch.from_numpy(f2_small[:,:,::-1].copy()).permute(2,0,1).unsqueeze(0).float() / 255.0 * 2 - 1
            
            if torch.cuda.is_available():
                f1_tensor = f1_tensor.cuda()
                f2_tensor = f2_tensor.cuda()
            
            with torch.no_grad():
                score = lpips_model(f1_tensor, f2_tensor).item()
            
            scores.append(score)
            processed += 1
            
            if processed % 100 == 0 or processed <= 10:
                print(f"Frame {frame_idx}: LPIPS = {score:.4f} ({processed}/{args.max_frames})")
                sys.stdout.flush()
            
            del f1_small, f2_small, f1_tensor, f2_tensor
        
        del f1, f2
        if frame_idx % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        frame_idx += 1
    
    gt_cap.release()
    pred_cap.release()
    
    if len(scores) > 0:
        print(f"\n{'='*40}")
        print(f"RESULTS")
        print(f"{'='*40}")
        print(f"Processed frames: {len(scores)}")
        print(f"Average LPIPS: {np.mean(scores):.4f}")
        print(f"Min LPIPS: {np.min(scores):.4f}")
        print(f"Max LPIPS: {np.max(scores):.4f}")
        print(f"{'='*40}")
    else:
        print("No frames processed!")

if __name__ == '__main__':
    main()
