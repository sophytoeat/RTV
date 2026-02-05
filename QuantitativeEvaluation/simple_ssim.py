"""
Simple SSIM computation between two videos.
Processes frame by frame to minimize memory usage.
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('--pred', type=str, required=True)
    parser.add_argument('--max_frames', type=int, default=100)
    args = parser.parse_args()
    
    gt_cap = cv2.VideoCapture(args.gt)
    pred_cap = cv2.VideoCapture(args.pred)
    
    scores = []
    for i in tqdm(range(args.max_frames)):
        ret1, f1 = gt_cap.read()
        ret2, f2 = pred_cap.read()
        if not ret1 or not ret2:
            break
        
        # Resize if needed
        if f1.shape != f2.shape:
            f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]))
        
        score = ssim(f1, f2, channel_axis=-1)
        scores.append(score)
        
        if i % 20 == 0:
            print(f"Frame {i}: SSIM = {score:.4f}")
    
    gt_cap.release()
    pred_cap.release()
    
    print(f"\nAverage SSIM: {np.mean(scores):.4f}")
    print(f"Processed {len(scores)} frames")

if __name__ == '__main__':
    main()
