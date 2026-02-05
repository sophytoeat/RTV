"""
Simple inference script with reliable video output
"""
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..","..")))

import cv2
from tqdm import tqdm
from VITON.viton_upperbody import FrameProcessor

def process_video(video_path, garment_name, output_path, use_beta=False):
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Input: {frame_count} frames, {width}x{height}, {fps} fps")
    print(f"Output: {output_path}")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Error: Cannot create output video")
        return
    
    # Setup frame processor
    frame_processor = FrameProcessor([garment_name], ckpt_dir='./checkpoints/', 
                                     use_beta=use_beta, fix_first_frame_beta=True)
    frame_processor.switch_to_target_garment(0)
    
    # Process frames
    for i in tqdm(range(frame_count), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break
        
        result = frame_processor(frame)
        out.write(result)
    
    # Cleanup
    cap.release()
    out.release()
    
    # Verify output
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\nSuccess! Output saved: {output_path} ({size_mb:.1f} MB)")
    else:
        print("\nError: Output file was not created")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, required=True)
    parser.add_argument('--garment_name', type=str, required=True)
    parser.add_argument('--output', type=str, default='./output_result.mp4')
    parser.add_argument('--use_beta', action='store_true')
    args = parser.parse_args()
    
    print(f"Using beta: {args.use_beta}")
    process_video(args.input_video, args.garment_name, args.output, args.use_beta)
