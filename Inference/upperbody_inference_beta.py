"""
Inference script for Extended Hybrid Representation with β Parameters

This script performs virtual try-on using the Extended Hybrid Representation:
    I_hybrid' = I_vm ⊕ I_sdp ⊕ I_β

Usage:
    python upperbody_inference_beta.py --input_video <video_path> --garment_name <garment> --use_beta
"""

import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from Inference.base_options import BaseOptions
from util.multithread_video_loader import MultithreadVideoLoader
from util.multithread_video_writer import MultithreadVideoWriter
from tqdm import tqdm
from VITON.viton_upperbody_beta import FrameProcessorBeta


def process_video(video_path, garment_name, use_beta=True, external_beta=None, output_path='./output.mp4'):
    """
    Process video with Extended Hybrid Representation.
    
    Args:
        video_path: Input video path
        garment_name: Garment checkpoint name
        use_beta: Whether to use β parameters
        external_beta: Optional fixed β parameters to use for all frames
        output_path: Output video path
    """
    video_loader = MultithreadVideoLoader(video_path, max_height=1024)
    video_writer = MultithreadVideoWriter(outvid=output_path, fps=video_loader.get_fps())
    
    print(f"Processing video: {video_path}")
    print(f"Garment: {garment_name}")
    print(f"Using β parameters: {use_beta}")
    
    # Create frame processor with Extended Hybrid Representation
    frame_processor = FrameProcessorBeta(
        [garment_name], 
        ckpt_dir='./checkpoints/',
        use_beta=use_beta
    )
    frame_processor.switch_to_target_garment(0)
    
    # If external β is provided, set it
    if external_beta is not None:
        frame_processor.set_beta(external_beta)
    
    for i in tqdm(range(len(video_loader)), desc="Processing frames"):
        frame = video_loader.cap()
        result = frame_processor(frame, external_beta=external_beta)
        video_writer.append(result)
        
        # Print β parameters periodically
        if i % 100 == 0 and use_beta:
            current_beta = frame_processor.get_current_beta()
            if current_beta is not None:
                print(f"Frame {i} β: {current_beta[:3]}...")  # Print first 3 values
    
    video_writer.make_video()
    video_writer.close()
    
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Virtual Try-On with Extended Hybrid Representation')
    parser.add_argument('--input_video', type=str, required=True, help='Input video path')
    parser.add_argument('--garment_name', type=str, required=True, help='Garment checkpoint name')
    parser.add_argument('--output', type=str, default='./output.mp4', help='Output video path')
    parser.add_argument('--use_beta', action='store_true', help='Use β parameters (Extended Hybrid)')
    parser.add_argument('--beta', type=float, nargs=10, default=None, 
                        help='External β parameters (10 values)')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/', 
                        help='Checkpoints directory')
    
    args = parser.parse_args()
    
    # Process the video
    process_video(
        args.input_video,
        args.garment_name,
        use_beta=args.use_beta,
        external_beta=args.beta,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
