import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..","..")))

from Inference.base_options import BaseOptions
from util.multithread_video_loader import MultithreadVideoLoader
from util.multithread_video_writer import MultithreadVideoWriter
from tqdm import tqdm
from VITON.viton_upperbody import FrameProcessor

def process_video(video_path, garment_name, use_beta=False, fix_first_frame_beta=True, output_path='./output.mp4'):
    video_loader = MultithreadVideoLoader(video_path,max_height=1024)
    video_writer = MultithreadVideoWriter(outvid=output_path,fps=video_loader.get_fps())
    # use_beta=True for models trained with Extended Hybrid Representation (16ch)
    # use_beta=False for models trained with original representation (6ch)
    # fix_first_frame_beta=True: use first frame's β throughout video for temporal consistency
    frame_processor = FrameProcessor([garment_name,], ckpt_dir='./checkpoints/', 
                                     use_beta=use_beta, fix_first_frame_beta=fix_first_frame_beta)
    frame_processor.switch_to_target_garment(0)
    for i in tqdm(range(len(video_loader))):
        frame = video_loader.cap()
        result = frame_processor(frame)
        video_writer.append(result)
    video_writer.make_video()
    video_writer.close()
    
    import time
    time.sleep(2)  # Wait for file to be finalized
    
    import os
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\nOutput saved: {output_path} ({size_mb:.1f} MB)")
    else:
        print(f"\nWarning: Output file not found at {output_path}")

if __name__ == '__main__':
    opts = BaseOptions()
    opt = opts.parse()
    video_path = opt.input_video
    garment_name = opt.garment_name
    # use_beta=True for models trained with --use_beta flag (16ch input)
    # use_beta=False for models trained without --use_beta flag (6ch input)
    # Handle both --use_beta and --no_beta flags
    if hasattr(opt, 'no_beta') and opt.no_beta:
        use_beta = False
    else:
        use_beta = getattr(opt, 'use_beta', False)
    
    # fix_first_frame_beta=True: use first frame's β throughout entire video
    # This ensures temporal consistency of body shape across frames
    if hasattr(opt, 'no_fix_first_frame_beta') and opt.no_fix_first_frame_beta:
        fix_first_frame_beta = False
    else:
        fix_first_frame_beta = getattr(opt, 'fix_first_frame_beta', True)
    
    output_path = getattr(opt, 'output', './output.mp4')
    
    print(f"Using Extended Hybrid Representation (beta): {use_beta}")
    print(f"Fix first frame beta: {fix_first_frame_beta}")
    print(f"Output: {output_path}")
    process_video(video_path, garment_name, use_beta=use_beta, fix_first_frame_beta=fix_first_frame_beta, output_path=output_path)


