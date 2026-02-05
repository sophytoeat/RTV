"""
Inference script for Extended Hybrid Representation with Full 10-dimensional SMPL β Parameters

This script performs inference using:
    I_hybrid' = I_vm ⊕ I_sdp ⊕ I_β (16 channels)

Usage:
    python Inference/upperbody_inference_beta10.py \
        --input_video "path/to/video.mp4" \
        --garment_name "checkpoint_name" \
        --output "output.mp4"
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm

from VITON.viton_upperbody import UpperBodyVITON
from util.beta_utils import beta_to_tensor, extract_beta_from_smpl_param


def parse_args():
    parser = argparse.ArgumentParser(description='Inference with 10D β parameters')
    parser.add_argument('--input_video', type=str, required=True, help='Input video path')
    parser.add_argument('--garment_name', type=str, required=True, help='Checkpoint name')
    parser.add_argument('--output', type=str, default='./output_beta10.mp4', help='Output video path')
    parser.add_argument('--beta', type=float, nargs=10, default=[2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        help='10-dimensional β parameters (default: [2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])')
    parser.add_argument('--use_estimated_beta', action='store_true',
                        help='Use β estimated from SMPL regressor instead of fixed β')
    return parser.parse_args()


def make_pix2pix_model_beta10(name, input_nc=16, output_nc=4, model_name='pix2pixHD_RGBA', ckpt_dir=None):
    """
    Create pix2pixHD model for 10D β inference.
    
    Args:
        name: Checkpoint name
        input_nc: Input channels (16 for 10D β)
        output_nc: Output channels (4 for RGBA)
        model_name: Model type
        ckpt_dir: Checkpoint directory
    """
    from options.test_options import TestOptions
    from model.pix2pixHD.models import create_model
    
    opt = TestOptions().parse(save=False)
    opt.name = name
    opt.input_nc = input_nc
    opt.output_nc = output_nc
    opt.model = model_name
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True
    
    if ckpt_dir:
        opt.checkpoints_dir = ckpt_dir
    
    model = create_model(opt)
    return model


class UpperBodyVITONBeta10(UpperBodyVITON):
    """
    Extended VITON class with full 10-dimensional β support.
    """
    
    def __init__(self, target_garment_id, garment_name, beta=None, use_estimated_beta=False):
        """
        Args:
            target_garment_id: Target garment ID
            garment_name: Checkpoint name
            beta: Fixed 10-dimensional β parameters (if not using estimated)
            use_estimated_beta: If True, estimate β from SMPL regressor
        """
        # Initialize parent class (but we'll override the model)
        self.target_garment_id = target_garment_id
        self.garment_name = garment_name
        self.use_estimated_beta = use_estimated_beta
        
        # Set fixed β parameters
        if beta is not None:
            self.fixed_beta = np.array(beta, dtype=np.float32)
        else:
            self.fixed_beta = np.array([2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        
        # Initialize components
        self._init_components(garment_name)
    
    def _init_components(self, garment_name):
        """Initialize VITON components."""
        from BodyModels.SMPL_Regressor import Smpl_Regressor
        from BodyModels.DensePose_inference import DensePose_Inference
        from Matching.UpperBodyMatching import UpperBodyMatching_3, UpperBodyMatching_3_Lite
        
        print(f"Loading from CPU target garment id: {self.target_garment_id}")
        
        # Load SMPL regressor
        self.smpl_regressor = Smpl_Regressor()
        
        # Load DensePose
        self.dense_pose = DensePose_Inference()
        
        # Load matching module
        self.upper_body_matching = UpperBodyMatching_3(
            target_garment_id=self.target_garment_id,
            garment_data_root='./Garments/f_merged'
        )
        
        print(f"Loading from disk target garment id: {self.target_garment_id}")
        
        # Load pix2pixHD model with 16 input channels (for 10D β)
        self.pix2pix_model = make_pix2pix_model_beta10(
            garment_name,
            input_nc=16,  # 3 (VM) + 3 (SDP) + 10 (β)
            output_nc=4,
            model_name='pix2pixHD_RGBA'
        )
        
        print("Finished loading model")
    
    def _get_beta_tensor(self, smpl_param, height, width, device='cuda'):
        """
        Get 10D β tensor for the current frame.
        
        Args:
            smpl_param: SMPL parameters (used if use_estimated_beta is True)
            height: Feature map height
            width: Feature map width
            device: Target device
        
        Returns:
            β tensor of shape (1, 10, H, W)
        """
        if self.use_estimated_beta and smpl_param is not None:
            # Extract β from SMPL regressor
            beta = extract_beta_from_smpl_param(smpl_param)
        else:
            # Use fixed β
            beta = self.fixed_beta
        
        # Convert to tensor
        beta_tensor = beta_to_tensor(beta, height, width, device)
        return beta_tensor
    
    def inference_frame(self, frame, background=None):
        """
        Process a single frame.
        
        Args:
            frame: Input frame (BGR)
            background: Optional background image
        
        Returns:
            Output frame with virtual try-on result
        """
        from util.densepose_util import IUV2SDP
        import torchvision.transforms as transforms
        
        # Get SMPL parameters
        smpl_param = self.smpl_regressor.inference(frame)
        
        if smpl_param is None:
            return frame if background is None else background
        
        # Get DensePose IUV
        iuv = self.dense_pose.inference(frame)
        
        if iuv is None:
            return frame if background is None else background
        
        # Get virtual measurement garment
        vm_img, trans2roi = self.upper_body_matching.match(frame, smpl_param)
        
        if vm_img is None:
            return frame if background is None else background
        
        # Get simplified DensePose
        sdp_img = IUV2SDP(iuv)
        
        # Resize to model input size
        H, W = 512, 512
        vm_resized = cv2.resize(vm_img, (W, H))
        sdp_resized = cv2.resize(sdp_img, (W, H), interpolation=cv2.INTER_NEAREST)
        
        # Convert to tensors
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        vm_tensor = transform(vm_resized).unsqueeze(0).cuda()
        sdp_tensor = transform(sdp_resized).unsqueeze(0).cuda()
        
        # Normalize to [-1, 1]
        vm_tensor = vm_tensor * 2.0 - 1.0
        sdp_tensor = sdp_tensor * 2.0 - 1.0
        
        # Get 10D β tensor
        beta_tensor = self._get_beta_tensor(smpl_param, H, W, device='cuda')
        
        # Create hybrid input (16 channels)
        input_tensor = torch.cat([vm_tensor, sdp_tensor, beta_tensor], dim=1)
        
        # Inference
        with torch.no_grad():
            output = self.pix2pix_model.inference(input_tensor)
        
        # Extract RGB and alpha
        output_rgb = output[:, :3, :, :]
        output_alpha = output[:, 3:4, :, :]
        
        # Denormalize
        output_rgb = (output_rgb + 1.0) / 2.0
        output_alpha = torch.clamp(output_alpha, 0, 1)
        
        # Convert to numpy
        output_rgb_np = output_rgb[0].permute(1, 2, 0).cpu().numpy()
        output_alpha_np = output_alpha[0, 0].cpu().numpy()
        
        # Convert RGB to BGR for OpenCV
        output_bgr = (output_rgb_np[:, :, ::-1] * 255).astype(np.uint8)
        
        # Resize back to original size
        orig_h, orig_w = frame.shape[:2]
        output_bgr = cv2.resize(output_bgr, (orig_w, orig_h))
        output_alpha_np = cv2.resize(output_alpha_np, (orig_w, orig_h))
        
        # Composite with background
        if background is None:
            background = frame
        
        alpha_3ch = np.stack([output_alpha_np] * 3, axis=-1)
        result = (output_bgr * alpha_3ch + background * (1 - alpha_3ch)).astype(np.uint8)
        
        return result


def main():
    args = parse_args()
    
    print(f"Processing video: {args.input_video}")
    print(f"Garment: {args.garment_name}")
    print(f"β parameters: {args.beta}")
    print(f"Use estimated β: {args.use_estimated_beta}")
    
    # Open video
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.input_video}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input: {total_frames} frames, {width}x{height}, {fps} fps")
    print(f"Output: {args.output}")
    
    # Create VITON instance
    viton = UpperBodyVITONBeta10(
        target_garment_id=0,
        garment_name=args.garment_name,
        beta=args.beta,
        use_estimated_beta=args.use_estimated_beta
    )
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Process frames
    for _ in tqdm(range(total_frames), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break
        
        result = viton.inference_frame(frame)
        out.write(result)
    
    cap.release()
    out.release()
    
    # Check output file size
    output_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nSuccess! Output saved: {args.output} ({output_size:.1f} MB)")


if __name__ == "__main__":
    main()
