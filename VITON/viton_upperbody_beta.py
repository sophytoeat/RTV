"""
Virtual Try-On with Extended Hybrid Representation (β Parameter Support)

This module extends the FrameProcessor to use the Extended Hybrid Representation:
    I_hybrid' = I_vm ⊕ I_sdp ⊕ I_β

The β parameter (10-dimensional SMPL body shape) is incorporated into the input
to enable body-shape-aware garment synthesis.
"""

import os
import threading
import numpy as np
import cv2
import torch

import util.util
from options.test_options import TestOptions
from model.pix2pixHD.models import create_model
import util.util as util
from util.multithread_video_loader import MultithreadVideoLoader
from util.image2video import Image2VideoWriter
from SMPL.smpl_regressor import SMPL_Regressor
import glm
from OffscreenRenderer.flat_renderer import FlatRenderer
from util.image_warp import zoom_in, zoom_out, shift_image_right, shift_image_down, rotate_image
from util.image_process import blur_image
from model.DensePose.densepose_extractor import DensePoseExtractor
from util.beta_utils import beta_to_tensor_1d, BetaFeatureGenerator, estimate_beta_from_joints
import time

from SMPL.upperbody_smpl.UpperBody import UpperBodySMPL
from tqdm import tqdm
from composition.naive_overlay import naive_overlay, naive_overlay_alpha
from util.densepose_util import IUV2UpperBodyImg, IUV2TorsoLeg, IUV2SDP
from threading import Thread


def make_pix2pix_model_beta(name, input_nc=7, output_nc=4, model_name='pix2pixHD_RGBA', ckpt_dir=None):
    """
    Create pix2pix model with Extended Hybrid Representation input.

    Args:
        name: Model checkpoint name
        input_nc: Input channels (default 7 = 3 + 3 + 1)
        output_nc: Output channels (default 4 = RGB + Alpha)
        model_name: Model architecture name
        ckpt_dir: Checkpoint directory
    """
    opt = TestOptions().parse(save=False, use_default=True, show_info=False)
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.name = name
    opt.input_nc = input_nc
    opt.output_nc = output_nc
    opt.isTrain = False
    opt.model = model_name
    if ckpt_dir is None:
        opt.checkpoints_dir = './rtv_ckpts'
    else:
        opt.checkpoints_dir = ckpt_dir
    opt.gpu_ids = ''  # Load to CPU first
    model = create_model(opt)
    return model


class FrameProcessorBeta:
    """
    Frame processor with Extended Hybrid Representation (β parameter support).

    This processor creates:
        I_hybrid' = I_vm ⊕ I_sdp ⊕ I_β

    where:
        - I_vm: Virtual Measurement Garment (3 channels)
        - I_sdp: Simplified DensePose (3 channels)
        - I_β0: Body Shape Feature Map using β[0] only (1 channel)

    The β parameter can be:
        1. Estimated from 2D joints (automatic)
        2. Provided externally (manual)
        3. Loaded from SMPL regressor output
    """

    def __init__(self, garment_name_list, ckpt_dir=None, use_beta=True):
        """
        Args:
            garment_name_list: List of garment checkpoint names
            ckpt_dir: Checkpoint directory
            use_beta: Whether to use β parameters (True for Extended Hybrid)
        """
        self.smpl_regressor = SMPL_Regressor(use_bev=True)
        self.viton_model = None
        self.ckpt_dir = ckpt_dir
        self.densepose_extractor = DensePoseExtractor()
        self.upper_body = UpperBodySMPL()
        self.garment_name_list = garment_name_list
        self.viton_model_list = [None for _ in range(len(self.garment_name_list))]
        self.lock = threading.Lock()

        # Extended Hybrid Representation settings
        self.use_beta = use_beta
        self.input_nc = 7 if use_beta else 6

        # β parameter state
        self.current_beta = None  # First frame's β (fixed after first frame)
        self.beta_smoothing = 0.9  # Temporal smoothing factor (not used when using first frame only)
        self.beta_generator = BetaFeatureGenerator(height=512, width=512)

        # Body-shape scalar estimation parameters
        self.shape_s = None
        self.mu_slim = 0.27
        self.mu_fat = 0.34

        # Start background model loading
        self.load_all = Thread(target=self.load_all_models, args=())
        self.load_all.daemon = True
        self.load_all.start()

    def load_all_models(self):
        """Load all models in background."""
        for i, garment_name in enumerate(self.garment_name_list):
            new_model = make_pix2pix_model_beta(
                garment_name,
                input_nc=self.input_nc,
                output_nc=4,
                ckpt_dir=self.ckpt_dir
            )
            if self.viton_model_list[i] is None:
                self.viton_model_list[i] = new_model

    def load_one_model(self, garment_name):
        """Load a single model."""
        new_model = make_pix2pix_model_beta(
            garment_name,
            input_nc=self.input_nc,
            output_nc=4,
            ckpt_dir=self.ckpt_dir
        )
        idx = self.garment_name_list.index(garment_name)
        if self.viton_model_list[idx] is None:
            self.viton_model_list[idx] = new_model

    def switch_to_target_garment(self, garment_id):
        """Switch to target garment model."""
        self.lock.acquire()
        print(f"Loading from CPU target garment id: {garment_id}")

        if self.viton_model_list[garment_id] is None and garment_id >= 0:
            print(f"Loading from disk target garment id: {garment_id}")
            self.load_one_model(self.garment_name_list[garment_id])

        old_model = self.viton_model
        new_model = self.viton_model_list[garment_id].to('cuda:0') if garment_id >= 0 else None

        if self.viton_model is not None:
            del self.viton_model
        self.viton_model = new_model

        if old_model is not None:
            old_model = old_model.to('cpu')
            del old_model
            torch.cuda.empty_cache()

        print("Finished loading model")
        self.lock.release()

    def set_target_garment(self, target_id):
        """Set target garment (async)."""
        t = Thread(target=self.switch_to_target_garment, args=(target_id,))
        t.daemon = True
        t.start()

    def set_beta(self, beta):
        """
        Manually set the β parameter.

        Args:
            beta: SMPL β parameters, shape (10,)
        """
        if beta is not None:
            self.current_beta = np.array(beta, dtype=np.float32).flatten()[:10]
        else:
            self.current_beta = None

    def _estimate_beta_from_smpl(self, smpl_param):
        """
        Extract or estimate β from SMPL regressor output.

        Args:
            smpl_param: SMPL parameter dictionary from regressor

        Returns:
            β parameters, shape (10,)
        """
        # Try to get β directly from SMPL parameters
        if 'smpl_betas' in smpl_param:
            betas = smpl_param['smpl_betas']
            if isinstance(betas, torch.Tensor):
                betas = betas.cpu().numpy()
            if betas.ndim > 1:
                betas = betas[0]
            return betas.flatten()[:10].astype(np.float32)

        # Backward-compatible key
        if 'betas' in smpl_param:
            betas = smpl_param['betas']
            if isinstance(betas, torch.Tensor):
                betas = betas.cpu().numpy()
            if betas.ndim > 1:
                betas = betas[0]
            return betas.flatten()[:10].astype(np.float32)

        # Try to estimate from 2D joints
        if 'pj2d_org' in smpl_param:
            try:
                joints = smpl_param['pj2d_org']
                cam_trans = smpl_param['cam_trans']
                depth_order = torch.sort(cam_trans[:, 2].cpu(), descending=False).indices.numpy()
                J = joints[depth_order][0].cpu().numpy()
                return estimate_beta_from_joints(J, self.mu_slim, self.mu_fat)
            except Exception:
                pass

        return np.zeros(10, dtype=np.float32)

    def _smooth_beta(self, new_beta):
        """Apply temporal smoothing to β parameters."""
        if self.current_beta is None:
            return new_beta

        # Exponential moving average
        return self.beta_smoothing * self.current_beta + (1 - self.beta_smoothing) * new_beta

    def __call__(self, input_frame, external_beta=None):
        """
        Process a single frame with Extended Hybrid Representation.

        Args:
            input_frame: Input video frame (BGR)
            external_beta: Optional external β parameters

        Returns:
            Processed frame with virtual garment
        """
        if self.viton_model is None:
            return input_frame

        resolution = 512
        raw_image = input_frame

        # Run SMPL regression
        smpl_data = self.smpl_regressor.forward(raw_image, True, size=1.45, roi_img_size=resolution)
        if len(smpl_data) < 3:
            return input_frame

        smpl_param, trans2roi, inv_trans2roi = smpl_data

        if smpl_param is None:
            return input_frame

        vertices = SMPL_Regressor.get_raw_verts(smpl_param)
        vertices = torch.from_numpy(vertices).unsqueeze(0)

        height = raw_image.shape[0]
        width = raw_image.shape[1]

        # Get β parameters
        # IMPORTANT: Training data used fixed beta[0]=2.0, so we must use the same value
        # for inference to match the training distribution
        if external_beta is not None:
            # External β provided: use it and save as first frame's β
            beta = np.array(external_beta, dtype=np.float32).flatten()[:10]
            if self.current_beta is None:
                self.current_beta = beta.copy()
        elif self.use_beta:
            # Use fixed beta[0]=2.0 to match training data
            # Training datasets (f_fat_gap_beta, f_no_gap_beta) all used beta[0]=2.0
            beta = np.zeros(10, dtype=np.float32)
            beta[0] = 2.0  # Match training data value
            if self.current_beta is None:
                print(f"[FrameProcessorBeta] Using training-matched β[0]=2.0")
                self.current_beta = beta.copy()
        else:
            beta = np.zeros(10, dtype=np.float32)

        # Render Virtual Measurement Garment (I_vm)
        raw_vm = self.upper_body.render(vertices[0], height=height, width=width)

        # Get DensePose (I_sdp)
        raw_IUV = self.densepose_extractor.get_IUV(raw_image, isRGB=False)
        if raw_IUV is None:
            return input_frame

        dpi_img = IUV2SDP(raw_IUV)
        roi_dpi_img = cv2.warpAffine(dpi_img, trans2roi, (resolution, resolution),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))

        roi_vm = cv2.warpAffine(raw_vm, trans2roi, (resolution, resolution),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))

        # Convert to tensors
        vm_tensor = util.im2tensor(roi_vm) * 2.0 - 1.0
        vm_tensor = vm_tensor[:, [2, 1, 0], :, :]  # BGR to RGB
        dp_tensor = util.im2tensor(roi_dpi_img) * 2.0 - 1.0

        self.lock.acquire()
        with torch.no_grad():
            if self.viton_model is not None:
                if self.use_beta:
                    # Create Extended Hybrid Representation
                    # I_hybrid' = I_vm ⊕ I_sdp ⊕ I_β0
                    beta_tensor = beta_to_tensor_1d(
                        beta,
                        resolution,
                        resolution,
                        device='cuda',
                        compression_method='first',  # use only β[0]
                    )
                    inp = torch.cat([vm_tensor.cuda(), dp_tensor.cuda(), beta_tensor], dim=1)
                else:
                    # Original representation
                    inp = torch.cat([vm_tensor, dp_tensor], dim=1).cuda()

                # Forward pass
                target_tensor = self.viton_model.forward(inp)
                self.lock.release()
            else:
                self.lock.release()
                return input_frame

        # Convert output to image
        roi_target = util.tensor2im(target_tensor[0, [0, 1, 2], :, :], normalize=True, rgb=False)
        roi_alpha = (target_tensor[0, 3, :, :].clamp(min=0.0, max=1.0).cpu().numpy() * 255).astype(np.uint8)

        # Warp back to original image space
        raw_target_img = cv2.warpAffine(roi_target, inv_trans2roi,
                                        (raw_image.shape[1], raw_image.shape[0]),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))
        raw_alpha = cv2.warpAffine(roi_alpha, inv_trans2roi,
                                   (raw_image.shape[1], raw_image.shape[0]),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0,))

        # Compose final image
        composed_img = naive_overlay_alpha(raw_image, raw_target_img, raw_alpha)

        return composed_img

    def get_current_beta(self):
        """Get the current β parameters."""
        return self.current_beta.copy() if self.current_beta is not None else None

    def reset_beta(self):
        """Reset β state (useful for new video)."""
        self.current_beta = None
        self.shape_s = None
        self.beta_generator.reset_cache()


# Backward compatible alias
class FrameProcessor(FrameProcessorBeta):
    """
    Backward compatible FrameProcessor that can operate in either mode.

    Set use_beta=False for original behavior, use_beta=True for Extended Hybrid.
    """
    def __init__(self, garment_name_list, ckpt_dir=None, use_beta=False):
        super().__init__(garment_name_list, ckpt_dir, use_beta=use_beta)
