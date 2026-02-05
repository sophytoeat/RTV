import os
import threading

import numpy as np
import cv2

import util.util
from options.test_options import TestOptions
from model.pix2pixHD.models import create_model
import util.util as util
import torch
from util.multithread_video_loader import MultithreadVideoLoader
from util.image2video import Image2VideoWriter
from SMPL.smpl_regressor import SMPL_Regressor
import glm
from OffscreenRenderer.flat_renderer import FlatRenderer
from util.image_warp import zoom_in, zoom_out, shift_image_right, shift_image_down, rotate_image
from util.image_process import blur_image
#from composition.short_sleeve_composition import ShortSleeveComposer
from model.DensePose.densepose_extractor import DensePoseExtractor
import time

from SMPL.upperbody_smpl.UpperBody import UpperBodySMPL
from tqdm import tqdm
from composition.naive_overlay import naive_overlay, naive_overlay_alpha
from util.densepose_util import IUV2UpperBodyImg, IUV2TorsoLeg, IUV2SDP
from util.beta_utils import create_hybrid_input, extract_beta_from_smpl_param
from threading import Thread


def make_pix2pix_model(name, input_nc, output_nc=3, model_name='pix2pixHD',ckpt_dir=None, use_beta=False):
    opt = TestOptions().parse(save=False, use_default=True, show_info=False)
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.name = name
    # Adjust input_nc based on β usage: 6 (vm+dp) or 16 (vm+dp+β)
    if use_beta:
        opt.input_nc = 16  # 3 (vm) + 3 (dp) + 10 (β)
    else:
        opt.input_nc = input_nc if input_nc else 6  # Default to 6 if not specified
    opt.output_nc = output_nc
    opt.isTrain = False
    opt.model = model_name
    if ckpt_dir is None:
        opt.checkpoints_dir='./rtv_ckpts'
    else:
        opt.checkpoints_dir=ckpt_dir
    opt.gpu_ids=''#load to cpu
    model = create_model(opt)
    # print(model)
    return model


class FrameProcessor:
    def __init__(self, garment_name_list,ckpt_dir=None, use_beta=False, fix_first_frame_beta=True):
        self.smpl_regressor = SMPL_Regressor(use_bev=True)
        self.viton_model = None
        self.ckpt_dir = ckpt_dir
        self.densepose_extractor = DensePoseExtractor()
        self.upper_body = UpperBodySMPL()
        self.garment_name_list = garment_name_list#[2, 3, 17, 18, 22]
        self.viton_model_list = [None for i in range(len(self.garment_name_list))]
        self.lock = threading.Lock()
        self.use_beta = use_beta  # Whether to use Extended Hybrid Representation with β
        
        # Whether to use first frame's β throughout entire video (default: True)
        # When True: First frame's β is cached and reused for all subsequent frames
        # When False: β is extracted fresh from each frame's SMPL parameters
        self.fix_first_frame_beta = fix_first_frame_beta

        # Body-shape scalar s in [0,1], estimated on first valid frame (for legacy blending)
        self.shape_s = None
        # Calibration medians for shoulder/height ratio (can be adjusted externally)
        # Defaults are placeholders; tune with your dataset medians
        self.mu_slim = 0.27
        self.mu_fat = 0.34
        
        # Cache for β parameters (for Extended Hybrid Representation)
        # When fix_first_frame_beta=True, first frame's β is stored here and reused
        self.cached_beta = None

        self.load_all = Thread(target=self.load_all_models, args=())
        self.load_all.daemon = True
        self.load_all.start()

    def load_all_models(self):
        for i, garment_name in enumerate(self.garment_name_list):
            new_model = make_pix2pix_model(garment_name, 6, output_nc=4,ckpt_dir=self.ckpt_dir, use_beta=self.use_beta)
            if self.viton_model_list[i] is None:
                self.viton_model_list[i]= new_model

    def load_one_models(self, garment_name):
        new_model = make_pix2pix_model(garment_name, 6, output_nc=4,ckpt_dir=self.ckpt_dir, use_beta=self.use_beta)
        id = self.garment_name_list.index(garment_name)
        if self.viton_model_list[id] is None:
            self.viton_model_list[id]= new_model

    def switch_to_target_garment(self,garment_id):
        self.lock.acquire()
        print("Loading from CPU target garment id: ", garment_id)
        id = garment_id
        if self.viton_model_list[id] is None and id >= 0:
            print("Loading from disk target garment id: ", garment_id)
            self.load_one_models(self.garment_name_list[id])
        old_model = self.viton_model
        new_model=self.viton_model_list[id].to('cuda:0') if garment_id>=0 else None
        print("Finished")
        if self.viton_model is not None:
            del self.viton_model
        self.viton_model = new_model
        if old_model is not None:
            old_model = old_model.to('cpu')
            del old_model
            torch.cuda.empty_cache()
        self.lock.release()


    def set_target_garment(self, target_id):
        #new_model = make_pix2pix_model(ckpt_dict[target_id], 6, output_nc=4)
        #self.viton_model = self.viton_model_list[target_id]
        t = Thread(target=self.switch_to_target_garment, args=(target_id,))
        t.daemon = True
        t.start()

    def reset_beta_cache(self):
        """
        Reset the cached β parameters.
        Call this when starting to process a new video to ensure
        the first frame's β of the new video is used.
        """
        self.cached_beta = None
        self.shape_s = None
        print("[FrameProcessor] β cache reset")

    def __call__(self, input_frame):
        if self.viton_model is None:
            return input_frame

        resolution = 512
        raw_image = input_frame

        smpl_data = self.smpl_regressor.forward(raw_image, True, size=1.45,
                                                roi_img_size=resolution)
        if len(smpl_data) < 3:
            return input_frame
        smpl_param, trans2roi, inv_trans2roi = smpl_data

        if smpl_param is None:
            return input_frame
        vertices = SMPL_Regressor.get_raw_verts(smpl_param)
        vertices = torch.from_numpy(vertices).unsqueeze(0)

        height = raw_image.shape[0]
        width = raw_image.shape[1]

        v = vertices

        # Estimate body-shape scalar s once using 2D joints (shoulder/height)
        if self.shape_s is None and 'pj2d_org' in smpl_param:
            try:
                # Follow the same joint indexing convention used in get_trans2roi
                joints = smpl_param['pj2d_org']
                cam_trans = smpl_param['cam_trans']
                depth_order = torch.sort(cam_trans[:, 2].cpu(), descending=False).indices.numpy()
                J = joints[depth_order][0].cpu().numpy()
                # Heuristic indices: 9 and 12 as shoulders, 0 head-top proxy, 15 lower body/ankle proxy
                shoulder_width = float(np.linalg.norm(J[9] - J[12]))
                body_height = float(np.linalg.norm(J[15] - J[0]))
                if body_height > 1e-6:
                    z = shoulder_width / body_height
                    s_raw = (z - self.mu_slim) / max(1e-6, (self.mu_fat - self.mu_slim))
                    self.shape_s = float(np.clip(s_raw, 0.0, 1.0))
                else:
                    self.shape_s = 0.5
            except Exception:
                self.shape_s = 0.5

        raw_vm = self.upper_body.render(v[0], height=height, width=width)

        raw_IUV = self.densepose_extractor.get_IUV(raw_image, isRGB=False)
        if raw_IUV is None:
            return input_frame
        dpi_img = IUV2SDP(raw_IUV)
        roi_dpi_img = cv2.warpAffine(dpi_img, trans2roi, (resolution, resolution), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))

        roi_vm = cv2.warpAffine(raw_vm, trans2roi, (resolution, resolution), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))

        vm_tensor = util.im2tensor(roi_vm) * 2.0 - 1.0
        vm_tensor = vm_tensor[:, [2, 1, 0], :, :]
        dp_tensor = util.im2tensor(roi_dpi_img) * 2.0 - 1.0
        
        # Extract β parameters for Extended Hybrid Representation
        beta = None
        if self.use_beta:
            if self.fix_first_frame_beta:
                # Use first frame's β throughout entire video for temporal consistency
                if self.cached_beta is None:
                    # Extract β from first frame's SMPL parameters and cache it
                    beta = extract_beta_from_smpl_param(smpl_param, self.mu_slim, self.mu_fat)
                    # IMPORTANT: Training data only used β[0], set other components to zero
                    # to match training data format
                    beta_fixed = np.zeros(10, dtype=np.float32)
                    beta_fixed[0] = beta[0]
                    beta = beta_fixed
                    self.cached_beta = beta
                    print(f"[FrameProcessor] First frame β cached: β[0]={beta[0]:.4f} (fix_first_frame_beta=True)")
                else:
                    beta = self.cached_beta
            else:
                # Extract fresh β from each frame's SMPL parameters
                beta = extract_beta_from_smpl_param(smpl_param, self.mu_slim, self.mu_fat)
                # IMPORTANT: Training data only used β[0], set other components to zero
                beta_fixed = np.zeros(10, dtype=np.float32)
                beta_fixed[0] = beta[0]
                beta = beta_fixed
        
        self.lock.acquire()
        with torch.no_grad():
            if self.viton_model is not None:
                # Create Extended Hybrid Representation: I_hybrid' = I_vm ⊕ I_sdp ⊕ I_β
                if self.use_beta and beta is not None:
                    inp = create_hybrid_input(vm_tensor, dp_tensor, beta).cuda()
                else:
                    inp = torch.cat([vm_tensor, dp_tensor], 1).cuda()
                
                # Legacy blending mode: If two models are available (e.g., slim and fat), blend their outputs by shape_s
                if not self.use_beta and len(self.viton_model_list) >= 2 and self.viton_model_list[0] is not None and self.viton_model_list[1] is not None:
                    # Ensure models are on GPU
                    model_a = self.viton_model_list[0].to('cuda:0')
                    model_b = self.viton_model_list[1].to('cuda:0')
                    out_a = model_a.forward(inp)
                    out_b = model_b.forward(inp)
                    s = 0.5 if self.shape_s is None else float(self.shape_s)
                    target_tensor = (1.0 - s) * out_a + s * out_b
                else:
                    target_tensor = self.viton_model.forward(inp)
                self.lock.release()
            else:
                self.lock.release()
                return input_frame
        roi_target = util.tensor2im(target_tensor[0, [0, 1, 2], :, :], normalize=True, rgb=False)
        roi_alpha = (target_tensor[0, 3, :, :].clamp(min=0.0, max=1.0).cpu().numpy() * 255).astype(np.uint8)

        raw_target_img = cv2.warpAffine(roi_target, inv_trans2roi, (raw_image.shape[1], raw_image.shape[0]),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))
        raw_alpha = cv2.warpAffine(roi_alpha, inv_trans2roi, (raw_image.shape[1], raw_image.shape[0]),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0,))
        composed_img = naive_overlay_alpha(raw_image, raw_target_img, raw_alpha)
        return composed_img
