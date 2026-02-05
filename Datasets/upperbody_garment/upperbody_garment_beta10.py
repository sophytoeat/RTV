"""
Extended UpperBodyGarment Dataset with Full 10-dimensional SMPL β Parameters

This dataset extends the original UpperBodyGarment to include full body shape information
for the Extended Hybrid Representation with 10-dimensional β.

I_hybrid' = I_vm ⊕ I_sdp ⊕ I_β (16 channels total)
- I_vm: 3 channels
- I_sdp: 3 channels  
- I_β: 10 channels
"""

import random
import PIL
import torchvision.transforms as transforms
import torch
import numpy as np
import os
import torch.utils.data as data
from PIL import Image
from util.densepose_util import IUV2UpperBodyImg, IUV2TorsoLeg, IUV2Img, IUV2SDP, IUV2SSDP
from util.cv2_trans_util import get_inverse_trans
from util.beta_utils import beta_to_tensor
import cv2
import json

from .upperbody_garment import UpperBodyGarment, RandomAffineMatrix


class UpperBodyGarmentBeta10(UpperBodyGarment):
    """
    Extended dataset that includes full 10-dimensional SMPL β parameters.
    
    This dataset returns:
        - garment_img: Ground truth garment image (3, H, W)
        - vm_img: Virtual Measurement Garment (3, H, W)
        - dp_img: Simplified DensePose (3, H, W)
        - beta_img: Body Shape Feature Map using all 10 β dimensions (10, H, W)
        - garment_mask: Garment alpha mask (1, H, W)
    
    The Extended Hybrid Representation can be created by:
        I_hybrid' = torch.cat([vm_img, dp_img, beta_img], dim=0)  # 16 channels
    """
    
    def __init__(self, path, img_size=512, use_random_beta=False):
        """
        Args:
            path: Dataset directory path
            img_size: Output image size
            use_random_beta: If True, use random β for data augmentation
        """
        super().__init__(path, img_size)
        self.use_random_beta = use_random_beta
        self.img_size = img_size
        
        # Try to load pre-computed β parameters if available
        self.beta_dict = {}
        beta_file = os.path.join(self.img_dir, 'beta_params.json')
        if os.path.exists(beta_file):
            with open(beta_file, 'r') as f:
                self.beta_dict = json.load(f)
    
    def __getitem__(self, index):
        garment_path = self.image_list[index]
        garment_img = np.array(Image.open(garment_path))
        raw_h, raw_w = self.raw_height, self.raw_width
        
        # Get frame ID
        frame_id = (os.path.basename(self.image_list[index])).split('_')[0]
        
        trans2roi_path = os.path.join(self.img_dir, frame_id + '_trans2roi.npy')
        trans2roi = np.load(trans2roi_path)
        inv_trans = get_inverse_trans(trans2roi)
        garment_img = cv2.warpAffine(garment_img, inv_trans, (raw_w, raw_h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))

        vm_path = os.path.join(self.img_dir, frame_id + '_vm.jpg')
        vm_img = np.array(Image.open(vm_path))
        vm_img = cv2.resize(vm_img, (raw_w, raw_h))
        
        mask_path = os.path.join(self.img_dir, frame_id + '_mask.png')
        mask_img = np.array(Image.open(mask_path))
        mask_img = cv2.resize(mask_img, (raw_w, raw_h))
        
        iuv_path = os.path.join(self.img_dir, frame_id + '_iuv.npy')
        IUV = np.load(iuv_path)
        dp_img = IUV2SDP(IUV)
        dp_img = cv2.resize(dp_img, (raw_w, raw_h), cv2.INTER_NEAREST)
        
        # Load or generate β parameters (10-dimensional)
        beta = self._get_beta(frame_id)
        
        # Apply random affine transformation
        new_trans2roi = self.randomaffine(trans2roi)
        
        roi_garment_img = cv2.warpAffine(garment_img, new_trans2roi, (1024, 1024),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0))
        roi_dp_img = cv2.warpAffine(dp_img, new_trans2roi, (1024, 1024),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0))
        roi_vm_img = cv2.warpAffine(vm_img, new_trans2roi, (1024, 1024),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0))
        roi_mask_img = cv2.warpAffine(mask_img, new_trans2roi, (1024, 1024),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0))
        
        if self.transform is not None:
            roi_garment_img = self.transform(PIL.Image.fromarray(roi_garment_img))
            roi_vm_img = self.transform(PIL.Image.fromarray(roi_vm_img))
            roi_mask_img = self.transform(PIL.Image.fromarray(roi_mask_img))
            roi_dp_img = self.transform(PIL.Image.fromarray(roi_dp_img))
        
        # Create 10-channel β feature map tensor
        # beta_to_tensor returns (1, 10, H, W), we need (10, H, W)
        beta_tensor = beta_to_tensor(
            beta,
            self.img_size,
            self.img_size,
            device='cpu',
        ).squeeze(0)  # (10, H, W)
        
        return (self._normalize(roi_garment_img), 
                self._normalize(roi_vm_img), 
                self._normalize(roi_dp_img),
                beta_tensor,  # 10 channels, already in [-1, 1] range
                roi_mask_img)
    
    def _get_beta(self, frame_id):
        """Get β parameters for a frame (10-dimensional)."""
        # Try to load from pre-computed dictionary
        if frame_id in self.beta_dict:
            return np.array(self.beta_dict[frame_id], dtype=np.float32)
        
        # Try to load from .npy file
        beta_path = os.path.join(self.img_dir, frame_id + '_beta.npy')
        if os.path.exists(beta_path):
            return np.load(beta_path).astype(np.float32)
        
        # Use random β for data augmentation or zero β as default
        if self.use_random_beta:
            # Random β in reasonable range [-2, 2]
            return np.random.uniform(-2, 2, 10).astype(np.float32)
        
        return np.zeros(10, dtype=np.float32)


class UpperBodyGarmentBeta10Hybrid(UpperBodyGarmentBeta10):
    """
    Dataset that directly returns the Extended Hybrid Representation with 10D β.
    
    This dataset returns:
        - garment_img: Ground truth garment image (3, H, W)
        - hybrid_img: Extended Hybrid Representation I_hybrid' (16, H, W)
        - garment_mask: Garment alpha mask (1, H, W)
    
    where I_hybrid' = I_vm ⊕ I_sdp ⊕ I_β (3 + 3 + 10 = 16 channels)
    """
    
    def __getitem__(self, index):
        garment_img, vm_img, dp_img, beta_img, mask_img = super().__getitem__(index)
        
        # Create Extended Hybrid Representation
        # I_hybrid' = I_vm (3) ⊕ I_sdp (3) ⊕ I_β (10) = (16, H, W)
        hybrid_img = torch.cat([vm_img, dp_img, beta_img], dim=0)
        
        return garment_img, hybrid_img, mask_img
