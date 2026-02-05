# Virtual Try-On Training with 10-Dimensional SMPL β Parameters

## Technical Documentation

---

## 1. Overview

This system utilizes 10-dimensional SMPL (Skinned Multi-Person Linear Model) β parameters to train a garment generation network that considers body shape information.

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Overall Processing Flow                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【Phase 1: Dataset Generation】                                        │
│  ┌─────────────────┐                                                   │
│  │ Input Video     │                                                   │
│  │ (f_fatsuit.MP4) │                                                   │
│  └────────┬────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐     ┌─────────────────┐                          │
│  │ BEV (SMPL Est.) │────→│ β Parameter     │                          │
│  │ fix_body=False  │     │ Extraction (10D)│                          │
│  └─────────────────┘     └────────┬────────┘                          │
│                                   │                                     │
│                                   ▼                                     │
│                          ┌─────────────────┐                           │
│                          │ {frame}_beta.npy│                           │
│                          │ beta_params.json│                           │
│                          └─────────────────┘                           │
│                                                                         │
│  【Phase 2: Training】                                                  │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ UpperBodyGarmentBeta10 Dataset Class                        │       │
│  │                                                             │       │
│  │  Input:                                                     │       │
│  │  ├─ I_vm (3ch): Virtual Measurement                        │       │
│  │  ├─ I_sdp (3ch): Simplified DensePose                      │       │
│  │  └─ I_β (10ch): 10-Dimensional β Feature Map               │       │
│  │                                                             │       │
│  │  Output:                                                    │       │
│  │  ├─ RGB (3ch): Generated Garment Image                     │       │
│  │  └─ Alpha (1ch): Garment Mask                              │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                           │                                             │
│                           ▼                                             │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ pix2pixHD_RGBA (GAN)                                        │       │
│  │ input_nc=16, output_nc=4                                    │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Phase 1: β Parameter Extraction

### 2.1 Extraction Script

**File:** `DatasetGeneration/extract_beta_only.py`

```python
"""
Extract SMPL β parameters from video using BEV

Script to extract actual β parameters from BEV.
By setting fix_body=False, we obtain the actual body shape parameters estimated by BEV.
"""

def extract_betas_from_video(video_path, output_dir, max_frames=None, skip_frames=1):
    # IMPORTANT: Initialize BEV with fix_body=False
    # If fix_body=True, β[0:4] will be overwritten to 0
    smpl_regressor = SMPL_Regressor(use_bev=True, fix_body=False)
    
    for frame_idx in range(total_frames):
        raw_image = video_loader.cap()
        
        # Execute SMPL regression
        result = smpl_regressor.forward(raw_image, roi=True, size=1.45, roi_img_size=1024)
        
        if result is not None:
            smpl_param = result[0]
            
            # Extract β parameters (10 dimensions)
            beta = extract_beta_from_smpl_param(smpl_param)
            
            # Save as individual npy file
            np.save(f"{output_dir}/{frame_id}_beta.npy", beta[:10])
```

### 2.2 Importance of BEV Configuration

**File:** `SMPL/my_bev.py`

```python
class MyBEV(BEV):
    def __init__(self, settings, fix_body=False):
        super(MyBEV, self).__init__(settings)
        self.fix_body = fix_body

    def process_normal_image(self, image, signal_ID):
        outputs, image_pad_info = self.single_image_forward(image)
        
        # If fix_body=True, β values are overwritten (we want to avoid this)
        if self.fix_body:
            outputs['smpl_betas'][:,10] = 0
            outputs['smpl_betas'][:, 0:4] = 0  # ← β[0]~β[3] are zeroed out
        
        # If fix_body=False, BEV's estimated values are used as-is
```

### 2.3 Extraction Command

```bash
cd /home/sophytoeat/project/RTV
python DatasetGeneration/extract_beta_only.py \
    --video_path ./video/f_fatsuit_gap.MP4 \
    --output_dir ./extracted_betas/f_fatsuit_gap
```

### 2.4 Extraction Results

```
Example of extracted β parameters:

Frame 0:   β = [3.12, 0.31, 0.89, -0.39, 0.10, 0.10, -0.12, -0.10, 0.07, -0.12]
Frame 100: β = [3.54, 0.41, 0.91, -0.54, 0.06, -0.13, -0.19, -0.03, 0.07, -0.18]
Frame 500: β = [2.36, 0.36, 0.79, -0.35, 0.04, 0.11, -0.09, -0.13, 0.09, -0.13]

Meaning of each dimension:
  β[0]: Body thickness/weight (high value ~3.0 when wearing fatsuit)
  β[1]: Height
  β[2]: Torso length
  β[3]: Shoulder width
  β[4]: Hip position
  β[5]: Chest size
  β[6]: Arm length
  β[7]: Leg length
  β[8]: Abdomen
  β[9]: Other body shape features
```

---

## 3. Phase 2: Dataset Class

### 3.1 Class Structure

**File:** `Datasets/upperbody_garment/upperbody_garment_beta10.py`

```python
class UpperBodyGarmentBeta10(UpperBodyGarment):
    """
    Extended dataset class including 10-dimensional SMPL β parameters
    
    Returns:
        - garment_img: Ground truth garment image (3, H, W)
        - vm_img: Virtual Measurement (3, H, W)
        - dp_img: Simplified DensePose (3, H, W)
        - beta_img: 10-dimensional β feature map (10, H, W)
        - garment_mask: Garment mask (1, H, W)
    """
```

### 3.2 Initialization

```python
def __init__(self, path, img_size=512, use_random_beta=False):
    super().__init__(path, img_size)  # Parent class initialization
    self.use_random_beta = use_random_beta
    self.img_size = img_size
    
    # Batch load β from beta_params.json (for efficiency)
    self.beta_dict = {}
    beta_file = os.path.join(self.img_dir, 'beta_params.json')
    if os.path.exists(beta_file):
        with open(beta_file, 'r') as f:
            self.beta_dict = json.load(f)
```

**Key Points:**
- Loading `beta_params.json` at initialization reduces I/O during `__getitem__` calls
- Setting `use_random_beta=True` enables random β generation for data augmentation

### 3.3 Data Retrieval `__getitem__`

```python
def __getitem__(self, index):
    # ===== 1. File Path Identification =====
    garment_path = self.image_list[index]
    frame_id = os.path.basename(garment_path).split('_')[0]  # "00000"
    
    # ===== 2. Load Each Image File =====
    # 2.1 Garment image (ground truth)
    garment_img = np.array(Image.open(garment_path))
    trans2roi = np.load(trans2roi_path)
    inv_trans = get_inverse_trans(trans2roi)
    garment_img = cv2.warpAffine(garment_img, inv_trans, (raw_w, raw_h), ...)
    
    # 2.2 Virtual Measurement (SMPL mesh rendering)
    vm_img = np.array(Image.open(vm_path))
    
    # 2.3 Mask
    mask_img = np.array(Image.open(mask_path))
    
    # 2.4 DensePose → Simplified DensePose
    IUV = np.load(iuv_path)
    dp_img = IUV2SDP(IUV)  # Convert IUV format to SDP format
    
    # 2.5 β parameters (10 dimensions)
    beta = self._get_beta(frame_id)
    
    # ===== 3. Data Augmentation (Random Affine Transform) =====
    new_trans2roi = self.randomaffine(trans2roi)
    
    # Apply same transform to all images (to maintain spatial relationships)
    roi_garment_img = cv2.warpAffine(garment_img, new_trans2roi, (1024, 1024), ...)
    roi_dp_img = cv2.warpAffine(dp_img, new_trans2roi, (1024, 1024), ...)
    roi_vm_img = cv2.warpAffine(vm_img, new_trans2roi, (1024, 1024), ...)
    roi_mask_img = cv2.warpAffine(mask_img, new_trans2roi, (1024, 1024), ...)
    
    # ===== 4. Tensor Conversion =====
    # 4.1 Convert images to PyTorch tensors
    if self.transform is not None:
        roi_garment_img = self.transform(PIL.Image.fromarray(roi_garment_img))
        roi_vm_img = self.transform(PIL.Image.fromarray(roi_vm_img))
        roi_dp_img = self.transform(PIL.Image.fromarray(roi_dp_img))
        roi_mask_img = self.transform(PIL.Image.fromarray(roi_mask_img))
    
    # 4.2 Convert β to 10-channel spatial tensor
    beta_tensor = beta_to_tensor(
        beta,           # [β0, β1, ..., β9] 10-dimensional vector
        self.img_size,  # H = 512
        self.img_size,  # W = 512
        device='cpu',
    ).squeeze(0)  # (1, 10, H, W) → (10, H, W)
    
    # ===== 5. Return Values =====
    return (self._normalize(roi_garment_img),  # (3, 512, 512) GT RGB
            self._normalize(roi_vm_img),       # (3, 512, 512) VM
            self._normalize(roi_dp_img),       # (3, 512, 512) DensePose
            beta_tensor,                       # (10, 512, 512) β feature map
            roi_mask_img)                      # (1, 512, 512) Mask
```

### 3.4 β Retrieval `_get_beta`

```python
def _get_beta(self, frame_id):
    """Retrieve β parameters (with priority ordering)"""
    
    # Priority 1: Retrieve from beta_params.json (fastest)
    if frame_id in self.beta_dict:
        return np.array(self.beta_dict[frame_id], dtype=np.float32)
    
    # Priority 2: Retrieve from individual .npy file
    beta_path = os.path.join(self.img_dir, frame_id + '_beta.npy')
    if os.path.exists(beta_path):
        return np.load(beta_path).astype(np.float32)
    
    # Priority 3: Random β (for data augmentation)
    if self.use_random_beta:
        return np.random.uniform(-2, 2, 10).astype(np.float32)
    
    # Default: Zero vector
    return np.zeros(10, dtype=np.float32)
```

### 3.5 β Spatial Expansion `beta_to_tensor`

**File:** `util/beta_utils.py`

```python
def beta_to_tensor(beta, height, width, device='cuda'):
    """
    Convert 10-dimensional β vector to spatial tensor
    
    Processing:
    1. Extract β[0:10]
    2. Normalize by β / 3.0 → Clip to [-1, 1] range
    3. Expand each β to H×W plane
    
    Input: beta = [3.12, 0.31, 0.89, ...] (10 dimensions)
    Output: (1, 10, H, W) tensor
            Channel 0: All pixels have value β[0]/3.0
            Channel 1: All pixels have value β[1]/3.0
            ...
    """
    beta = np.array(beta, dtype=np.float32).reshape(1, -1)[:, :10]
    
    # Normalization: β values are typically in [-3, 3] range, divide by 3 to get [-1, 1]
    beta_normalized = np.clip(beta / 3.0, -1.0, 1.0)
    
    # Spatial expansion: (1, 10) → (1, 10, H, W)
    beta_tensor = torch.from_numpy(beta_normalized).float()
    I_beta = beta_tensor.view(1, 10, 1, 1).expand(1, 10, height, width).clone()
    
    return I_beta
```

---

## 4. Phase 3: Training

### 4.1 Training Script

**File:** `Training/upperbody_training_beta10.py`

```python
def main():
    opt = TrainOptions().parse()
    
    # ===== Network Configuration =====
    opt.input_nc = 16   # I_vm(3) + I_sdp(3) + I_β(10) = 16 channels
    opt.output_nc = 4   # RGB(3) + Alpha(1) = 4 channels
    opt.model = 'pix2pixHD_RGBA'
    
    # ===== Dataset Loading =====
    dataset = UpperBodyGarmentBeta10(
        path_list[0], 
        img_size=opt.img_size, 
        use_random_beta=False
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # ===== Model Creation =====
    model = create_model(opt)  # pix2pixHD (GAN)
    
    # ===== Training Loop =====
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        for i, data in enumerate(dataloader):
            # Unpack data
            garment_img, vm_img, dp_img, beta_img, garment_mask = data
            
            # Transfer to GPU
            garment_img = garment_img.cuda()
            vm_img = vm_img.cuda()
            dp_img = dp_img.cuda()
            beta_img = beta_img.cuda()
            garment_mask = garment_mask.cuda()
            
            # ===== Create Input Tensor =====
            # Extended Hybrid Representation (16 channels)
            input_img = torch.cat([vm_img, dp_img, beta_img], dim=1)  # (B, 16, H, W)
            
            # Ground truth tensor (4 channels)
            gt_image = torch.cat([garment_img, garment_mask], dim=1)  # (B, 4, H, W)
            
            # ===== Forward Pass =====
            losses, generated = model(input_img, gt_image, infer=save_fake)
            
            # Loss calculation
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict['G_VGG']
            
            # ===== Backward Pass =====
            # Generator update
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            
            # Discriminator update
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
```

### 4.2 Input/Output Details

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Generator (pix2pixHD_RGBA)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: input_img (B, 16, H, W)                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Channel  0-2:  I_vm  (Virtual Measurement)                      │   │
│  │                Rendered SMPL mesh image                         │   │
│  │                Body silhouette and pose information             │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ Channel  3-5:  I_sdp (Simplified DensePose)                     │   │
│  │                Body surface UV coordinate information           │   │
│  │                Body part and orientation information            │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ Channel  6-15: I_β   (10-Dimensional β Feature Map)             │   │
│  │                ch6:  β[0] Body thickness/weight                 │   │
│  │                ch7:  β[1] Height                                │   │
│  │                ch8:  β[2] Torso length                          │   │
│  │                ch9:  β[3] Shoulder width                        │   │
│  │                ch10: β[4] Hip position                          │   │
│  │                ch11: β[5] Chest size                            │   │
│  │                ch12: β[6] Arm length                            │   │
│  │                ch13: β[7] Leg length                            │   │
│  │                ch14: β[8] Abdomen                               │   │
│  │                ch15: β[9] Other features                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                   │                                     │
│                                   ▼                                     │
│                          Neural Network                                 │
│                          (pix2pixHD Generator)                          │
│                                   │                                     │
│                                   ▼                                     │
│  Output: generated (B, 4, H, W)                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Channel 0-2: RGB                                                │   │
│  │              Generated garment image                            │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ Channel 3:   Alpha                                              │   │
│  │              Garment mask (indicates garment region)            │   │
│  │              White = garment area, Black = background           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Loss Functions

| Loss | Description | Role |
|------|-------------|------|
| `G_GAN` | Generator adversarial loss | Fool the discriminator |
| `G_GAN_Feat` | Feature matching loss | Match intermediate features |
| `G_VGG` | VGG perceptual loss | Similarity close to human perception |
| `D_fake` | Discriminator fake image judgment | Classify generated images as fake |
| `D_real` | Discriminator real image judgment | Classify ground truth as real |

### 4.4 Training Command

```bash
cd /home/sophytoeat/project/RTV
python Training/upperbody_training_beta10.py \
    --name f_fat_gap_beta10 \
    --dataset_path ./PerGarmentDatasets/f_fat_gap_beta \
    --batchSize 4 \
    --niter 50 \
    --niter_decay 50 \
    --save_epoch_freq 10
```

---

## 5. Comparison with Previous Methods

| Item | Previous Method (No β) |  10-Dimensional β (This Method) |
|------|----------------------|-------------------------------|
| Input Channels | 6 | **16** |
| Body Shape Info | None | **All 10 dimensions** |
| β Values | - | **BEV estimated values** |
| Body Adaptability | Low | **High** |

---

## 6. File Structure

```
RTV/
├── DatasetGeneration/
│   ├── extract_beta_only.py          # β parameter extraction script
│   └── upperbody_dataset_generation_beta.py  # Complete dataset generation
│
├── Datasets/upperbody_garment/
│   ├── upperbody_garment.py          # Base class
│   ├── upperbody_garment_beta.py     # 1-dimensional β version
│   └── upperbody_garment_beta10.py   # 10-dimensional β version ★
│
├── Training/
│   ├── upperbody_training.py         # Original training
│   ├── upperbody_training_beta.py    # 1-dimensional β version
│   └── upperbody_training_beta10.py  # 10-dimensional β version ★
│
├── util/
│   └── beta_utils.py                 # β conversion utilities
│
├── SMPL/
│   ├── my_bev.py                     # BEV wrapper (fix_body setting)
│   └── smpl_regressor.py             # SMPL regression class
│
├── PerGarmentDatasets/
│   └── f_fat_gap_beta/               # Dataset
│       ├── 00000_garment.jpg         # Garment image
│       ├── 00000_vm.jpg              # Virtual Measurement
│       ├── 00000_mask.png            # Mask
│       ├── 00000_iuv.npy             # DensePose
│       ├── 00000_beta.npy            # 10-dimensional β parameters ★
│       ├── 00000_trans2roi.npy       # ROI transformation matrix
│       └── beta_params.json          # Batch β file ★
│
└── extracted_betas/
    └── f_fatsuit_gap/                # Extracted β
        ├── 00000_beta.npy
        ├── beta_params.json
        └── beta_stats.json           # Statistics
```

---

## 7. Experimental Results

### Statistics of Extracted β Parameters

| Dimension | Meaning | Mean | Std Dev | Min | Max |
|-----------|---------|------|---------|-----|-----|
| β[0] | Body thickness | **2.73** | 1.42 | -1.03 | 5.95 |
| β[1] | Height | 0.29 | 0.17 | -0.81 | 0.94 |
| β[2] | Torso length | 0.82 | 0.20 | 0.14 | 1.43 |
| β[3] | Shoulder width | -0.29 | 0.32 | -1.05 | 2.77 |
| β[4] | Hip position | 0.05 | 0.07 | -0.08 | 0.27 |
| β[5] | Chest size | 0.08 | 0.13 | -0.25 | 0.60 |
| β[6] | Arm length | -0.06 | 0.09 | -0.33 | 0.17 |
| β[7] | Leg length | -0.10 | 0.04 | -0.19 | 0.01 |
| β[8] | Abdomen | 0.05 | 0.04 | -0.09 | 0.25 |
| β[9] | Other features | -0.10 | 0.06 | -0.32 | 0.03 |

**Observation:** Since the data involves fatsuit wearing, β[0] (body thickness) shows high values (mean 2.73).

---

## 8. Summary

This system provides the following technical contributions:

1. **Real β Parameter Extraction from BEV**: By setting `fix_body=False`, we obtain the actual body shape parameters estimated by BEV

2. **10-Dimensional β Feature Map**: All 10 dimensions, conveying detailed body shape information to the network

3. **Spatial Feature Map Conversion**: Converts β vector to (10, H, W) spatial tensor, making it processable by CNNs

4. **16-Channel Input**: Extended Hybrid Representation with I_vm(3) + I_sdp(3) + I_β(10) = 16 channels

---

## 9. Data Augmentation

The dataset class applies random affine transformations to training images for better generalization:

```python
self.randomaffine = RandomAffineMatrix(
    degrees=20,           # Rotation: ±20 degrees
    translate=(0.2, 0.2), # Translation: ±20% of image size
    scale=(0.8, 1.5),     # Scale: 80% to 150%
    shear=(-5, 5, -5, 5)  # Shear: ±5 degrees
)
```

**Important:** While visual inputs (garment, VM, DensePose, mask) undergo these transformations, the 10-dimensional β parameters remain unchanged as they represent body shape information independent of spatial transformations.

---

## 10. Key Implementation Notes

### β Normalization Strategy

β parameters from BEV typically range from [-3, 3]. We normalize them to [-1, 1] for neural network compatibility:

```python
beta_normalized = np.clip(beta / 3.0, -1.0, 1.0)
```

### Priority-Based β Loading

The dataset class implements a priority system for loading β values:

1. **Primary**: `beta_params.json` (pre-aggregated, fastest)
2. **Secondary**: Individual `{frame}_beta.npy` files
3. **Fallback**: Random β (if `use_random_beta=True`) or zero vector

This design balances flexibility with performance.

---

*Document generated for RTV Virtual Try-On System*
*Last updated: 2025*
