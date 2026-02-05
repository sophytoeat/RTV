# Extended Hybrid Representation (I_hybrid') - Channel Concatenation Method

## Overview

This implementation adds SMPL β parameter support to the Virtual Try-On system for enhanced body shape adaptability.

### Mathematical Formulation

The Extended Hybrid Representation is defined as:

$$\mathbf{I}_{hybrid}' = \mathbf{I}_{vm} \oplus \mathbf{I}_{sdp} \oplus \mathbf{I}_{\beta}$$

Where:
- **I_vm** (3 channels): Virtual Measurement Garment
- **I_sdp** (3 channels): Simplified DensePose map
- **I_β** (10 channels): Body Shape Feature Map

**Total: 16 input channels**

## File Structure

```
RTV/
├── util/
│   └── beta_utils.py                    # β parameter utilities
├── Datasets/
│   └── upperbody_garment/
│       ├── upperbody_garment.py         # Original dataset
│       └── upperbody_garment_beta.py    # Extended dataset with β
├── Training/
│   ├── upperbody_training.py            # Original training
│   ├── upperbody_training_beta.py       # Extended training
│   └── upperbody_training_beta.sh       # Training script
├── Inference/
│   ├── upperbody_inference.py           # Original inference
│   └── upperbody_inference_beta.py      # Extended inference
├── VITON/
│   ├── viton_upperbody.py               # Original frame processor
│   └── viton_upperbody_beta.py          # Extended frame processor
├── DatasetGeneration/
│   ├── upperbody_dataset_generation.py      # Original dataset gen
│   └── upperbody_dataset_generation_beta.py # Extended with β
└── docs/
    └── extended_hybrid_representation.md    # This file
```

## Usage

### 1. Dataset Generation (with β parameters)

```bash
python DatasetGeneration/upperbody_dataset_generation_beta.py \
    --video_path <input_video.mp4> \
    --mask_dir <mask_directory> \
    --dataset_name <output_name>
```

This generates:
- `{frame_id}_garment.jpg` - Ground truth garment
- `{frame_id}_vm.jpg` - Virtual Measurement Garment
- `{frame_id}_mask.png` - Garment mask
- `{frame_id}_iuv.npy` - DensePose IUV
- `{frame_id}_trans2roi.npy` - Transformation matrix
- `{frame_id}_beta.npy` - **SMPL β parameters** (NEW)
- `beta_params.json` - All β parameters in JSON

### 2. Training

```bash
# Option 1: Use the shell script
bash Training/upperbody_training_beta.sh

# Option 2: Direct Python call
python Training/upperbody_training_beta.py \
    --dataset_path ./PerGarmentDatasets/your_dataset \
    --name my_beta_model \
    --input_nc 16 \
    --output_nc 4 \
    --batchSize 4 \
    --niter 50 \
    --niter_decay 50
```

### 3. Inference

```bash
# With β estimation (automatic)
python Inference/upperbody_inference_beta.py \
    --input_video input.mp4 \
    --garment_name my_beta_model \
    --use_beta \
    --output output.mp4

# With manual β parameters
python Inference/upperbody_inference_beta.py \
    --input_video input.mp4 \
    --garment_name my_beta_model \
    --use_beta \
    --beta 0.5 -0.3 0.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
    --output output.mp4
```

## API Reference

### beta_utils.py

```python
from util.beta_utils import (
    beta_to_feature_map,      # β → numpy feature map (H, W, 10)
    beta_to_tensor,           # β → PyTorch tensor (B, 10, H, W)
    create_hybrid_input,      # Create I_hybrid' from components
    estimate_beta_from_joints, # Estimate β from 2D joints
    BetaFeatureGenerator      # Cached β feature generator
)
```

### FrameProcessorBeta

```python
from VITON.viton_upperbody_beta import FrameProcessorBeta

# Create processor
processor = FrameProcessorBeta(
    garment_name_list=['garment_model'],
    ckpt_dir='./checkpoints/',
    use_beta=True  # Enable Extended Hybrid Representation
)

# Process frame
result = processor(frame, external_beta=None)

# Manual β control
processor.set_beta([0.5, -0.3, 0.1, 0, 0, 0, 0, 0, 0, 0])

# Get current β
current_beta = processor.get_current_beta()
```

## β Parameter Meaning

The 10-dimensional SMPL β parameter controls body shape:

| Index | Primary Effect |
|-------|----------------|
| β[0]  | Overall body volume/weight |
| β[1]  | Height |
| β[2]  | Body proportion (torso/legs) |
| β[3]  | Shoulder width |
| β[4]  | Hip width |
| β[5-9] | Fine-grained shape details |

Typical range: [-3, 3] (normalized to [-1, 1] in the network)

## Network Architecture

```
Input: I_hybrid' (B, 16, H, W)
       ├── I_vm   (B, 3, H, W)  - Virtual Measurement
       ├── I_sdp  (B, 3, H, W)  - Simplified DensePose
       └── I_β    (B, 10, H, W) - Body Shape Feature Map

Encoder → ResNet Blocks → Decoder

Output: (B, 4, H, W)
       ├── RGB   (B, 3, H, W)  - Garment colors
       └── Alpha (B, 1, H, W)  - Garment mask
```

## Effect

By incorporating β parameters, the Garment Synthesis network learns to:

1. **Adapt fit** to different body volumes (slim ↔ fat)
2. **Adjust proportions** based on body height and limb ratios
3. **Generate coherent drape** that respects body shape

This enables a single model to produce appropriate results across diverse body types.
