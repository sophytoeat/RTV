#!/bin/bash

# Training script for Extended Hybrid Representation with β Parameters
# 
# This trains the Garment Synthesis (GS) network using:
#     I_hybrid' = I_vm ⊕ I_sdp ⊕ I_β
#
# Input channels: 16 (3 + 3 + 10)
# Output channels: 4 (RGB + Alpha)

# Configuration
DATASET_PATH="./PerGarmentDatasets/your_garment_dataset"
NAME="upperbody_beta_model"
BATCH_SIZE=4
IMG_SIZE=512
NITER=50
NITER_DECAY=50
DISPLAY_FREQ=100
PRINT_FREQ=100
SAVE_LATEST_FREQ=1000
SAVE_EPOCH_FREQ=5

# GPU settings
export CUDA_VISIBLE_DEVICES=0

# Run training
python Training/upperbody_training_beta.py \
    --dataset_path ${DATASET_PATH} \
    --name ${NAME} \
    --batchSize ${BATCH_SIZE} \
    --img_size ${IMG_SIZE} \
    --niter ${NITER} \
    --niter_decay ${NITER_DECAY} \
    --display_freq ${DISPLAY_FREQ} \
    --print_freq ${PRINT_FREQ} \
    --save_latest_freq ${SAVE_LATEST_FREQ} \
    --save_epoch_freq ${SAVE_EPOCH_FREQ} \
    --input_nc 16 \
    --output_nc 4 \
    --model pix2pixHD_RGBA \
    --netG global \
    --ngf 64 \
    --n_downsample_global 3 \
    --n_blocks_global 9 \
    --gpu_ids 0

echo "Training completed!"
