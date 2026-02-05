#!/bin/bash
# Setup script for PCA training environment

echo "Setting up virtual environment for PCA training..."

# Create virtual environment
python3 -m venv venv_pca

# Activate virtual environment
source venv_pca/bin/activate

# Install required packages
pip install numpy scikit-learn

echo "✓ Virtual environment created and packages installed"
echo ""
echo "To use the environment, run:"
echo "  source venv_pca/bin/activate"
echo "  python3 util/train_pca_beta.py --dataset_paths ./PerGarmentDatasets/f_fat_gap_beta ./PerGarmentDatasets/f_no_gap_beta --output ./pca_beta_model.pkl"
