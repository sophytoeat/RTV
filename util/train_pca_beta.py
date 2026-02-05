"""
Train PCA model for β parameter compression.

This script trains a PCA model from β parameters in dataset(s) to compress
10-dimensional β to 1 dimension while preserving maximum information.
"""

import numpy as np
import os
import json
import sys
from sklearn.decomposition import PCA
import pickle

def collect_betas_from_dataset(dataset_path):
    """Collect all β parameters from a dataset."""
    betas = []
    
    # Try loading from beta_params.json
    beta_file = os.path.join(dataset_path, 'beta_params.json')
    if os.path.exists(beta_file):
        print(f"  Loading from {beta_file}...")
        with open(beta_file, 'r') as f:
            beta_dict = json.load(f)
            for frame_id, beta in beta_dict.items():
                beta_array = np.array(beta, dtype=np.float32)
                if len(beta_array) >= 10:
                    betas.append(beta_array[:10])
        print(f"  Loaded {len(betas)} β parameters from JSON")
    
    # Also try loading from individual .npy files
    if len(betas) == 0:
        print(f"  Loading from individual .npy files...")
        count = 0
        for filename in sorted(os.listdir(dataset_path)):
            if filename.endswith('_beta.npy'):
                beta_path = os.path.join(dataset_path, filename)
                try:
                    beta = np.load(beta_path).astype(np.float32)
                    if len(beta) >= 10:
                        betas.append(beta[:10])
                        count += 1
                except Exception as e:
                    print(f"    Warning: Could not load {filename}: {e}")
        print(f"  Loaded {count} β parameters from .npy files")
    
    return betas


def train_pca_from_datasets(dataset_paths, output_path=None):
    """
    Train PCA model from β parameters in dataset(s).
    
    Args:
        dataset_paths: List of dataset directory paths or single path string
        output_path: Path to save PCA model (optional)
    
    Returns:
        Trained PCA model (sklearn.decomposition.PCA)
    """
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]
    
    all_betas = []
    
    # Collect all β parameters from datasets
    print(f"Collecting β parameters from {len(dataset_paths)} dataset(s)...")
    for dataset_path in dataset_paths:
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset path does not exist: {dataset_path}")
            continue
        
        print(f"\nProcessing: {dataset_path}")
        betas = collect_betas_from_dataset(dataset_path)
        all_betas.extend(betas)
        print(f"  Total collected so far: {len(all_betas)}")
    
    if len(all_betas) == 0:
        raise ValueError(f"No β parameters found in datasets: {dataset_paths}")
    
    # Convert to numpy array
    all_betas = np.array(all_betas, dtype=np.float32)
    print(f"\n✓ Collected {len(all_betas)} β parameters total")
    print(f"  Shape: {all_betas.shape}")
    print(f"  Mean: {all_betas.mean(axis=0)[:5]}... (first 5 dims)")
    print(f"  Std:  {all_betas.std(axis=0)[:5]}... (first 5 dims)")
    
    # Train PCA with 1 component
    print("\nTraining PCA with 1 component...")
    pca = PCA(n_components=1)
    pca.fit(all_betas)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_[0]
    print(f"\n✓ PCA training completed!")
    print(f"  - Explained variance: {explained_variance*100:.2f}%")
    print(f"  - Principal component shape: {pca.components_.shape}")
    print(f"  - Principal component (all 10 values):")
    for i, val in enumerate(pca.components_[0]):
        print(f"      β[{i}]: {val:7.4f}")
    
    # Save PCA model if output path is provided
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(pca, f)
        print(f"\n✓ PCA model saved to: {output_path}")
    
    return pca


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PCA model for β compression')
    parser.add_argument('--dataset_paths', type=str, nargs='+', required=True,
                       help='Dataset directory paths (space-separated)')
    parser.add_argument('--output', type=str, default='./pca_beta_model.pkl',
                       help='Output path for PCA model (default: ./pca_beta_model.pkl)')
    
    args = parser.parse_args()
    
    try:
        pca_model = train_pca_from_datasets(args.dataset_paths, args.output)
        print("\n" + "="*60)
        print("PCA model training completed successfully!")
        print("="*60)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
