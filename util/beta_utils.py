"""
Body Shape Feature Map Utilities for Extended Hybrid Representation

This module provides utilities to convert SMPL β parameters into spatial feature maps
for channel concatenation with I_vm and I_sdp.

The 10-dimensional β vector represents body shape parameters:
- β[0]: Overall body volume/weight
- β[1]: Height
- β[2-9]: Body proportions (limb lengths, torso shape, etc.)
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
import os
import json


def beta_to_feature_map(beta: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Convert SMPL β parameters to a spatial Body Shape Feature Map (I_β).
    
    The β vector is spatially expanded by replicating each of the 10 values
    across the entire H×W spatial resolution, resulting in a (10, H, W) feature map.
    
    Args:
        beta: SMPL shape parameters, shape (10,)
        height: Target spatial height
        width: Target spatial width
    
    Returns:
        Body Shape Feature Map (I_β), shape (H, W, 10) in HWC format
    """
    if beta is None:
        beta = np.zeros(10, dtype=np.float32)
    
    beta = np.array(beta, dtype=np.float32).flatten()
    
    # Ensure beta has exactly 10 dimensions
    if len(beta) < 10:
        beta = np.pad(beta, (0, 10 - len(beta)), mode='constant', constant_values=0)
    elif len(beta) > 10:
        beta = beta[:10]
    
    # Normalize β to [-1, 1] range (typical SMPL β values are in range [-3, 3])
    # This normalization helps with network training stability
    beta_normalized = np.clip(beta / 3.0, -1.0, 1.0)
    
    # Spatial expansion: (10,) -> (H, W, 10)
    # Each channel contains the same value replicated across all spatial locations
    I_beta = np.zeros((height, width, 10), dtype=np.float32)
    for i in range(10):
        I_beta[:, :, i] = beta_normalized[i]
    
    return I_beta


def beta_to_tensor(beta: np.ndarray, height: int, width: int, device: str = 'cuda') -> torch.Tensor:
    """
    Convert SMPL β parameters to a PyTorch tensor feature map.
    
    Args:
        beta: SMPL shape parameters, shape (10,) or (B, 10)
        height: Target spatial height
        width: Target spatial width
        device: Target device ('cuda' or 'cpu')
    
    Returns:
        Body Shape Feature Map tensor, shape (1, 10, H, W) or (B, 10, H, W)
    """
    if beta is None:
        beta = np.zeros(10, dtype=np.float32)
    
    beta = np.array(beta, dtype=np.float32)
    
    # Handle batched input
    if beta.ndim == 1:
        beta = beta.reshape(1, -1)
    
    batch_size = beta.shape[0]
    
    # Ensure 10 dimensions
    if beta.shape[1] < 10:
        beta = np.pad(beta, ((0, 0), (0, 10 - beta.shape[1])), mode='constant', constant_values=0)
    elif beta.shape[1] > 10:
        beta = beta[:, :10]
    
    # Normalize to [-1, 1]
    beta_normalized = np.clip(beta / 3.0, -1.0, 1.0)
    
    # Create tensor: (B, 10, H, W)
    # Convert to torch tensor first, then expand spatially
    beta_tensor = torch.from_numpy(beta_normalized).float()  # (B, 10)
    I_beta = beta_tensor.view(batch_size, 10, 1, 1).expand(batch_size, 10, height, width).clone()
    
    if device == 'cuda' and torch.cuda.is_available():
        I_beta = I_beta.cuda()
    
    return I_beta


def create_hybrid_input(vm_tensor: torch.Tensor, dp_tensor: torch.Tensor, 
                        beta: np.ndarray = None) -> torch.Tensor:
    """
    Create Extended Hybrid Representation (I_hybrid') by channel concatenation.
    
    I_hybrid' = I_vm ⊕ I_sdp ⊕ I_β
    
    Args:
        vm_tensor: Virtual Measurement Garment tensor, shape (B, 3, H, W), range [-1, 1]
        dp_tensor: Simplified DensePose tensor, shape (B, 3, H, W), range [-1, 1]
        beta: SMPL β parameters, shape (10,) or (B, 10)
    
    Returns:
        Extended Hybrid Representation tensor, shape (B, 16, H, W)
    """
    batch_size, _, height, width = vm_tensor.shape
    device = 'cuda' if vm_tensor.is_cuda else 'cpu'
    
    # Create β feature map
    beta_tensor = beta_to_tensor(beta, height, width, device)
    
    # Ensure batch size matches
    if beta_tensor.shape[0] == 1 and batch_size > 1:
        beta_tensor = beta_tensor.expand(batch_size, -1, -1, -1)
    
    # Channel concatenation: (B, 3, H, W) + (B, 3, H, W) + (B, 10, H, W) = (B, 16, H, W)
    I_hybrid_prime = torch.cat([vm_tensor, dp_tensor, beta_tensor], dim=1)
    
    return I_hybrid_prime


def extract_beta_from_smpl_param(smpl_param, mu_slim=0.27, mu_fat=0.34):
    """
    Extract β parameters from SMPL regressor output.
    
    Args:
        smpl_param: SMPL parameter dictionary from SMPL_Regressor
        mu_slim: Median ratio for slim body type (not used if β is directly available)
        mu_fat: Median ratio for fat body type (not used if β is directly available)
    
    Returns:
        β parameters, shape (10,)
    """
    import torch
    
    # Try to get β directly from SMPL parameters
    if 'smpl_betas' in smpl_param:
        betas = smpl_param['smpl_betas']
        if isinstance(betas, torch.Tensor):
            betas = betas.cpu().numpy()
        if betas.ndim > 1:
            betas = betas[0]  # Take first person if batch
        return betas.flatten()[:10].astype(np.float32)
    
    # Estimate from 2D joints
    if 'pj2d_org' in smpl_param:
        try:
            joints = smpl_param['pj2d_org']
            cam_trans = smpl_param['cam_trans']
            depth_order = torch.sort(cam_trans[:, 2].cpu(), descending=False).indices.numpy()
            J = joints[depth_order][0].cpu().numpy()
            return estimate_beta_from_joints(J, mu_slim, mu_fat)
        except Exception as e:
            print(f"Warning: Could not estimate β from joints: {e}")
    
    return np.zeros(10, dtype=np.float32)


def estimate_beta_from_joints(joints_2d: np.ndarray, 
                               mu_slim: float = 0.27, 
                               mu_fat: float = 0.34) -> np.ndarray:
    """
    Estimate a simplified β parameter from 2D joint positions.
    
    This is a heuristic estimation based on shoulder width to body height ratio.
    For accurate β parameters, use a proper SMPL regressor.
    
    Args:
        joints_2d: 2D joint positions, shape (N, 2) where N >= 16
        mu_slim: Median ratio for slim body type
        mu_fat: Median ratio for fat body type
    
    Returns:
        Estimated β parameters, shape (10,)
    """
    beta = np.zeros(10, dtype=np.float32)
    
    try:
        # Joint indices (common SMPL/COCO convention):
        # 9, 12: left/right shoulder
        # 0: head/nose
        # 15: lower body reference
        shoulder_width = np.linalg.norm(joints_2d[9] - joints_2d[12])
        body_height = np.linalg.norm(joints_2d[15] - joints_2d[0])
        
        if body_height > 1e-6:
            ratio = shoulder_width / body_height
            s_raw = (ratio - mu_slim) / max(1e-6, (mu_fat - mu_slim))
            s = np.clip(s_raw, 0.0, 1.0)
            
            # Map scalar s to β[0] (body volume)
            # s=0 -> slim (β[0] = -2), s=1 -> fat (β[0] = 2)
            beta[0] = (s - 0.5) * 4.0
    except (IndexError, ValueError):
        pass
    
    return beta


def train_pca_from_dataset(dataset_paths, output_path: str = None):
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
    for dataset_path in dataset_paths:
        beta_file = os.path.join(dataset_path, 'beta_params.json')
        if os.path.exists(beta_file):
            with open(beta_file, 'r') as f:
                beta_dict = json.load(f)
                for frame_id, beta in beta_dict.items():
                    beta_array = np.array(beta, dtype=np.float32)
                    if len(beta_array) >= 10:
                        all_betas.append(beta_array[:10])
        
        # Also try loading from individual .npy files
        if len(all_betas) == 0:
            for filename in os.listdir(dataset_path):
                if filename.endswith('_beta.npy'):
                    beta_path = os.path.join(dataset_path, filename)
                    beta = np.load(beta_path).astype(np.float32)
                    if len(beta) >= 10:
                        all_betas.append(beta[:10])
    
    if len(all_betas) == 0:
        raise ValueError(f"No β parameters found in datasets: {dataset_paths}")
    
    # Convert to numpy array
    all_betas = np.array(all_betas, dtype=np.float32)
    print(f"Collected {len(all_betas)} β parameters for PCA training")
    
    # Train PCA with 1 component
    pca = PCA(n_components=1)
    pca.fit(all_betas)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_[0]
    print(f"PCA first component explains {explained_variance*100:.2f}% of variance")
    
    # Save PCA model if output path is provided
    if output_path:
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(pca, f)
        print(f"PCA model saved to {output_path}")
    
    return pca


def load_pca_model(pca_path: str):
    """
    Load pre-trained PCA model.
    
    Args:
        pca_path: Path to saved PCA model
    
    Returns:
        Trained PCA model
    """
    import pickle
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)
    return pca


# Global PCA model cache
_pca_model = None
_pca_model_path = None


def compress_beta_to_1d(beta: np.ndarray, method: str = 'weighted', 
                        pca_model=None, pca_model_path: str = None) -> np.ndarray:
    """
    Compress 10-dimensional β parameters to 1 dimension.
    
    Args:
        beta: SMPL shape parameters, shape (10,)
        method: Compression method
            - 'first': Use only β[0] (body volume/weight)
            - 'weighted': Weighted linear combination (β[0] with higher weight)
            - 'pca': PCA-based compression (requires pre-computed PCA components)
            - 'l2_norm': L2 norm of the β vector
            - 'dot_product': Dot product with learned principal component
        pca_model: Pre-trained PCA model (for 'pca' method)
        pca_model_path: Path to saved PCA model (for 'pca' method)
    
    Returns:
        Compressed β parameter, shape (1,)
    """
    global _pca_model, _pca_model_path
    
    beta = np.array(beta, dtype=np.float32).flatten()
    
    # Ensure 10 dimensions
    if len(beta) < 10:
        beta = np.pad(beta, (0, 10 - len(beta)), mode='constant', constant_values=0)
    elif len(beta) > 10:
        beta = beta[:10]
    
    if method == 'first':
        # Simply use β[0] (body volume/weight)
        return np.array([beta[0]], dtype=np.float32)
    
    elif method == 'weighted':
        # Weighted linear combination: emphasize β[0] but include other dimensions
        # Weights based on importance: β[0] (volume) is most important
        weights = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005], dtype=np.float32)
        weights = weights / weights.sum()  # Normalize to sum to 1
        compressed = np.dot(beta, weights)
        return np.array([compressed], dtype=np.float32)
    
    elif method == 'l2_norm':
        # Use L2 norm of the β vector
        l2_norm = np.linalg.norm(beta)
        # Normalize to similar range as β[0] (roughly [-3, 3])
        # Scale by a factor to match typical β[0] range
        compressed = l2_norm * np.sign(beta[0]) if beta[0] != 0 else l2_norm
        return np.array([compressed], dtype=np.float32)
    
    elif method == 'pca' or method == 'dot_product':
        # PCA-based compression
        # Try to get PCA model
        if pca_model is not None:
            pca = pca_model
        elif pca_model_path is not None:
            if _pca_model_path != pca_model_path or _pca_model is None:
                _pca_model = load_pca_model(pca_model_path)
                _pca_model_path = pca_model_path
            pca = _pca_model
        else:
            # Try to use cached model
            if _pca_model is None:
                raise ValueError(
                    "PCA compression requires a trained PCA model. "
                    "Use train_pca_from_dataset() first or provide pca_model/pca_model_path."
                )
            pca = _pca_model
        
        # Transform using PCA (returns shape (1, 1))
        compressed = pca.transform(beta.reshape(1, -1))[0, 0]
        return np.array([compressed], dtype=np.float32)
    
    else:
        raise ValueError(f"Unknown compression method: {method}. "
                        f"Available methods: 'first', 'weighted', 'l2_norm', 'pca', 'dot_product'")


def beta_to_tensor_1d(beta: np.ndarray, height: int, width: int, 
                      device: str = 'cuda', compression_method: str = 'weighted',
                      pca_model=None, pca_model_path: str = None) -> torch.Tensor:
    """
    Convert SMPL β parameters to a 1-channel Body Shape Feature Map.
    
    This function compresses 10-dimensional β to 1 dimension before creating the feature map.
    
    Args:
        beta: SMPL shape parameters, shape (10,) or (B, 10)
        height: Target spatial height
        width: Target spatial width
        device: Target device ('cuda' or 'cpu')
        compression_method: Method to compress β from 10D to 1D
            - 'first': Use only β[0]
            - 'weighted': Weighted linear combination (default)
            - 'l2_norm': L2 norm of β vector
    
    Returns:
        Body Shape Feature Map tensor, shape (1, 1, H, W) or (B, 1, H, W)
    """
    beta = np.array(beta, dtype=np.float32)
    
    # Handle batched input
    if beta.ndim == 1:
        beta = beta.reshape(1, -1)
    
    batch_size = beta.shape[0]
    
    # Compress each β vector from 10D to 1D
    beta_1d = np.zeros((batch_size, 1), dtype=np.float32)
    for i in range(batch_size):
        beta_1d[i, 0] = compress_beta_to_1d(beta[i], method=compression_method, 
                                            pca_model=pca_model, 
                                            pca_model_path=pca_model_path)[0]
    
    # Normalize to [-1, 1] range
    beta_normalized = np.clip(beta_1d / 3.0, -1.0, 1.0)
    
    # Create tensor: (B, 1, H, W)
    beta_tensor = torch.from_numpy(beta_normalized).float()  # (B, 1)
    I_beta = beta_tensor.view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width).clone()
    
    if device == 'cuda' and torch.cuda.is_available():
        I_beta = I_beta.cuda()
    
    return I_beta


def create_hybrid_input_1d(vm_tensor: torch.Tensor, dp_tensor: torch.Tensor, 
                          beta: np.ndarray = None, compression_method: str = 'weighted',
                          pca_model=None, pca_model_path: str = None) -> torch.Tensor:
    """
    Create Extended Hybrid Representation with 1D β compression.
    
    I_hybrid' = I_vm ⊕ I_sdp ⊕ I_β (1D)
    
    Args:
        vm_tensor: Virtual Measurement Garment tensor, shape (B, 3, H, W), range [-1, 1]
        dp_tensor: Simplified DensePose tensor, shape (B, 3, H, W), range [-1, 1]
        beta: SMPL β parameters, shape (10,) or (B, 10)
        compression_method: Method to compress β from 10D to 1D
    
    Returns:
        Extended Hybrid Representation tensor, shape (B, 7, H, W)
    """
    batch_size, _, height, width = vm_tensor.shape
    device = 'cuda' if vm_tensor.is_cuda else 'cpu'
    
    # Create 1D β feature map
    beta_tensor = beta_to_tensor_1d(beta, height, width, device, compression_method,
                                   pca_model=pca_model, pca_model_path=pca_model_path)
    
    # Ensure batch size matches
    if beta_tensor.shape[0] == 1 and batch_size > 1:
        beta_tensor = beta_tensor.expand(batch_size, -1, -1, -1)
    
    # Channel concatenation: (B, 3, H, W) + (B, 3, H, W) + (B, 1, H, W) = (B, 7, H, W)
    I_hybrid_prime = torch.cat([vm_tensor, dp_tensor, beta_tensor], dim=1)
    
    return I_hybrid_prime


class BetaFeatureGenerator:
    """
    Generator for Body Shape Feature Maps with caching for efficiency.
    """
    
    def __init__(self, height: int = 512, width: int = 512, use_1d: bool = False, 
                 compression_method: str = 'weighted', pca_model=None, pca_model_path: str = None):
        self.height = height
        self.width = width
        self.use_1d = use_1d
        self.compression_method = compression_method
        self.pca_model = pca_model
        self.pca_model_path = pca_model_path
        self._cached_beta = None
        self._cached_tensor = None
    
    def __call__(self, beta: np.ndarray, device: str = 'cuda') -> torch.Tensor:
        """
        Generate Body Shape Feature Map tensor.
        
        Uses caching to avoid redundant computation when β doesn't change.
        """
        beta = np.array(beta, dtype=np.float32).flatten()[:10]
        
        # Check cache
        if self._cached_beta is not None and np.allclose(beta, self._cached_beta):
            if self._cached_tensor.device.type == device:
                return self._cached_tensor
        
        # Generate new tensor
        if self.use_1d:
            tensor = beta_to_tensor_1d(beta, self.height, self.width, device, 
                                      self.compression_method,
                                      pca_model=self.pca_model,
                                      pca_model_path=self.pca_model_path)
        else:
            tensor = beta_to_tensor(beta, self.height, self.width, device)
        
        # Update cache
        self._cached_beta = beta.copy()
        self._cached_tensor = tensor
        
        return tensor
    
    def reset_cache(self):
        """Clear the cache."""
        self._cached_beta = None
        self._cached_tensor = None
