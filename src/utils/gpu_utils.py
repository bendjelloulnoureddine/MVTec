import torch
import numpy as np
import cv2

def compute_embedding_stats_gpu(embeddings_tensor):
    """
    Compute embedding statistics using GPU for speed
    """
    N, C, H, W = embeddings_tensor.shape
    device = embeddings_tensor.device
    
    # Reshape to (N, H*W, C)
    embeddings = embeddings_tensor.permute(0, 2, 3, 1).reshape(N, -1, C)
    
    # Compute mean
    mean = embeddings.mean(dim=0)  # (H*W, C)
    
    # Compute covariance matrices for each spatial position
    cov_matrices = []
    for i in range(H * W):
        centered = embeddings[:, i, :] - mean[i]  # (N, C)
        cov = torch.cov(centered.T)  # (C, C)
        cov_matrices.append(cov)
    
    cov_tensor = torch.stack(cov_matrices, dim=0)  # (H*W, C, C)
    
    return mean.reshape(H, W, C), cov_tensor.reshape(H, W, C, C)

def mahalanobis_map_gpu(test_embed, mean, cov_inv):
    """
    Compute Mahalanobis distance map using GPU
    """
    H, W, C = mean.shape
    device = mean.device
    
    # Reshape test embedding
    test_embed = test_embed.permute(0, 2, 3, 1)  # (B, H, W, C)
    
    # Compute differences
    diff = test_embed[0] - mean  # (H, W, C)
    
    # Compute Mahalanobis distance for each pixel
    scores = torch.zeros((H, W), device=device)
    for i in range(H):
        for j in range(W):
            d = diff[i, j]  # (C,)
            # Mahalanobis distance: sqrt(d^T * cov_inv * d)
            mahal_dist = torch.dot(d, torch.matmul(cov_inv[i, j], d))
            # Clamp to avoid negative values and NaN
            mahal_dist = torch.clamp(mahal_dist, min=1e-8)
            scores[i, j] = torch.sqrt(mahal_dist)
    
    # Convert to numpy and resize
    scores_np = scores.cpu().numpy()
    # Handle NaN values
    scores_np = np.nan_to_num(scores_np, nan=0.0, posinf=1.0, neginf=0.0)
    scores_resized = cv2.resize(scores_np, (256, 256))
    
    return scores_resized

def ensure_gpu_tensor(tensor):
    """Ensure tensor is on GPU if available"""
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

def clear_gpu_cache():
    """Clear GPU cache to free memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()