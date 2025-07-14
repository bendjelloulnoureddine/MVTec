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

def get_gpu_memory_usage():
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0.0, 0.0

def print_gpu_memory_stats(label=""):
    """Print detailed GPU memory statistics"""
    if torch.cuda.is_available():
        allocated, reserved = get_gpu_memory_usage()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free_memory = total_memory - allocated
        
        print(f"GPU Memory Stats {label}:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Free: {free_memory:.2f} GB")
        print(f"  Total: {total_memory:.2f} GB")
        print(f"  Utilization: {(allocated/total_memory)*100:.1f}%")
    else:
        print("GPU not available")

def monitor_memory_usage(func):
    """Decorator to monitor memory usage of a function"""
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_allocated, start_reserved = get_gpu_memory_usage()
            
            result = func(*args, **kwargs)
            
            torch.cuda.synchronize()
            end_allocated, end_reserved = get_gpu_memory_usage()
            
            print(f"Memory usage for {func.__name__}:")
            print(f"  Before: {start_allocated:.2f} GB allocated, {start_reserved:.2f} GB reserved")
            print(f"  After: {end_allocated:.2f} GB allocated, {end_reserved:.2f} GB reserved")
            print(f"  Delta: {end_allocated - start_allocated:.2f} GB allocated, {end_reserved - start_reserved:.2f} GB reserved")
            
            return result
        else:
            return func(*args, **kwargs)
    return wrapper

def check_memory_availability(required_gb=1.0):
    """Check if enough GPU memory is available"""
    if torch.cuda.is_available():
        allocated, reserved = get_gpu_memory_usage()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free_memory = total_memory - allocated
        
        if free_memory < required_gb:
            print(f"Warning: Only {free_memory:.2f} GB available, but {required_gb:.2f} GB required")
            return False
        return True
    return False

def set_memory_fraction(fraction=0.8):
    """Set memory fraction to prevent OOM errors"""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)
        print(f"Set GPU memory fraction to {fraction}")

def optimize_memory_settings():
    """Apply memory optimization settings"""
    if torch.cuda.is_available():
        # Set memory fraction to 80% to leave room for other processes
        torch.cuda.set_per_process_memory_fraction(0.8)
        
        # Enable memory caching allocator
        torch.cuda.empty_cache()
        
        # Set deterministic algorithms for consistent memory usage
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print("Applied memory optimization settings")
    else:
        print("GPU not available - skipping memory optimization")