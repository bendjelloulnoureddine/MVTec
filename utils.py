import numpy as np
from sklearn.covariance import LedoitWolf
import cv2

def compute_embedding_stats(embeddings: np.ndarray):
    N, C, H, W = embeddings.shape
    embeddings = embeddings.transpose(0, 2, 3, 1).reshape(N, -1, C)
    mean = embeddings.mean(axis=0)
    cov = np.stack([LedoitWolf().fit(embeddings[:, i, :]).covariance_ for i in range(H * W)], axis=0)
    return mean.reshape(H, W, C), cov.reshape(H, W, C, C)

def mahalanobis_map(test_embed, mean, cov_inv):
    H, W, C = mean.shape
    test_embed = test_embed.transpose(0, 2, 3, 1)  # (B, H, W, C)
    scores = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            diff = test_embed[0, i, j] - mean[i, j]
            scores[i, j] = np.sqrt(np.dot(np.dot(diff, cov_inv[i, j]), diff.T))
    return cv2.resize(scores, (256, 256))

