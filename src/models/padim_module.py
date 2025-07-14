import timm
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from src.utils.utils import compute_embedding_stats, mahalanobis_map
from src.utils.gpu_utils import clear_gpu_cache
import gc

class FeatureExtractor(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.model = timm.create_model("resnet18", pretrained=True, features_only=True)
        self.layers = layers

    def forward(self, x):
        features = self.model(x)
        selected = [features[i] for i in self.layers]
        upsampled = [nn.functional.interpolate(f, size=(x.shape[2]//8, x.shape[3]//8), mode='bilinear') for f in selected]
        return torch.cat(upsampled, dim=1)

class PaDiM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.feature_extractor = FeatureExtractor([1, 2, 3])  # equivalent to resnet layers [4,5,6]
        self.features = []

    def training_step(self, batch, batch_idx):
        x = batch
        feats = self.feature_extractor(x)
        self.features.append(feats.cpu())
        
        # Clear GPU cache every 50 batches to prevent memory accumulation
        if batch_idx % 50 == 0:
            clear_gpu_cache()
        
        # Return a dummy loss for PyTorch Lightning
        return torch.tensor(0.0, requires_grad=True, device=x.device)

    def on_train_end(self):
        # Clear GPU cache before processing
        clear_gpu_cache()
        gc.collect()
        
        features = torch.cat(self.features).detach().numpy()
        
        # Clear features list to free memory
        del self.features
        gc.collect()
        
        mean, cov = compute_embedding_stats(features)
        self.mean = mean
        # Add regularization to prevent singular matrices
        eps = 1e-4
        H, W, C, _ = cov.shape
        regularized_cov = cov + eps * np.eye(C).reshape(1, 1, C, C)
        self.cov_inv = np.linalg.inv(regularized_cov)
        
        # Clear variables to free memory
        del features, cov, regularized_cov
        gc.collect()
        clear_gpu_cache()

    def infer_anomaly_map(self, x):
        with torch.no_grad():
            feats = self.feature_extractor(x).cpu().numpy()
            return mahalanobis_map(feats, self.mean, self.cov_inv)
    def configure_optimizers(self):
        # PaDiM doesn't require optimization, but Lightning needs an optimizer
        # Use a dummy optimizer with very small learning rate
        return torch.optim.Adam(self.parameters(), lr=1e-8)


