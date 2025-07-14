import timm
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from src.utils.utils import compute_embedding_stats, mahalanobis_map

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
        return None

    def on_train_end(self):
        features = torch.cat(self.features).detach().numpy()
        mean, cov = compute_embedding_stats(features)
        self.mean = mean
        self.cov_inv = np.linalg.inv(cov)

    def infer_anomaly_map(self, x):
        with torch.no_grad():
            feats = self.feature_extractor(x).cpu().numpy()
            return mahalanobis_map(feats, self.mean, self.cov_inv)
    def configure_optimizers(self):
        return None


