import timm
import torch
import torch.nn as nn
import pytorch_lightning as pl
import faiss
import numpy as np
from src.utils.gpu_utils import clear_gpu_cache
import gc

class FeatureExtractor(nn.Module):
    def __init__(self, layers=[1, 2, 3]):
        super().__init__()
        self.model = timm.create_model("resnet18", pretrained=True, features_only=True)
        self.layers = layers

    def forward(self, x):
        features = self.model(x)
        selected = [features[i] for i in self.layers]
        upsampled = [nn.functional.interpolate(f, size=(x.shape[2] // 8, x.shape[3] // 8), mode='bilinear') for f in selected]
        return torch.cat(upsampled, dim=1)  # (B, C, H, W)

class PatchCore(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.embeddings = []
        self.index = None
        self.patches = []
        self.patch_shape = None

    def training_step(self, batch, batch_idx):
        features = self.feature_extractor(batch)
        patches = features.unfold(2, 1, 1).unfold(3, 1, 1)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, features.shape[1])
        self.patches.append(patches.detach().cpu())
        
        # Clear GPU cache every 50 batches to prevent memory accumulation
        if batch_idx % 50 == 0:
            clear_gpu_cache()
        
        # Return a dummy loss for PyTorch Lightning
        return torch.tensor(0.0, requires_grad=True, device=batch.device)

    def on_train_end(self):
        # Clear GPU cache before processing
        clear_gpu_cache()
        gc.collect()
        
        all_patches = torch.cat(self.patches, dim=0).numpy().astype(np.float32)
        
        # Clear patches list to free memory
        del self.patches
        gc.collect()
        
        self.index = faiss.IndexFlatL2(all_patches.shape[1])
        self.index.add(all_patches)
        self.patch_shape = (32, 32)  # H, W
        
        # Clear patches array to free memory
        del all_patches
        gc.collect()
        clear_gpu_cache()

    def infer_anomaly_map(self, x):
        self.eval()
        with torch.no_grad():
            f = self.feature_extractor(x).cpu()
            patches = f.unfold(2, 1, 1).unfold(3, 1, 1)
            patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, f.shape[1])
            D, _ = self.index.search(patches.numpy().astype(np.float32), 1)
            scores = D.reshape(self.patch_shape)
            scores = (scores - scores.min()) / (scores.max() - scores.min())
            return scores

    def configure_optimizers(self):
        # PatchCore doesn't require optimization, but Lightning needs an optimizer
        # Use a dummy optimizer with very small learning rate
        return torch.optim.Adam(self.parameters(), lr=1e-8)

