import torch
import torch.nn as nn
import timm
import numpy as np
from tqdm import tqdm
from src.data.dataset import get_train_loader, get_test_images
from src.utils.utils import compute_embedding_stats, mahalanobis_map
from config.config import Config as C
import matplotlib.pyplot as plt
import cv2

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

class PaDiM:
    def __init__(self):
        self.feature_extractor = FeatureExtractor([1, 2, 3])
        self.feature_extractor.eval()
        self.features = []
        self.mean = None
        self.cov_inv = None
        
    def train(self, dataloader):
        print("📦 Extracting features from training data...")
        self.feature_extractor.cuda()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Training"):
                batch = batch.cuda()
                feats = self.feature_extractor(batch)
                self.features.append(feats.cpu())
                
        # Compute statistics
        print("📊 Computing statistics...")
        features = torch.cat(self.features).detach().numpy()
        self.mean, cov = compute_embedding_stats(features)
        self.cov_inv = np.linalg.inv(cov)
        print("✅ Training completed!")
        
    def infer_anomaly_map(self, x):
        with torch.no_grad():
            self.feature_extractor.cuda()
            x = x.cuda()
            feats = self.feature_extractor(x).cpu().numpy()
            return mahalanobis_map(feats, self.mean, self.cov_inv)

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (C.IMAGE_SIZE, C.IMAGE_SIZE))
    img = torch.tensor(img).float().permute(2, 0, 1) / 255
    img = (img - 0.5) / 0.5
    return img.unsqueeze(0)

if __name__ == "__main__":
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # Set precision
    torch.set_float32_matmul_precision('medium')
    
    # Initialize model
    model = PaDiM()
    
    # Train the model
    train_loader = get_train_loader()
    model.train(train_loader)
    
    # Test inference
    print("🧪 Testing inference...")
    test_imgs = get_test_images()
    
    for i, img_path in enumerate(test_imgs[:3]):
        img_tensor = preprocess_image(img_path)
        anomaly_map = model.infer_anomaly_map(img_tensor)
        
        # Display results
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 3, 1)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title("Original")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(anomaly_map, cmap='hot')
        plt.title("Anomaly Map")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(img)
        plt.imshow(anomaly_map, cmap='hot', alpha=0.5)
        plt.title("Overlay")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"result_{i}.png")
        plt.show()
    
    print("🎉 All done!")