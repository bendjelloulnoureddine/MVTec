import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import cv2
import numpy as np

from src.models.patchcore_module import PatchCore
from src.data.dataset import get_train_loader, get_test_images
from config.config import Config as C

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (C.IMAGE_SIZE, C.IMAGE_SIZE))
    img = torch.tensor(img).float().permute(2, 0, 1) / 255
    img = (img - 0.5) / 0.5
    return img.unsqueeze(0)

if __name__ == "__main__":
    # Optimize tensor cores for better performance
    torch.set_float32_matmul_precision('medium')
    
    model = PatchCore()
    trainer = pl.Trainer(
        accelerator="auto", 
        max_epochs=1, 
        logger=False, 
        enable_checkpointing=False,
        log_every_n_steps=10,  # More frequent logging
        enable_progress_bar=True,
        enable_model_summary=True
    )
    trainer.fit(model, get_train_loader())

    print("🔍 Inference...")
    for img_path in get_test_images()[:5]:
        img_tensor = preprocess_image(img_path)
        anomaly_map = model.infer_anomaly_map(img_tensor)

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        plt.title("Input")

        plt.subplot(1, 2, 2)
        plt.imshow(anomaly_map, cmap='hot')
        plt.title("Anomaly Map")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

