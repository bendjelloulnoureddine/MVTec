import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import seaborn as sns
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
import gc
warnings.filterwarnings('ignore')

class ScrewDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_train = is_train
        
        # For training, only use good samples
        if is_train:
            self.image_paths = list((self.root_dir / "train" / "good").glob("*.png"))
            self.labels = [0] * len(self.image_paths)  # 0 for good
        else:
            # For testing, use both good and defective samples
            good_paths = list((self.root_dir / "test" / "good").glob("*.png"))
            defect_paths = []
            defect_dir = self.root_dir / "test"
            for defect_type in defect_dir.iterdir():
                if defect_type.is_dir() and defect_type.name != "good":
                    defect_paths.extend(list(defect_type.glob("*.png")))
            
            self.image_paths = good_paths + defect_paths
            self.labels = [0] * len(good_paths) + [1] * len(defect_paths)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            if isinstance(self.transform, A.Compose):
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                image = self.transform(image)
        
        return image, self.labels[idx], str(img_path)

class ConvAutoEncoder(nn.Module):
    def __init__(self, input_dim=3, latent_dim=512):
        super(ConvAutoEncoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # First block
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # First block
            nn.Conv2d(latent_dim, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # Second block
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # Third block
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # Output block
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, input_dim, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AnomalyDetector(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, latent_dim=512):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = ConvAutoEncoder(latent_dim=latent_dim)
        self.criterion = nn.MSELoss()
        
        # For tracking training metrics
        self.train_losses = []
        self.val_losses = []
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels, paths = batch
        reconstructed = self.forward(images)
        
        # Only use good samples for training
        good_mask = labels == 0
        if good_mask.sum() > 0:
            loss = self.criterion(reconstructed[good_mask], images[good_mask])
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            
            # Clear GPU cache every 100 batches to prevent memory accumulation
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
            
            return loss
        return None
    
    def validation_step(self, batch, batch_idx):
        images, labels, paths = batch
        reconstructed = self.forward(images)
        
        # Calculate reconstruction error
        mse = F.mse_loss(reconstructed, images, reduction='none')
        reconstruction_error = mse.view(mse.size(0), -1).mean(dim=1)
        
        # Only use good samples for validation loss
        good_mask = labels == 0
        if good_mask.sum() > 0:
            val_loss = reconstruction_error[good_mask].mean()
            self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return {
            'reconstruction_error': reconstruction_error,
            'labels': labels,
            'images': images,
            'reconstructed': reconstructed,
            'paths': paths
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

class MemoryManagementCallback(pl.Callback):
    """Callback to manage GPU memory during training"""
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Clear memory at the start of each epoch"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            current_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU memory at epoch start: {current_memory:.2f} GB")
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Clear memory at the end of each epoch"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            current_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU memory at epoch end: {current_memory:.2f} GB")
        gc.collect()
    
    def on_validation_epoch_start(self, trainer, pl_module):
        """Clear memory before validation"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Clear memory after validation"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, img_size=256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        
        # Data augmentation for training
        self.train_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # No augmentation for validation/test
        self.val_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # Training dataset (only good samples)
            self.train_dataset = ScrewDataset(
                self.data_dir, 
                transform=self.train_transform, 
                is_train=True
            )
            
            # Validation dataset (both good and defective)
            self.val_dataset = ScrewDataset(
                self.data_dir, 
                transform=self.val_transform, 
                is_train=False
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = ScrewDataset(
                self.data_dir, 
                transform=self.val_transform, 
                is_train=False
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )

def calculate_threshold(model, dataloader, percentile=95):
    """Calculate threshold based on reconstruction errors of good samples"""
    model.eval()
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            images, labels, paths = batch
            images = images.to(model.device)
            
            # Only use good samples
            good_mask = labels == 0
            if good_mask.sum() > 0:
                good_images = images[good_mask]
                reconstructed = model(good_images)
                
                mse = F.mse_loss(reconstructed, good_images, reduction='none')
                error = mse.view(mse.size(0), -1).mean(dim=1)
                reconstruction_errors.extend(error.cpu().numpy())
    
    threshold = np.percentile(reconstruction_errors, percentile)
    return threshold, reconstruction_errors

def visualize_results(model, dataloader, threshold, num_samples=8):
    """Visualize reconstruction results and defect localization"""
    model.eval()
    
    fig, axes = plt.subplots(4, num_samples, figsize=(20, 12))
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= num_samples:
                break
                
            images, labels, paths = batch
            images = images.to(model.device)
            reconstructed = model(images)
            
            # Calculate reconstruction error maps
            mse = F.mse_loss(reconstructed, images, reduction='none')
            error_maps = mse.mean(dim=1, keepdim=True)
            
            for i in range(min(images.size(0), num_samples - sample_count)):
                idx = sample_count + i
                if idx >= num_samples:
                    break
                
                # Original image
                orig_img = images[i].cpu().permute(1, 2, 0).numpy()
                orig_img = (orig_img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
                orig_img = np.clip(orig_img, 0, 1)
                axes[0, idx].imshow(orig_img)
                axes[0, idx].set_title(f'Original - {"Good" if labels[i] == 0 else "Defect"}')
                axes[0, idx].axis('off')
                
                # Reconstructed image
                recon_img = reconstructed[i].cpu().permute(1, 2, 0).numpy()
                recon_img = (recon_img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
                recon_img = np.clip(recon_img, 0, 1)
                axes[1, idx].imshow(recon_img)
                axes[1, idx].set_title('Reconstructed')
                axes[1, idx].axis('off')
                
                # Error map
                error_map = error_maps[i, 0].cpu().numpy()
                im = axes[2, idx].imshow(error_map, cmap='hot')
                axes[2, idx].set_title('Error Map')
                axes[2, idx].axis('off')
                
                # Thresholded error map (defect localization)
                thresh_map = (error_map > threshold * 0.5).astype(float)
                axes[3, idx].imshow(thresh_map, cmap='Reds')
                axes[3, idx].set_title('Defect Localization')
                axes[3, idx].axis('off')
            
            sample_count += images.size(0)
    
    plt.tight_layout()
    return fig

def evaluate_model(model, dataloader, threshold):
    """Evaluate model performance"""
    model.eval()
    
    all_errors = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for batch in dataloader:
            images, labels, paths = batch
            images = images.to(model.device)
            
            reconstructed = model(images)
            mse = F.mse_loss(reconstructed, images, reduction='none')
            reconstruction_error = mse.view(mse.size(0), -1).mean(dim=1)
            
            all_errors.extend(reconstruction_error.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
    
    # Calculate predictions
    predictions = (np.array(all_errors) > threshold).astype(int)
    
    # Calculate metrics
    auc_score = roc_auc_score(all_labels, all_errors)
    precision, recall, _ = precision_recall_curve(all_labels, all_errors)
    pr_auc = auc(recall, precision)
    
    # Calculate accuracy, precision, recall for binary classification
    tp = np.sum((predictions == 1) & (np.array(all_labels) == 1))
    fp = np.sum((predictions == 1) & (np.array(all_labels) == 0))
    tn = np.sum((predictions == 0) & (np.array(all_labels) == 0))
    fn = np.sum((predictions == 0) & (np.array(all_labels) == 1))
    
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
    
    return {
        'auc_score': auc_score,
        'pr_auc': pr_auc,
        'accuracy': accuracy,
        'precision': precision_score,
        'recall': recall_score,
        'f1_score': f1_score,
        'threshold': threshold
    }

def main():
    # Clear initial GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Set tensor core precision for better performance on RTX GPUs
    torch.set_float32_matmul_precision('medium')
    
    # Configuration with memory-efficient settings
    DATA_DIR = "dataset/screw"
    BATCH_SIZE = 2  # Reduced for memory efficiency
    IMG_SIZE = 256
    LEARNING_RATE = 1e-3
    LATENT_DIM = 512
    MAX_EPOCHS = 50
    ACCUMULATE_GRAD_BATCHES = 4  # Gradient accumulation for effective batch size
    
    # Initialize wandb
    wandb.init(
        project="mvtec-screw-anomaly-detection",
        config={
            "batch_size": BATCH_SIZE,
            "img_size": IMG_SIZE,
            "learning_rate": LEARNING_RATE,
            "latent_dim": LATENT_DIM,
            "max_epochs": MAX_EPOCHS,
            "architecture": "ConvAutoEncoder"
        }
    )
    
    # Setup data module
    data_module = DataModule(DATA_DIR, BATCH_SIZE, IMG_SIZE)
    
    # Setup model
    model = AnomalyDetector(LEARNING_RATE, LATENT_DIM)
    
    # Setup callbacks with memory management
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='best-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True
    )
    
    memory_callback = MemoryManagementCallback()
    
    # Setup trainer with memory management
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback, early_stopping, memory_callback],
        logger=WandbLogger(),
        accelerator='auto',
        devices='auto',
        precision=16,  # Use mixed precision for memory efficiency
        log_every_n_steps=10,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        gradient_clip_val=1.0,  # Prevent gradient explosions
        enable_checkpointing=True,
        enable_progress_bar=True
    )
    
    # Train model with memory management
    print("Starting training...")
    try:
        trainer.fit(model, data_module)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"CUDA out of memory error during training: {e}")
            print("Try reducing batch size, image size, or model complexity")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            raise e
        else:
            raise e
    
    # Load best model
    best_model = AnomalyDetector.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        learning_rate=LEARNING_RATE,
        latent_dim=LATENT_DIM
    )
    
    # Clear memory after loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Calculate threshold using validation set
    print("Calculating threshold...")
    data_module.setup('fit')
    threshold, good_errors = calculate_threshold(best_model, data_module.val_dataloader())
    
    print(f"Threshold: {threshold:.4f}")
    print(f"Mean reconstruction error for good samples: {np.mean(good_errors):.4f}")
    
    # Evaluate model
    print("Evaluating model...")
    data_module.setup('test')
    metrics = evaluate_model(best_model, data_module.test_dataloader(), threshold)
    
    print(f"Results:")
    print(f"AUC Score: {metrics['auc_score']:.4f}")
    print(f"PR AUC: {metrics['pr_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    # Log metrics to wandb
    wandb.log(metrics)
    
    # Generate visualizations
    print("Generating visualizations...")
    fig = visualize_results(best_model, data_module.test_dataloader(), threshold)
    wandb.log({"reconstruction_results": wandb.Image(fig)})
    plt.show()
    
    # Save model
    torch.save(best_model.state_dict(), 'screw_anomaly_detector.pth')
    print("Model saved as 'screw_anomaly_detector.pth'")
    
    wandb.finish()

if __name__ == "__main__":
    main()
