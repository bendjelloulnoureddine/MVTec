import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
import seaborn as sns
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import json

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

class AnomalyDetector(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.model = ConvAutoEncoder(latent_dim=latent_dim)
        
    def forward(self, x):
        return self.model(x)

class ScrewTestDataset(Dataset):
    def __init__(self, root_dir, transform=None, single_image=None):
        self.root_dir = Path(root_dir) if root_dir else None
        self.transform = transform
        self.single_image = single_image
        
        if single_image:
            self.image_paths = [single_image]
            self.labels = [0]  # Unknown label for single image
        else:
            # Load test dataset
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
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, self.labels[idx], str(img_path)

def load_model(model_path, latent_dim=512, device='cuda'):
    """Load the trained model"""
    model = AnomalyDetector(latent_dim=latent_dim)
    
    # Load state dict
    if model_path.endswith('.ckpt'):
        # PyTorch Lightning checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix from keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.' prefix
                new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)
    else:
        # Regular PyTorch model
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    model.eval()
    return model

def calculate_threshold_from_good_samples(model, data_dir, device='cuda', percentile=90):
    """Calculate threshold using good samples from training data"""
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    good_dir = Path(data_dir) / "train" / "good"
    good_paths = list(good_dir.glob("*.png"))
    
    reconstruction_errors = []
    
    with torch.no_grad():
        for img_path in good_paths:
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            augmented = transform(image=image)
            image_tensor = augmented['image'].unsqueeze(0).to(device)
            
            reconstructed = model(image_tensor)
            mse = F.mse_loss(reconstructed, image_tensor, reduction='none')
            error = mse.view(mse.size(0), -1).mean(dim=1)
            reconstruction_errors.append(error.cpu().item())
    
    threshold = np.percentile(reconstruction_errors, percentile)
    print(f"Reconstruction errors statistics:")
    print(f"  Mean: {np.mean(reconstruction_errors):.4f}")
    print(f"  Std: {np.std(reconstruction_errors):.4f}")
    print(f"  Min: {np.min(reconstruction_errors):.4f}")
    print(f"  Max: {np.max(reconstruction_errors):.4f}")
    print(f"  {percentile}th percentile: {threshold:.4f}")
    
    return threshold, reconstruction_errors

def test_single_image(model, image_path, threshold, device='cuda', save_results=True):
    """Test a single image and show results"""
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    
    augmented = transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        reconstructed = model(image_tensor)
        
        # Calculate reconstruction error
        mse = F.mse_loss(reconstructed, image_tensor, reduction='none')
        reconstruction_error = mse.view(mse.size(0), -1).mean(dim=1).cpu().item()
        
        # Calculate error map
        error_map = mse.mean(dim=1, keepdim=True).squeeze().cpu().numpy()
        
        # Make prediction
        is_defective = reconstruction_error > threshold
        
        # Denormalize images for visualization
        img_denorm = image_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        img_denorm = (img_denorm * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img_denorm = np.clip(img_denorm, 0, 1)
        
        recon_denorm = reconstructed.squeeze().cpu().permute(1, 2, 0).numpy()
        recon_denorm = (recon_denorm * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        recon_denorm = np.clip(recon_denorm, 0, 1)
        
        # Create defect localization map
        defect_map = (error_map > threshold * 0.5).astype(float)
        
        # Print results
        print(f"\nResults for {image_path}:")
        print(f"Reconstruction Error: {reconstruction_error:.4f}")
        print(f"Threshold: {threshold:.4f}")
        print(f"Prediction: {'DEFECTIVE' if is_defective else 'GOOD'}")
        print(f"Confidence: {abs(reconstruction_error - threshold):.4f}")
        
        if save_results:
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Original image
            axes[0, 0].imshow(img_denorm)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Reconstructed image
            axes[0, 1].imshow(recon_denorm)
            axes[0, 1].set_title('Reconstructed Image')
            axes[0, 1].axis('off')
            
            # Error map
            im1 = axes[1, 0].imshow(error_map, cmap='hot')
            axes[1, 0].set_title('Reconstruction Error Map')
            axes[1, 0].axis('off')
            plt.colorbar(im1, ax=axes[1, 0])
            
            # Defect localization
            im2 = axes[1, 1].imshow(defect_map, cmap='Reds')
            axes[1, 1].set_title('Defect Localization')
            axes[1, 1].axis('off')
            
            plt.suptitle(f'Prediction: {"DEFECTIVE" if is_defective else "GOOD"} (Error: {reconstruction_error:.4f})')
            plt.tight_layout()
            
            # Save result
            output_path = f"test_result_{Path(image_path).stem}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Results saved to: {output_path}")
            plt.show()
        
        return {
            'reconstruction_error': float(reconstruction_error),  # Convert to float
            'is_defective': bool(is_defective),  # Convert to bool explicitly
            'threshold': float(threshold),  # Convert to float
            'confidence': float(abs(reconstruction_error - threshold))  # Convert to float
        }

def test_dataset(model, data_dir, threshold, device='cuda'):
    """Test the entire test dataset"""
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    dataset = ScrewTestDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    
    all_errors = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for batch in dataloader:
            images, labels, paths = batch
            images = images.to(device)
            
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
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
    
    # Print results
    print(f"\n=== Test Dataset Results ===")
    print(f"Total samples: {len(all_labels)}")
    print(f"Good samples: {np.sum(np.array(all_labels) == 0)}")
    print(f"Defective samples: {np.sum(np.array(all_labels) == 1)}")
    print(f"\nMetrics:")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision_score:.4f}")
    print(f"Recall: {recall_score:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Threshold: {threshold:.4f}")
    
    # Print confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Good', 'Defective'], 
                yticklabels=['Good', 'Defective'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'auc_score': float(auc_score),
        'pr_auc': float(pr_auc),
        'accuracy': float(accuracy),
        'precision': float(precision_score),
        'recall': float(recall_score),
        'f1_score': float(f1_score),
        'threshold': float(threshold),
        'confusion_matrix': cm.tolist()
    }

def main():
    parser = argparse.ArgumentParser(description='Test MVTec Screw Anomaly Detection Model')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the trained model (.pth or .ckpt file)')
    parser.add_argument('--data_dir', type=str, default='dataset/screw',
                       help='Path to the dataset directory')
    parser.add_argument('--single_image', type=str, default=None,
                       help='Path to a single image to test')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Threshold for anomaly detection (if not provided, will be calculated)')
    parser.add_argument('--percentile', type=int, default=90,
                       help='Percentile for threshold calculation (default: 90)')
    parser.add_argument('--latent_dim', type=int, default=512,
                       help='Latent dimension of the model')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path, args.latent_dim, args.device)
    print("Model loaded successfully!")
    
    # Calculate or use provided threshold
    if args.threshold is None:
        print(f"Calculating threshold from good training samples (percentile={args.percentile})...")
        threshold, good_errors = calculate_threshold_from_good_samples(
            model, args.data_dir, args.device, args.percentile
        )
        print(f"Calculated threshold: {threshold:.4f}")
    else:
        threshold = args.threshold
        print(f"Using provided threshold: {threshold:.4f}")
    
    # Test single image or entire dataset
    if args.single_image:
        print(f"\nTesting single image: {args.single_image}")
        result = test_single_image(model, args.single_image, threshold, args.device)
        
        # Save results to JSON
        with open('single_image_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        print("Results saved to: single_image_result.json")
        
    else:
        print("\nTesting entire dataset...")
        results = test_dataset(model, args.data_dir, threshold, args.device)
        
        # Save results to JSON
        with open('dataset_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("Results saved to: dataset_test_results.json")

if __name__ == "__main__":
    main()
