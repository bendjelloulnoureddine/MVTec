import torch
import torch.nn as nn
import timm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from datetime import datetime
from tqdm import tqdm
import gc

# Import custom modules
from src.data.dataset import get_train_loader, get_test_images
from src.utils.gpu_utils import compute_embedding_stats_gpu, mahalanobis_map_gpu, ensure_gpu_tensor, clear_gpu_cache
from src.utils.database import InferenceDatabase
from src.utils.file_manager import ResultsManager
from config.config import Config as C

class FeatureExtractor(nn.Module):
    def __init__(self, layers=[1, 2, 3]):
        super().__init__()
        self.model = timm.create_model("resnet18", pretrained=True, features_only=True)
        self.layers = layers

    def forward(self, x):
        features = self.model(x)
        selected = [features[i] for i in self.layers]
        upsampled = [nn.functional.interpolate(f, size=(x.shape[2]//8, x.shape[3]//8), mode='bilinear') for f in selected]
        return torch.cat(upsampled, dim=1)

class PaDiMGPU:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.feature_extractor = FeatureExtractor([1, 2, 3]).to(device)
        self.feature_extractor.eval()
        self.features = []
        self.mean = None
        self.cov_inv = None
        self.trained = False
        
    def train(self, dataloader):
        """Train PaDiM model using GPU acceleration with memory management"""
        print("🚀 Starting GPU-accelerated training...")
        clear_gpu_cache()
        gc.collect()
        
        # Extract features with memory management
        print("📦 Extracting features from training data...")
        self.features = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Feature extraction")):
                batch = ensure_gpu_tensor(batch)
                feats = self.feature_extractor(batch)
                self.features.append(feats)
                
                # Clear GPU cache every 10 batches to prevent memory accumulation
                if batch_idx % 10 == 0:
                    clear_gpu_cache()
                
                # Monitor memory usage
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated() / 1024**3
                    print(f"Memory usage at batch {batch_idx}: {current_memory:.2f} GB")
        
        # Concatenate all features with memory management
        print("🔗 Concatenating features...")
        all_features = torch.cat(self.features, dim=0)
        
        # Clear intermediate features to free memory
        del self.features
        clear_gpu_cache()
        gc.collect()
        
        print(f"📊 Computing statistics for {all_features.shape[0]} samples...")
        
        # Compute statistics using GPU
        self.mean, cov = compute_embedding_stats_gpu(all_features)
        
        # Clear all_features to free memory
        del all_features
        clear_gpu_cache()
        gc.collect()
        
        # Compute inverse covariance matrices with memory management
        print("🔢 Computing inverse covariance matrices...")
        H, W, C, _ = cov.shape
        self.cov_inv = torch.zeros_like(cov)
        
        for i in range(H):
            for j in range(W):
                try:
                    self.cov_inv[i, j] = torch.inverse(cov[i, j] + 1e-6 * torch.eye(C, device=self.device))
                except:
                    # Use pseudoinverse if matrix is singular
                    self.cov_inv[i, j] = torch.pinverse(cov[i, j] + 1e-6 * torch.eye(C, device=self.device))
            
            # Clear GPU cache every 10 rows to prevent memory accumulation
            if i % 10 == 0:
                clear_gpu_cache()
        
        # Final memory cleanup
        clear_gpu_cache()
        gc.collect()
        
        self.trained = True
        print("✅ Training completed successfully!")
        
        # Print final memory usage
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"Final GPU memory usage: {final_memory:.2f} GB")
        
    def infer_anomaly_map(self, x):
        """Compute anomaly map for input image using GPU"""
        if not self.trained:
            raise ValueError("Model must be trained before inference")
            
        with torch.no_grad():
            x = ensure_gpu_tensor(x)
            feats = self.feature_extractor(x)
            return mahalanobis_map_gpu(feats, self.mean, self.cov_inv)
    
    def get_anomaly_score(self, anomaly_map):
        """Compute overall anomaly score from anomaly map"""
        return float(np.max(anomaly_map))

def preprocess_image(path):
    """Preprocess image for inference"""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (C.IMAGE_SIZE, C.IMAGE_SIZE))
    img = torch.tensor(img).float().permute(2, 0, 1) / 255
    img = (img - 0.5) / 0.5
    return img.unsqueeze(0)

def save_visualization(input_image, anomaly_map, output_path, filename_prefix="result"):
    """Save visualization of results"""
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(input_image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Anomaly map
    plt.subplot(1, 3, 2)
    plt.imshow(anomaly_map, cmap='hot')
    plt.title("Anomaly Map")
    plt.axis('off')
    plt.colorbar()
    
    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(input_image)
    plt.imshow(anomaly_map, cmap='hot', alpha=0.5)
    plt.title("Overlay")
    plt.axis('off')
    
    plt.tight_layout()
    viz_path = os.path.join(output_path, f"{filename_prefix}_visualization.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return viz_path

def main():
    # Initialize components
    print("🔧 Initializing components...")
    db = InferenceDatabase()
    results_manager = ResultsManager()
    
    # Clear GPU cache and garbage collection
    clear_gpu_cache()
    gc.collect()
    torch.set_float32_matmul_precision('medium')
    
    # Print initial memory usage
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"Initial GPU memory usage: {initial_memory:.2f} GB")
    
    # Get next model ID
    model_id = results_manager.get_next_model_id()
    print(f"📝 Model ID: {model_id}")
    
    # Create result folder
    output_folder, folder_name = results_manager.create_result_folder(model_id)
    print(f"📁 Results folder: {output_folder}")
    
    try:
        # Initialize and train model
        print("🏗️  Initializing PaDiM model...")
        model = PaDiMGPU()
        
        # Train the model with memory management
        train_loader = get_train_loader()
        model.train(train_loader)
        
        # Clear memory after training
        del train_loader
        clear_gpu_cache()
        gc.collect()
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"CUDA out of memory error during training: {e}")
            print("Try reducing batch size or image size")
            clear_gpu_cache()
            gc.collect()
            raise e
        else:
            raise e
    
    # Save model
    model_path = os.path.join(output_folder, "model.pth")
    model_metadata = {
        'model_type': 'PaDiM',
        'training_date': datetime.now().isoformat(),
        'dataset': 'MVTec-screw',
        'model_id': model_id,
        'device': str(model.device),
        'feature_layers': [1, 2, 3]
    }
    
    results_manager.save_model(model, model_path, model_metadata)
    print(f"💾 Model saved to: {model_path}")
    
    # Save model info to database
    db_model_id = db.save_model_info(
        model_name=f"PaDiM_{model_id:03d}",
        model_type="PaDiM",
        model_path=model_path,
        performance_metrics=model_metadata
    )
    
    # Create model summary
    results_manager.create_model_summary(output_folder, model_metadata)
    
    # Test inference on sample images with memory management
    print("🧪 Running inference on test images...")
    test_imgs = get_test_images()
    
    inference_results = []
    for i, img_path in enumerate(test_imgs[:5]):
        print(f"Processing image {i+1}/5: {os.path.basename(img_path)}")
        
        try:
            # Preprocess image
            img_tensor = preprocess_image(img_path)
            
            # Run inference with memory management
            anomaly_map = model.infer_anomaly_map(img_tensor)
            anomaly_score = model.get_anomaly_score(anomaly_map)
            
            # Load original image for visualization
            original_img = cv2.imread(img_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            # Clear GPU cache after each inference
            clear_gpu_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"CUDA out of memory error during inference: {e}")
                clear_gpu_cache()
                gc.collect()
                continue
            else:
                raise e
        
        # Save visualization
        viz_path = save_visualization(
            original_img, 
            anomaly_map, 
            output_folder, 
            f"test_{i+1:03d}"
        )
        
        # Copy input image to results folder
        input_copy_path = results_manager.copy_input_image(
            img_path, 
            output_folder, 
            f"input_{i+1:03d}.png"
        )
        
        # Save anomaly map
        anomaly_map_path = results_manager.save_result_image(
            anomaly_map, 
            output_folder, 
            f"anomaly_map_{i+1:03d}.png"
        )
        
        # Save inference result to database
        inference_id = db.save_inference_result(
            model_type="PaDiM",
            input_image_path=img_path,
            output_folder=output_folder,
            model_path=model_path,
            anomaly_score=anomaly_score,
            metadata={
                'visualization_path': viz_path,
                'anomaly_map_path': anomaly_map_path,
                'input_copy_path': input_copy_path,
                'image_index': i+1
            }
        )
        
        inference_results.append({
            'inference_id': inference_id,
            'image_path': img_path,
            'anomaly_score': anomaly_score,
            'output_folder': output_folder
        })
        
        print(f"  ✅ Inference ID: {inference_id}")
        print(f"  📊 Anomaly score: {anomaly_score:.4f}")
    
    # Print summary
    print("\n" + "="*60)
    print("🎉 Training and inference completed successfully!")
    print(f"📁 Results saved to: {output_folder}")
    print(f"💾 Model saved as: {model_path}")
    print(f"🗄️  Database entries created: {len(inference_results)} inference results")
    print("="*60)
    
    # Print inference results summary
    print("\n📋 Inference Results Summary:")
    for result in inference_results:
        print(f"  🔍 {result['inference_id']}: {os.path.basename(result['image_path'])} -> Score: {result['anomaly_score']:.4f}")
    
    print(f"\n📊 Database location: {db.db_path}")
    print(f"📁 Results location: {output_folder}")

if __name__ == "__main__":
    main()