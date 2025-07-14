# MVTec Anomaly Detection System

A comprehensive anomaly detection system for industrial quality control using the MVTec dataset. This system implements both PaDiM (Patch Distribution Modeling) and Convolutional Autoencoder approaches for detecting defects in manufactured products.

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Training Process](#training-process)
- [Testing Process](#testing-process)
- [Model Implementations](#model-implementations)
- [API and Web Interface](#api-and-web-interface)
- [Docker Setup](#docker-setup)
- [Installation](#installation)
- [Usage](#usage)
- [Performance Metrics](#performance-metrics)

## 🎯 Overview

This system provides industrial-grade anomaly detection capabilities for quality control in manufacturing environments. It supports multiple training approaches and algorithms:

### Supported Algorithms
1. **PaDiM (Patch Distribution Modeling)**: Uses feature extraction from pretrained ResNet to model normal patterns
2. **PatchCore**: Advanced patch-based anomaly detection with FAISS-based nearest neighbor search  
3. **Convolutional Autoencoder**: Learns to reconstruct normal images and detects anomalies based on reconstruction error

### Key Features
- **Multiple Training Scripts**: Production-ready, research-focused, and simple training options
- **GPU Memory Management**: Comprehensive memory optimization and monitoring utilities
- **Database Integration**: SQLite database for experiment tracking and result management
- **Automatic Model Management**: Organized result folders with unique model IDs
- **REST API**: Full API support for model deployment and inference
- **Docker Support**: Containerized deployment for production environments

## 🏗️ System Architecture

```
MVTec Anomaly Detection System
├── main.py              # Unified training script with CLI arguments
├── src/
│   ├── data/              # Dataset handling and preprocessing
│   │   └── dataset.py     # MVTec dataset loader
│   ├── models/            # Model implementations
│   │   ├── padim_module.py      # PaDiM anomaly detection
│   │   ├── patchcore_module.py  # PatchCore implementation
│   │   ├── autoencoder.py       # Autoencoder model
│   │   └── base.py             # Base model class
│   ├── training/          # Training scripts and pipelines
│   │   └── cl.py         # Complete training pipeline with wandb
│   ├── utils/            # Utility functions
│   │   ├── utils.py      # Helper functions for statistics
│   │   ├── bbox_detector.py  # Bounding box detection utilities
│   │   ├── database.py   # SQLite database management
│   │   └── file_manager.py   # Results and model management
│   ├── api/              # REST API implementation
│   │   └── app.py        # Flask API server
│   ├── core/             # Core system components
│   │   └── logging.py    # Logging configuration
│   └── inference/        # Inference pipeline
├── config/               # Configuration files
│   ├── config.py         # Main configuration
│   └── settings.py       # Additional settings
├── tests/               # Test files
│   └── test_model.py    # Model testing
├── api/                 # API templates and requirements
│   ├── templates/       # HTML templates
│   └── requirements.txt # API dependencies
├── docker/              # Docker configurations
├── results/             # Training results and models (auto-generated)
└── scripts/             # Deployment scripts
```

## 🎓 Training Process

### PaDiM Training Process

The PaDiM approach follows these steps:

1. **Feature Extraction**
   - Uses pretrained ResNet-18 as feature extractor
   - Extracts features from layers 1, 2, and 3 (equivalent to ResNet layers 4, 5, 6)
   - Upsamples features to consistent spatial dimensions

2. **Statistical Modeling**
   - Computes mean and covariance statistics for each spatial location
   - Uses Ledoit-Wolf shrinkage for robust covariance estimation
   - Creates multivariate Gaussian distribution for each patch

3. **Training Steps**
   ```python
   # 1. Load normal (good) training images
   train_loader = get_train_loader()
   
   # 2. Extract features from all training images
   for batch in train_loader:
       features = feature_extractor(batch)
       all_features.append(features)
   
   # 3. Compute statistical parameters
   mean, cov = compute_embedding_stats(all_features)
   ```

### Autoencoder Training Process

The Convolutional Autoencoder follows traditional deep learning training:

1. **Architecture**
   - Encoder: 4 convolutional blocks with BatchNorm and ReLU
   - Decoder: 4 deconvolutional blocks with upsampling
   - Bottleneck: Configurable latent dimension (default: 512)

2. **Training Objectives**
   - Minimize reconstruction loss (MSE) on normal images
   - Learn compressed representation of normal patterns
   - Optimize for minimal reconstruction error on good samples

3. **Training Configuration**
   ```python
   # Training hyperparameters
   BATCH_SIZE = 16
   LEARNING_RATE = 1e-4
   EPOCHS = 50
   LATENT_DIM = 512
   ```

## 🧪 Testing Process

### 1. Model Loading
```python
# Load trained model
model = load_model(model_path, latent_dim=512, device='cuda')
```

### 2. Threshold Calculation
The system calculates anomaly thresholds from normal training samples:

```python
def calculate_threshold_from_good_samples(model, data_dir, percentile=90):
    # Load good training samples
    good_samples = load_good_samples(data_dir)
    
    # Calculate reconstruction errors
    errors = []
    for sample in good_samples:
        reconstructed = model(sample)
        error = mse_loss(reconstructed, sample)
        errors.append(error)
    
    # Use percentile as threshold
    threshold = np.percentile(errors, percentile)
    return threshold
```

### 3. Anomaly Detection Pipeline

#### For PaDiM:
1. **Feature Extraction**: Extract features from test image
2. **Mahalanobis Distance**: Calculate distance from learned distribution
3. **Anomaly Map**: Generate pixel-level anomaly scores
4. **Classification**: Compare with threshold for binary decision

#### For Autoencoder:
1. **Preprocessing**: Normalize and resize input image
2. **Reconstruction**: Pass through encoder-decoder network
3. **Error Calculation**: Compute MSE between original and reconstructed
4. **Classification**: Compare reconstruction error with threshold

### 4. Result Interpretation

The system provides comprehensive results:

```python
{
    "reconstruction_error": 0.0156,      # Quantitative anomaly score
    "is_defective": True,                # Binary classification
    "threshold": 0.0123,                 # Decision threshold
    "confidence": 0.0033,                # |error - threshold|
    "anomaly_map": [[...]]               # Pixel-level anomaly scores
}
```

## 🔧 Model Implementations

### PaDiM Implementation
- **Feature Extractor**: ResNet-18 pretrained on ImageNet
- **Patch Size**: 8x8 patches for spatial modeling
- **Distribution**: Multivariate Gaussian per patch location
- **Advantages**: Fast inference, interpretable results, no training required

### Autoencoder Implementation
- **Architecture**: Symmetric encoder-decoder with skip connections
- **Loss Function**: Mean Squared Error (MSE)
- **Optimization**: Adam optimizer with learning rate scheduling
- **Advantages**: End-to-end learnable, good reconstruction quality

## 🌐 API and Web Interface

### REST API Endpoints

1. **Model Management**
   - `POST /api/load_model`: Load and configure model
   - `GET /api/status`: Check model status

2. **Inference**
   - `POST /api/test_image`: Test single image
   - `POST /api/batch_test`: Test multiple images

### Web Interface Features

- **Interactive Dashboard**: Real-time model status and configuration
- **File Upload**: Drag-and-drop image testing
- **Results Visualization**: Anomaly maps and detailed metrics
- **Threshold Tuning**: Interactive threshold adjustment

## 🐳 Docker Setup

The system provides containerized deployment with Docker Compose:

### Architecture
```
docker-compose.yml
├── training-service    # Model training environment
├── inference-api      # REST API service
├── web-interface      # Frontend dashboard
└── database          # Results storage (optional)
```

### Services Configuration
- **GPU Support**: NVIDIA Docker runtime for training
- **Volume Mounting**: Persistent model and data storage
- **Network**: Internal communication between services
- **Environment**: Configurable through environment variables

## 📊 Performance Metrics

### Evaluation Metrics

1. **Classification Metrics**
   - Accuracy: Overall correctness
   - Precision: True positives / (True positives + False positives)
   - Recall: True positives / (True positives + False negatives)
   - F1-Score: Harmonic mean of precision and recall

2. **Anomaly Detection Metrics**
   - AUC-ROC: Area under ROC curve
   - AUC-PR: Area under Precision-Recall curve
   - Optimal Threshold: Best threshold for F1-score

3. **Localization Metrics**
   - Pixel-level AUC: Anomaly localization accuracy
   - Intersection over Union (IoU): Segmentation quality

### Benchmark Results

| Model | Dataset | AUC-ROC | AUC-PR | F1-Score |
|-------|---------|---------|---------|----------|
| PaDiM | Screw   | 0.95    | 0.88    | 0.85     |
| AutoEncoder | Screw | 0.92 | 0.84 | 0.82 |

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Docker & Docker Compose (for containerized deployment)

### Local Installation
```bash
# Clone repository
git clone <repository-url>
cd MVTec

# Install dependencies
pip install -r requirements.txt

# Download MVTec dataset
# Place dataset in dataset/ directory

# Train model
python main.py

# Run inference API
python api/app.py
```

### Docker Installation
```bash
# Build and run all services
docker-compose up --build

# Run specific service
docker-compose up inference-api
```

## 💡 Usage

### Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**
   ```bash
   # Place MVTec dataset in dataset/ directory
   # Structure: dataset/screw/train/good/ and dataset/screw/test/
   ```

### Training Models

The system provides three training approaches with different features and complexity levels:

#### 1. Unified Training Script (Recommended)
The main entry point with comprehensive CLI options and memory management:

```bash
# Basic training with PaDiM (default)
python main.py

# Train PatchCore model
python main.py --algorithm patchcore --gpu

# Advanced training with custom parameters
python main.py --algorithm padim \
               --dataset /path/to/dataset \
               --epochs 5 \
               --threshold 0.3 \
               --output-dir results \
               --gpu \
               --batch-size 8 \
               --accumulate-grad-batches 2
```

**Command Line Arguments:**
- `--algorithm/-a`: Choose algorithm (`padim` or `patchcore`) - default: `padim`
- `--dataset/-d`: Path to dataset directory - default: uses config default
- `--epochs/-e`: Number of training epochs - default: `1`
- `--threshold/-t`: Anomaly detection threshold - default: `0.5`
- `--output-dir/-o`: Output directory for results - default: `results`
- `--gpu/-g`: Use GPU if available - default: `False`
- `--batch-size/-b`: Batch size for training - default: `16`
- `--accumulate-grad-batches/-acc`: Gradient accumulation batches - default: `1`

**Key Features:**
- **Memory Management**: Advanced GPU memory optimization with PyTorch Lightning
- **Automatic Model Management**: Creates organized result folders with unique IDs
- **Database Integration**: SQLite database for tracking all training runs
- **Production Ready**: Clean training mode without visualization for deployment
- **Comprehensive Logging**: Detailed model summaries and training information

#### 2. Enhanced Training with Visualization
For development and research with full GPU acceleration and visualization:

```bash
# GPU-accelerated training with visualization
python train_enhanced.py

# Features:
# - GPU-accelerated feature extraction and computation
# - Automatic memory management and monitoring
# - Visualization of training results
# - Database integration for experiment tracking
# - Detailed inference testing on sample images
```

#### 3. Simple Training Script
For quick testing and development:

```bash
# Simple training with basic visualization
python train_simple.py

# Features:
# - Lightweight implementation
# - Basic GPU memory management
# - Simple visualization with matplotlib
# - Good for prototyping and testing
```

#### Training Examples
```bash
# Production training with memory optimization
python main.py -a patchcore -e 10 -g -b 8 -acc 2

# Quick development test
python main.py -a padim -e 1

# Custom dataset training
python main.py -a padim -d /path/to/custom/dataset -e 5 -g

# High-sensitivity detection
python main.py -a patchcore -t 0.2 -e 15 -g
```

#### Legacy Training Scripts
```bash
# Complete training pipeline with wandb logging (legacy)
python src/training/cl.py

# Features:
# - Automatic threshold calculation
# - Comprehensive evaluation metrics
# - W&B experiment tracking
# - Model checkpointing and early stopping
```

### Testing and Inference

#### Single Image Testing
```bash
# Test with PaDiM model
python -c "
from src.models.padim_module import PaDiM
from src.data.dataset import get_test_images
import torch

model = PaDiM()
# Load your trained model here
test_images = get_test_images()
anomaly_map = model.infer_anomaly_map(test_images[0])
print(f'Anomaly detected: {anomaly_map.max() > threshold}')
"
```

#### Batch Testing
```bash
# Test multiple images from dataset
python tests/test_model.py
```

### API Usage

#### Start API Server
```bash
# Run Flask API server
python src/api/app.py

# Or using the API in api/ directory
python api/app.py
```

#### Test API Endpoints
```bash
# Test single image
curl -X POST -F "image=@test_image.png" http://localhost:5000/api/test_image

# Check API status
curl http://localhost:5000/api/status

# Load specific model
curl -X POST -H "Content-Type: application/json" \
  -d '{"model_type": "padim", "model_path": "checkpoints/best_model.pth"}' \
  http://localhost:5000/api/load_model
```

### Docker Usage

#### Build and Run
```bash
# Build all services
docker-compose up --build

# Run specific service
docker-compose up inference-api

# Development mode
docker-compose -f docker-compose.dev.yml up
```

#### Docker Services
- **training-service**: Model training environment
- **inference-api**: REST API for model inference
- **web-interface**: Web dashboard for testing
- **jupyter**: Jupyter notebook environment

### Configuration

#### Model Configuration
```python
# config/config.py
class Config:
    IMAGE_SIZE = 256          # Input image size
    BATCH_SIZE = 16          # Training batch size
    LEARNING_RATE = 1e-4     # Learning rate
    LATENT_DIM = 512         # Autoencoder latent dimension
    THRESHOLD_PERCENTILE = 90 # Anomaly threshold percentile
```

#### Data Configuration
```python
# Modify dataset path in src/data/dataset.py
def get_train_loader():
    return DataLoader(
        MVTecDataset("dataset/screw/train/good"), 
        batch_size=C.BATCH_SIZE, 
        shuffle=False
    )
```

### Advanced Usage

#### GPU Memory Management
The system includes comprehensive GPU memory management utilities:

```python
# GPU utilities for memory optimization
from src.utils.gpu_utils import (
    clear_gpu_cache, 
    print_gpu_memory_stats,
    monitor_memory_usage,
    optimize_memory_settings
)

# Apply memory optimization settings
optimize_memory_settings()

# Monitor memory usage of a function
@monitor_memory_usage
def train_model():
    # Your training code here
    pass

# Clear GPU cache during training
clear_gpu_cache()

# Print detailed memory statistics
print_gpu_memory_stats("After training")
```

#### Memory Management Features
- **GPU Cache Management**: Automatic clearing of GPU memory cache
- **Memory Monitoring**: Real-time memory usage tracking during training
- **Memory Optimization**: Automatic memory optimization settings
- **OOM Prevention**: Gradient accumulation and batch size optimization
- **Memory Statistics**: Detailed GPU memory usage reporting

#### Custom Dataset
```python
# Create custom dataset class
from src.data.dataset import MVTecDataset

class CustomDataset(MVTecDataset):
    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir)
        self.transform = transform
    
    def __getitem__(self, idx):
        # Custom preprocessing
        return processed_image
```

#### Model Comparison
```python
# Compare different models
from src.models.padim_module import PaDiM
from src.models.patchcore_module import PatchCore

# Train both models
padim_model = PaDiM()
patchcore_model = PatchCore()

# Compare performance
padim_result = padim_model.infer_anomaly_map(test_image)
patchcore_result = patchcore_model.infer_anomaly_map(test_image)
```

## 📁 Project Structure Details

### Core Components

#### Training Scripts

**`main.py`** - Unified Training Script (Recommended)
- **Production Ready**: Single entry point for both PaDiM and PatchCore training
- **CLI Arguments**: Comprehensive command-line interface with algorithm selection
- **Memory Management**: PyTorch Lightning with automatic memory optimization
- **Result Management**: Automatic model saving and database integration
- **No Visualization**: Clean training mode for production environments

**`train_enhanced.py`** - GPU-Accelerated Training with Visualization
- **GPU Acceleration**: Full GPU acceleration for feature extraction and computation
- **Memory Monitoring**: Real-time memory usage tracking and optimization
- **Visualization**: Automatic generation of training result visualizations
- **Database Integration**: Complete experiment tracking and result storage
- **Research Focus**: Ideal for development and research workflows

**`train_simple.py`** - Simple Training Script
- **Lightweight**: Basic implementation for quick testing and prototyping
- **Minimal Dependencies**: Simple visualization with matplotlib
- **Memory Management**: Basic GPU memory management utilities
- **Development**: Good for understanding the algorithm and quick tests

#### `/src/data/`
- **`dataset.py`**: MVTec dataset loader with proper transforms and data loading utilities
- Handles train/test splits and image preprocessing
- Supports multiple MVTec product categories

#### `/src/models/`
- **`padim_module.py`**: PaDiM implementation with feature extraction and Mahalanobis distance
- **`patchcore_module.py`**: PatchCore with FAISS-based nearest neighbor search
- **`autoencoder.py`**: Convolutional autoencoder for reconstruction-based anomaly detection
- **`base.py`**: Base model class with common functionality

#### `/src/training/`
- **`cl.py`**: Complete training pipeline with:
  - Automatic threshold calculation
  - Comprehensive evaluation metrics
  - W&B experiment tracking
  - Model checkpointing and early stopping

#### `/src/utils/`
- **`utils.py`**: Statistical computations (mean, covariance) and Mahalanobis distance
- **`gpu_utils.py`**: GPU memory management and optimization utilities
- **`bbox_detector.py`**: Bounding box detection for anomaly localization
- **`database.py`**: SQLite database management for tracking training runs
- **`file_manager.py`**: Results organization and model persistence

#### `/config/`
- **`config.py`**: Main configuration file with model hyperparameters
- **`settings.py`**: Additional system settings and paths

### API and Web Interface

#### `/src/api/` and `/api/`
- REST API implementation with Flask
- File upload handling and model inference
- Response formatting and error handling

#### `/docker/`
- **`Dockerfile.training`**: Training environment
- **`Dockerfile.inference`**: Inference API container
- **`Dockerfile.web`**: Web interface container
- **`Dockerfile.jupyter`**: Jupyter notebook environment

### Configuration Files

#### Model Configuration
```python
# config/config.py
class Config:
    IMAGE_SIZE = 256          # Input image size
    BATCH_SIZE = 16          # Training batch size
    LEARNING_RATE = 1e-4     # Learning rate
    LATENT_DIM = 512         # Autoencoder latent dimension
    THRESHOLD_PERCENTILE = 90 # Anomaly threshold percentile
```

#### API Configuration
```python
# API settings
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'checkpoints/best_model.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Use GPU memory management utilities: `clear_gpu_cache()`
   - Reduce batch size: `--batch-size 8`
   - Enable gradient accumulation: `--accumulate-grad-batches 2`
   - Use memory optimization: `optimize_memory_settings()`
   - Check memory usage: `print_gpu_memory_stats()`

2. **Model Loading Errors**
   - Check model path and format
   - Verify PyTorch version compatibility
   - Ensure correct model architecture

3. **Dataset Issues**
   - Verify dataset structure matches MVTec format
   - Check image formats and sizes
   - Ensure proper train/test split

4. **Memory Management Issues**
   - Monitor memory usage with `@monitor_memory_usage` decorator
   - Use `train_enhanced.py` for automatic memory management
   - Clear GPU cache regularly during training
   - Set appropriate memory fraction with `set_memory_fraction()`

### Performance Optimization

1. **Training Optimization**
   - Use GPU-accelerated training: `--gpu`
   - Enable gradient accumulation to reduce memory usage
   - Use PyTorch Lightning for automatic optimization
   - Apply memory optimization settings before training

2. **Inference Optimization**
   - Model quantization for reduced memory footprint
   - TensorRT optimization for faster inference
   - Batch inference for multiple images
   - GPU memory management during inference

3. **Memory Optimization**
   - Use `train_enhanced.py` for GPU-accelerated training
   - Enable automatic memory management callbacks
   - Monitor memory usage throughout training
   - Clear GPU cache between epochs and batches

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Contact the development team
- Check the documentation wiki