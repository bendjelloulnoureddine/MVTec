"""
Base model classes for anomaly detection
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from src.core.logging import get_logger

logger = get_logger(__name__)

class BaseAnomalyModel(ABC, nn.Module):
    """Abstract base class for anomaly detection models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.is_trained = False
        self.threshold = None
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model"""
        pass
    
    @abstractmethod
    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score for input tensor"""
        pass
    
    @abstractmethod
    def fit(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Train the model on normal data"""
        pass
    
    def predict(self, x: torch.Tensor, threshold: Optional[float] = None) -> Dict[str, Any]:
        """Predict anomalies in input tensor"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        self.eval()
        with torch.no_grad():
            anomaly_scores = self.compute_anomaly_score(x)
            
            # Use provided threshold or model's threshold
            thresh = threshold if threshold is not None else self.threshold
            if thresh is None:
                raise ValueError("No threshold available for prediction")
            
            # Binary prediction
            predictions = (anomaly_scores > thresh).float()
            
            # Confidence scores
            confidence_scores = torch.abs(anomaly_scores - thresh)
            
            return {
                'anomaly_scores': anomaly_scores,
                'predictions': predictions,
                'confidence_scores': confidence_scores,
                'threshold': thresh
            }
    
    def calculate_threshold(self, dataloader: torch.utils.data.DataLoader, 
                          percentile: float = 90) -> float:
        """Calculate threshold from normal training data"""
        logger.info(f"Calculating threshold at {percentile}th percentile")
        
        self.eval()
        anomaly_scores = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                
                x = x.to(self.device)
                scores = self.compute_anomaly_score(x)
                anomaly_scores.extend(scores.cpu().numpy().flatten())
        
        threshold = np.percentile(anomaly_scores, percentile)
        self.threshold = threshold
        
        logger.info(f"Calculated threshold: {threshold:.6f}")
        logger.info(f"Anomaly scores - Mean: {np.mean(anomaly_scores):.6f}, "
                   f"Std: {np.std(anomaly_scores):.6f}, "
                   f"Min: {np.min(anomaly_scores):.6f}, "
                   f"Max: {np.max(anomaly_scores):.6f}")
        
        return threshold
    
    def save_model(self, path: str):
        """Save model state"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'threshold': self.threshold,
            'is_trained': self.is_trained
        }
        
        torch.save(state, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model state"""
        state = torch.load(path, map_location=self.device)
        
        self.load_state_dict(state['model_state_dict'])
        self.threshold = state.get('threshold')
        self.is_trained = state.get('is_trained', False)
        
        logger.info(f"Model loaded from {path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_type': self.__class__.__name__,
            'config': self.config,
            'device': self.device,
            'is_trained': self.is_trained,
            'threshold': self.threshold,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

class BaseFeatureExtractor(nn.Module):
    """Base class for feature extraction models"""
    
    def __init__(self, model_name: str, layers: list, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        self.layers = layers
        self.pretrained = pretrained
        self.features = {}
        
    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from input tensor"""
        pass
    
    def register_hooks(self):
        """Register forward hooks for feature extraction"""
        def hook_fn(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        for name, module in self.named_modules():
            if any(layer in name for layer in self.layers):
                module.register_forward_hook(hook_fn(name))

class ModelFactory:
    """Factory class for creating anomaly detection models"""
    
    _models = {}
    
    @classmethod
    def register_model(cls, name: str, model_class: type):
        """Register a model class"""
        cls._models[name] = model_class
        logger.info(f"Registered model: {name}")
    
    @classmethod
    def create_model(cls, name: str, config: Dict[str, Any]) -> BaseAnomalyModel:
        """Create model instance"""
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}. Available models: {list(cls._models.keys())}")
        
        model_class = cls._models[name]
        model = model_class(config)
        
        logger.info(f"Created model: {name}")
        logger.info(f"Model info: {model.get_model_info()}")
        
        return model
    
    @classmethod
    def list_models(cls) -> list:
        """List available models"""
        return list(cls._models.keys())

# Utility functions
def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def model_summary(model: nn.Module) -> str:
    """Generate model summary"""
    total_params, trainable_params = count_parameters(model)
    
    summary = f"""
Model Summary:
- Model type: {model.__class__.__name__}
- Total parameters: {total_params:,}
- Trainable parameters: {trainable_params:,}
- Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)
"""
    
    return summary