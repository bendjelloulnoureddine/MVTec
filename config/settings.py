"""
Configuration settings for MVTec Anomaly Detection System
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from dataclasses import dataclass
from enum import Enum

class ModelType(Enum):
    PADIM = "padim"
    AUTOENCODER = "autoencoder"

class Environment(Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    username: str = "postgres"
    password: str = "password"
    database: str = "anomaly_detection"

@dataclass
class ModelConfig:
    """Model configuration settings"""
    image_size: int = 256
    batch_size: int = 16
    learning_rate: float = 1e-4
    latent_dim: int = 512
    max_epochs: int = 50
    patience: int = 10
    threshold_percentile: int = 90
    feature_layers: list = None
    
    def __post_init__(self):
        if self.feature_layers is None:
            self.feature_layers = [1, 2, 3]

@dataclass
class TrainingConfig:
    """Training configuration settings"""
    data_dir: str = "dataset"
    output_dir: str = "checkpoints"
    log_dir: str = "logs"
    model_type: ModelType = ModelType.AUTOENCODER
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True
    gradient_clip_val: float = 1.0
    
@dataclass
class InferenceConfig:
    """Inference configuration settings"""
    model_path: str = "checkpoints/best_model.pth"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    save_results: bool = True
    visualization: bool = True

@dataclass
class APIConfig:
    """API configuration settings"""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    max_content_length: int = 16 * 1024 * 1024  # 16MB
    allowed_extensions: set = None
    upload_folder: str = "uploads"
    results_folder: str = "results"
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = {'png', 'jpg', 'jpeg'}

@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler: bool = True
    console_handler: bool = True
    log_file: str = "logs/app.log"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

class Config:
    """Main configuration class"""
    
    def __init__(self, env: Environment = Environment.DEVELOPMENT):
        self.env = env
        self.project_root = Path(__file__).parent.parent
        
        # Load environment variables
        self._load_environment_variables()
        
        # Initialize configuration objects
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.inference = InferenceConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()
        self.database = DatabaseConfig()
        
        # Apply environment-specific settings
        self._apply_environment_settings()
    
    def _load_environment_variables(self):
        """Load configuration from environment variables"""
        self.model_path = os.getenv("MODEL_PATH", "checkpoints/best_model.pth")
        self.data_dir = os.getenv("DATA_DIR", "dataset")
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "5000"))
        self.debug = os.getenv("DEBUG", "False").lower() == "true"
        
    def _apply_environment_settings(self):
        """Apply environment-specific configuration"""
        if self.env == Environment.PRODUCTION:
            self.api.debug = False
            self.logging.level = "WARNING"
            self.training.mixed_precision = True
            
        elif self.env == Environment.DEVELOPMENT:
            self.api.debug = True
            self.logging.level = "DEBUG"
            self.logging.console_handler = True
            
        elif self.env == Environment.TESTING:
            self.api.debug = False
            self.logging.level = "ERROR"
            self.model.batch_size = 2
            self.training.max_epochs = 1
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration"""
        return self.model
    
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration"""
        return self.training
    
    def get_inference_config(self) -> InferenceConfig:
        """Get inference configuration"""
        return self.inference
    
    def get_api_config(self) -> APIConfig:
        """Get API configuration"""
        return self.api
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration"""
        return self.logging
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "inference": self.inference.__dict__,
            "api": self.api.__dict__,
            "logging": self.logging.__dict__,
            "database": self.database.__dict__
        }
    
    def save_config(self, path: str):
        """Save configuration to file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load_config(cls, path: str) -> 'Config':
        """Load configuration from file"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        # Update configuration with loaded values
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config

# Global configuration instance
config = Config()

# Convenience functions
def get_config() -> Config:
    """Get global configuration instance"""
    return config

def set_environment(env: Environment):
    """Set application environment"""
    global config
    config = Config(env)

def get_model_config() -> ModelConfig:
    """Get model configuration"""
    return config.get_model_config()

def get_training_config() -> TrainingConfig:
    """Get training configuration"""
    return config.get_training_config()

def get_inference_config() -> InferenceConfig:
    """Get inference configuration"""
    return config.get_inference_config()

def get_api_config() -> APIConfig:
    """Get API configuration"""
    return config.get_api_config()

def get_logging_config() -> LoggingConfig:
    """Get logging configuration"""
    return config.get_logging_config()