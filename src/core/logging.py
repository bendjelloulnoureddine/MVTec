"""
Centralized logging system for MVTec Anomaly Detection System
"""
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

from config.settings import get_logging_config

class CustomFormatter(logging.Formatter):
    """Custom formatter with color support and structured logging"""
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m'  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to level name for console output
        if hasattr(record, 'color') and record.color:
            levelname = f"{self.COLORS.get(record.levelname, '')}{record.levelname}{self.RESET}"
            record.levelname = levelname
        
        # Add structured data
        record.timestamp = datetime.utcnow().isoformat()
        record.module = record.name
        
        return super().format(record)

class StructuredLogger:
    """Structured logger with JSON output support"""
    
    def __init__(self, name: str, config: Optional[dict] = None):
        self.name = name
        self.config = config or get_logging_config()
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with handlers and formatters"""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        self.logger.setLevel(getattr(logging, self.config.level.upper()))
        
        # Create formatters
        file_formatter = CustomFormatter(
            fmt='%(timestamp)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = CustomFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add file handler
        if self.config.file_handler:
            self._add_file_handler(file_formatter)
        
        # Add console handler
        if self.config.console_handler:
            self._add_console_handler(console_formatter)
    
    def _add_file_handler(self, formatter):
        """Add rotating file handler"""
        os.makedirs(os.path.dirname(self.config.log_file), exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=self.config.log_file,
            maxBytes=self.config.max_bytes,
            backupCount=self.config.backup_count
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def _add_console_handler(self, formatter):
        """Add console handler with color support"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # Add color flag for console output
        def add_color_flag(record):
            record.color = True
            return True
        
        console_handler.addFilter(add_color_flag)
        self.logger.addHandler(console_handler)
    
    def debug(self, message: str, extra: Optional[dict] = None):
        """Log debug message"""
        self.logger.debug(message, extra=extra or {})
    
    def info(self, message: str, extra: Optional[dict] = None):
        """Log info message"""
        self.logger.info(message, extra=extra or {})
    
    def warning(self, message: str, extra: Optional[dict] = None):
        """Log warning message"""
        self.logger.warning(message, extra=extra or {})
    
    def error(self, message: str, extra: Optional[dict] = None, exc_info: bool = False):
        """Log error message"""
        self.logger.error(message, extra=extra or {}, exc_info=exc_info)
    
    def critical(self, message: str, extra: Optional[dict] = None, exc_info: bool = False):
        """Log critical message"""
        self.logger.critical(message, extra=extra or {}, exc_info=exc_info)
    
    def log_exception(self, message: str, exception: Exception, extra: Optional[dict] = None):
        """Log exception with full traceback"""
        extra = extra or {}
        extra.update({
            'exception_type': type(exception).__name__,
            'exception_message': str(exception)
        })
        self.logger.error(message, extra=extra, exc_info=True)
    
    def log_model_metrics(self, metrics: dict, epoch: Optional[int] = None):
        """Log model training/validation metrics"""
        extra = {'metrics': metrics}
        if epoch is not None:
            extra['epoch'] = epoch
        self.info(f"Model metrics: {json.dumps(metrics, indent=2)}", extra=extra)
    
    def log_inference_result(self, image_path: str, result: dict):
        """Log inference result"""
        extra = {
            'image_path': image_path,
            'result': result
        }
        self.info(f"Inference result for {image_path}: {json.dumps(result, indent=2)}", extra=extra)
    
    def log_api_request(self, endpoint: str, method: str, status_code: int, 
                       response_time: float, user_id: Optional[str] = None):
        """Log API request"""
        extra = {
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'response_time': response_time,
            'user_id': user_id
        }
        self.info(f"API {method} {endpoint} - {status_code} ({response_time:.3f}s)", extra=extra)

class LoggerManager:
    """Centralized logger manager"""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str) -> StructuredLogger:
        """Get or create logger instance"""
        if name not in cls._loggers:
            cls._loggers[name] = StructuredLogger(name)
        return cls._loggers[name]
    
    @classmethod
    def configure_all_loggers(cls, config: dict):
        """Configure all existing loggers"""
        for logger in cls._loggers.values():
            logger.config = config
            logger._setup_logger()

# Convenience functions
def get_logger(name: str) -> StructuredLogger:
    """Get logger instance"""
    return LoggerManager.get_logger(name)

def setup_logging(config: Optional[dict] = None):
    """Setup logging configuration"""
    if config:
        LoggerManager.configure_all_loggers(config)

# Pre-configured loggers for different components
training_logger = get_logger('training')
inference_logger = get_logger('inference')
api_logger = get_logger('api')
model_logger = get_logger('model')
data_logger = get_logger('data')