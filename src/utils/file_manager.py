import os
import shutil
from datetime import datetime
from typing import Tuple
import torch
import pickle

class ResultsManager:
    def __init__(self, base_results_dir: str = "results"):
        self.base_results_dir = base_results_dir
        self.ensure_base_dir()
    
    def ensure_base_dir(self):
        """Create base results directory if it doesn't exist"""
        os.makedirs(self.base_results_dir, exist_ok=True)
    
    def create_result_folder(self, model_id: int) -> Tuple[str, str]:
        """
        Create a result folder with format: id(int:001)-date(day-month-year)
        Returns: (folder_path, folder_name)
        """
        today = datetime.now()
        date_str = today.strftime("%d-%m-%Y")
        folder_name = f"{model_id:03d}-{date_str}"
        folder_path = os.path.join(self.base_results_dir, folder_name)
        
        os.makedirs(folder_path, exist_ok=True)
        return folder_path, folder_name
    
    def save_model(self, model, model_path: str, metadata: dict = None):
        """Save model to specified path"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model state
        model_data = {
            'model_state': model.__dict__,
            'metadata': metadata,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, model_path: str, model_class):
        """Load model from specified path"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new model instance
        model = model_class()
        
        # Restore model state
        model.__dict__.update(model_data['model_state'])
        
        return model, model_data.get('metadata', {})
    
    def save_result_image(self, image_array, output_path: str, filename: str = None):
        """Save result image to output path"""
        if filename is None:
            filename = f"result_{datetime.now().strftime('%H%M%S')}.png"
        
        full_path = os.path.join(output_path, filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Save image
        import cv2
        import numpy as np
        
        # Normalize image if needed
        if image_array.dtype != np.uint8:
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        
        cv2.imwrite(full_path, image_array)
        return full_path
    
    def copy_input_image(self, input_path: str, output_folder: str, filename: str = "input.png"):
        """Copy input image to result folder"""
        output_path = os.path.join(output_folder, filename)
        shutil.copy2(input_path, output_path)
        return output_path
    
    def get_next_model_id(self) -> int:
        """Get next available model ID based on existing folders"""
        if not os.path.exists(self.base_results_dir):
            return 1
        
        existing_folders = [f for f in os.listdir(self.base_results_dir) 
                          if os.path.isdir(os.path.join(self.base_results_dir, f))]
        
        max_id = 0
        for folder in existing_folders:
            try:
                # Extract ID from folder name format: 001-dd-mm-yyyy
                folder_id = int(folder.split('-')[0])
                max_id = max(max_id, folder_id)
            except (ValueError, IndexError):
                continue
        
        return max_id + 1
    
    def create_model_summary(self, output_folder: str, model_info: dict):
        """Create a summary file with model information"""
        summary_path = os.path.join(output_folder, "model_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("Model Training Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Type: {model_info.get('model_type', 'Unknown')}\n")
            f.write(f"Training Date: {model_info.get('training_date', 'Unknown')}\n")
            f.write(f"Model ID: {model_info.get('model_id', 'Unknown')}\n")
            f.write(f"Dataset: {model_info.get('dataset', 'Unknown')}\n")
            f.write(f"Performance Metrics: {model_info.get('performance_metrics', 'N/A')}\n")
            f.write(f"Created At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return summary_path