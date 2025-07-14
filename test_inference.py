import torch
import numpy as np
from src.utils.file_manager import ResultsManager
from src.utils.database import InferenceDatabase
from src.data.dataset import get_test_images
import cv2

# Import the enhanced PaDiM
from train_enhanced import PaDiMGPU, preprocess_image, save_visualization

def test_saved_model():
    """Test inference using saved model"""
    # Initialize components
    results_manager = ResultsManager()
    db = InferenceDatabase()
    
    # Load the saved model
    model_path = "results/001-14-07-2025/model.pth"
    
    print(f"🔄 Loading model from: {model_path}")
    model, metadata = results_manager.load_model(model_path, PaDiMGPU)
    
    print(f"📊 Model metadata: {metadata}")
    
    # Test inference on a single image
    test_imgs = get_test_images()
    test_img = test_imgs[0]
    
    print(f"🧪 Testing inference on: {test_img}")
    
    # Preprocess image
    img_tensor = preprocess_image(test_img)
    
    # Run inference
    anomaly_map = model.infer_anomaly_map(img_tensor)
    anomaly_score = model.get_anomaly_score(anomaly_map)
    
    print(f"📊 Anomaly score: {anomaly_score}")
    print(f"📊 Anomaly map shape: {anomaly_map.shape}")
    print(f"📊 Anomaly map range: [{np.min(anomaly_map):.4f}, {np.max(anomaly_map):.4f}]")
    
    # Load original image for visualization
    original_img = cv2.imread(test_img)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Save test visualization
    viz_path = save_visualization(
        original_img, 
        anomaly_map, 
        ".", 
        "test_loaded_model"
    )
    
    print(f"🎨 Visualization saved to: {viz_path}")
    
    # Query database for recent results
    print("\n🗄️  Recent inference results:")
    recent_results = db.get_inference_results(limit=5)
    for result in recent_results:
        print(f"  🔍 {result['inference_id']}: Score {result['anomaly_score']}")

if __name__ == "__main__":
    test_saved_model()