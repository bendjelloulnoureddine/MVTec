import cv2
import numpy as np
from typing import List, Tuple

def detect_anomaly_bboxes(anomaly_map: np.ndarray, 
                         threshold: float = 0.5, 
                         min_area: int = 100) -> List[Tuple[int, int, int, int]]:
    """
    Detect bounding boxes around anomalous regions in an anomaly map.
    
    Args:
        anomaly_map: 2D numpy array representing anomaly scores
        threshold: Threshold for considering a region anomalous
        min_area: Minimum area for a bounding box to be considered valid
        
    Returns:
        List of bounding boxes as (x, y, w, h) tuples
    """
    # Normalize anomaly map to 0-255 range
    normalized = ((anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min()) * 255).astype(np.uint8)
    
    # Apply threshold
    _, binary = cv2.threshold(normalized, int(threshold * 255), 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append((x, y, w, h))
    
    return bboxes

def draw_bboxes_on_image(image: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """
    Draw bounding boxes on an image.
    
    Args:
        image: Input image (RGB format)
        bboxes: List of bounding boxes as (x, y, w, h) tuples
        
    Returns:
        Image with bounding boxes drawn
    """
    result = image.copy()
    for x, y, w, h in bboxes:
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    return result

def save_bbox_annotations(bboxes: List[Tuple[int, int, int, int]], 
                         output_path: str, 
                         image_shape: Tuple[int, int]) -> None:
    """
    Save bounding box annotations to a text file.
    
    Args:
        bboxes: List of bounding boxes as (x, y, w, h) tuples
        output_path: Path to save the annotations
        image_shape: (height, width) of the image
    """
    with open(output_path, 'w') as f:
        for x, y, w, h in bboxes:
            # Convert to normalized coordinates (YOLO format)
            center_x = (x + w/2) / image_shape[1]
            center_y = (y + h/2) / image_shape[0]
            norm_w = w / image_shape[1]
            norm_h = h / image_shape[0]
            
            # Class 0 for anomaly
            f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")