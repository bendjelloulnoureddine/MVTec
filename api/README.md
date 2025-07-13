# MVTec Anomaly Detection API

A Flask-based REST API and web interface for testing images with the MVTec anomaly detection model.

## Features

- **REST API** for programmatic access
- **Web interface** for easy image testing
- **Model loading** with configurable parameters
- **Threshold configuration** (auto-calculated or custom)
- **Real-time image testing** with detailed results

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the trained model file in the parent directory (e.g., `screw_anomaly_detector.pth`)

3. Run the Flask application:
```bash
python app.py
```

4. Open your browser and go to `http://localhost:5000`

## API Endpoints

### GET `/api/status`
Get the current status of the API (model loaded, threshold, device).

### POST `/api/load_model`
Load the anomaly detection model.

**Request body:**
```json
{
    "model_path": "screw_anomaly_detector.pth",
    "data_dir": "dataset/screw",
    "latent_dim": 512,
    "percentile": 90,
    "threshold": 0.0123  // optional custom threshold
}
```

### POST `/api/test_image`
Test a single image for anomaly detection.

**Request:**
- Form data with `image` file
- Optional `threshold` parameter

**Response:**
```json
{
    "success": true,
    "result": {
        "reconstruction_error": 0.0156,
        "is_defective": true,
        "threshold": 0.0123,
        "confidence": 0.0033
    },
    "filename": "test_image.png"
}
```

## Web Interface

The web interface provides:

1. **Model Status**: Shows if model is loaded and current threshold
2. **Load Model**: Form to load the model with configurable parameters
3. **Test Image**: Upload and test images with optional custom threshold
4. **Results Display**: Shows prediction results with detailed metrics

## Usage Example

1. Start the server: `python app.py`
2. Open `http://localhost:5000` in your browser
3. Load a model using the "Load Model" section
4. Upload an image in the "Test Image" section
5. View the prediction results

## Configuration

- **Model Path**: Path to the trained PyTorch model file
- **Data Directory**: Path to the dataset (used for threshold calculation)
- **Latent Dimension**: Model latent dimension (default: 512)
- **Percentile**: Percentile for threshold calculation (default: 90)
- **Custom Threshold**: Optional custom threshold value

## File Structure

```
api/
├── app.py              # Flask application
├── templates/
│   └── index.html      # Web interface
├── requirements.txt    # Python dependencies
└── README.md          # This file
```