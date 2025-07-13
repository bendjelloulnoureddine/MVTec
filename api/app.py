import os
import sys
import json
import tempfile
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import torch
from pathlib import Path

# Add parent directory to path to import test_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_model import load_model, test_single_image, calculate_threshold_from_good_samples

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables to store model and threshold
model = None
threshold = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/load_model', methods=['POST'])
def load_model_api():
    """Load the anomaly detection model"""
    global model, threshold
    
    try:
        data = request.get_json()
        model_path = data.get('model_path', 'screw_anomaly_detector.pth')
        data_dir = data.get('data_dir', 'dataset/screw')
        latent_dim = data.get('latent_dim', 512)
        percentile = data.get('percentile', 90)
        custom_threshold = data.get('threshold')
        
        # Convert relative paths to absolute
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), model_path)
        
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_dir)
        
        # Load model
        model = load_model(model_path, latent_dim, device)
        
        # Calculate or use provided threshold
        if custom_threshold is None:
            threshold, _ = calculate_threshold_from_good_samples(
                model, data_dir, device, percentile
            )
        else:
            threshold = custom_threshold
        
        return jsonify({
            'success': True,
            'message': 'Model loaded successfully',
            'threshold': float(threshold),
            'device': device
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/test_image', methods=['POST'])
def test_image_api():
    """Test a single image for anomaly detection"""
    global model, threshold
    
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please load a model first.'
        }), 400
    
    try:
        # Handle file upload
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'File type not allowed. Please use PNG, JPG, or JPEG.'
            }), 400
        
        # Get threshold from form data if provided
        form_threshold = request.form.get('threshold')
        if form_threshold:
            try:
                test_threshold = float(form_threshold)
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid threshold value'
                }), 400
        else:
            test_threshold = threshold
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name
        
        try:
            # Test the image
            result = test_single_image(model, temp_path, test_threshold, device, save_results=False)
            
            return jsonify({
                'success': True,
                'result': result,
                'filename': filename
            })
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Get the current status of the API"""
    return jsonify({
        'model_loaded': model is not None,
        'threshold': float(threshold) if threshold is not None else None,
        'device': device
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)