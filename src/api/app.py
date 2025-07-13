"""
Enterprise-grade Flask API for MVTec Anomaly Detection System
"""
import os
import sys
import json
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any

from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import torch
import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.settings import get_api_config, get_model_config, get_inference_config
from src.core.logging import get_logger, api_logger
from src.api.middleware import setup_cors, setup_error_handlers, setup_request_logging
from src.api.utils import allowed_file, validate_image, process_image
from src.api.models import ModelManager
from src.api.validators import validate_load_model_request, validate_test_image_request

app = Flask(__name__)

# Load configuration
api_config = get_api_config()
model_config = get_model_config()
inference_config = get_inference_config()

# Configure Flask app
app.config['MAX_CONTENT_LENGTH'] = api_config.max_content_length
app.config['UPLOAD_FOLDER'] = api_config.upload_folder
app.config['RESULTS_FOLDER'] = api_config.results_folder

# Setup middleware
setup_cors(app)
setup_error_handlers(app)
setup_request_logging(app)

# Initialize model manager
model_manager = ModelManager()

# Create necessary directories
os.makedirs(api_config.upload_folder, exist_ok=True)
os.makedirs(api_config.results_folder, exist_ok=True)

logger = get_logger(__name__)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'model_loaded': model_manager.is_model_loaded(),
        'device': model_manager.get_device()
    })

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get API status"""
    try:
        status = model_manager.get_status()
        return jsonify({
            'success': True,
            'status': status
        })
    except Exception as e:
        api_logger.log_exception("Error getting status", e)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/load_model', methods=['POST'])
def load_model():
    """Load anomaly detection model"""
    start_time = time.time()
    
    try:
        # Validate request
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        validation_result = validate_load_model_request(data)
        if not validation_result['valid']:
            return jsonify({
                'success': False,
                'error': validation_result['error']
            }), 400
        
        # Load model
        result = model_manager.load_model(data)
        
        response_time = time.time() - start_time
        api_logger.log_api_request('/api/load_model', 'POST', 200, response_time)
        
        return jsonify({
            'success': True,
            'result': result,
            'response_time': response_time
        })
        
    except Exception as e:
        response_time = time.time() - start_time
        api_logger.log_exception("Error loading model", e)
        api_logger.log_api_request('/api/load_model', 'POST', 500, response_time)
        
        return jsonify({
            'success': False,
            'error': str(e),
            'response_time': response_time
        }), 500

@app.route('/api/test_image', methods=['POST'])
def test_image():
    """Test single image for anomaly detection"""
    start_time = time.time()
    temp_path = None
    
    try:
        # Check if model is loaded
        if not model_manager.is_model_loaded():
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please load a model first.'
            }), 400
        
        # Validate file upload
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
        
        if not allowed_file(file.filename, api_config.allowed_extensions):
            return jsonify({
                'success': False,
                'error': f'File type not allowed. Supported formats: {", ".join(api_config.allowed_extensions)}'
            }), 400
        
        # Get threshold from form data
        threshold = request.form.get('threshold')
        if threshold:
            try:
                threshold = float(threshold)
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid threshold value'
                }), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name
        
        # Validate and process image
        if not validate_image(temp_path):
            return jsonify({
                'success': False,
                'error': 'Invalid image file'
            }), 400
        
        # Test the image
        result = model_manager.test_image(temp_path, threshold)
        
        response_time = time.time() - start_time
        api_logger.log_api_request('/api/test_image', 'POST', 200, response_time)
        api_logger.log_inference_result(filename, result)
        
        return jsonify({
            'success': True,
            'result': result,
            'filename': filename,
            'response_time': response_time
        })
        
    except Exception as e:
        response_time = time.time() - start_time
        api_logger.log_exception("Error testing image", e)
        api_logger.log_api_request('/api/test_image', 'POST', 500, response_time)
        
        return jsonify({
            'success': False,
            'error': str(e),
            'response_time': response_time
        }), 500
    
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_path}: {e}")

@app.route('/api/batch_test', methods=['POST'])
def batch_test():
    """Test multiple images for anomaly detection"""
    start_time = time.time()
    
    try:
        # Check if model is loaded
        if not model_manager.is_model_loaded():
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please load a model first.'
            }), 400
        
        # Validate files
        if 'images' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image files provided'
            }), 400
        
        files = request.files.getlist('images')
        if not files:
            return jsonify({
                'success': False,
                'error': 'No files selected'
            }), 400
        
        # Get threshold from form data
        threshold = request.form.get('threshold')
        if threshold:
            try:
                threshold = float(threshold)
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid threshold value'
                }), 400
        
        # Process all images
        results = []
        temp_paths = []
        
        for file in files:
            if not allowed_file(file.filename, api_config.allowed_extensions):
                continue
            
            # Save file temporarily
            filename = secure_filename(file.filename)
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
                file.save(tmp_file.name)
                temp_paths.append(tmp_file.name)
                
                # Validate and test image
                if validate_image(tmp_file.name):
                    try:
                        result = model_manager.test_image(tmp_file.name, threshold)
                        results.append({
                            'filename': filename,
                            'result': result,
                            'success': True
                        })
                    except Exception as e:
                        results.append({
                            'filename': filename,
                            'error': str(e),
                            'success': False
                        })
                else:
                    results.append({
                        'filename': filename,
                        'error': 'Invalid image file',
                        'success': False
                    })
        
        response_time = time.time() - start_time
        api_logger.log_api_request('/api/batch_test', 'POST', 200, response_time)
        
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(results),
            'response_time': response_time
        })
        
    except Exception as e:
        response_time = time.time() - start_time
        api_logger.log_exception("Error in batch test", e)
        api_logger.log_api_request('/api/batch_test', 'POST', 500, response_time)
        
        return jsonify({
            'success': False,
            'error': str(e),
            'response_time': response_time
        }), 500
    
    finally:
        # Clean up temporary files
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {temp_path}: {e}")

@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    """Get detailed model information"""
    try:
        if not model_manager.is_model_loaded():
            return jsonify({
                'success': False,
                'error': 'No model loaded'
            }), 400
        
        info = model_manager.get_model_info()
        return jsonify({
            'success': True,
            'model_info': info
        })
        
    except Exception as e:
        api_logger.log_exception("Error getting model info", e)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get API metrics"""
    try:
        metrics = {
            'requests_total': 0,  # Would be tracked by middleware
            'model_loaded': model_manager.is_model_loaded(),
            'uptime': time.time() - app.config.get('START_TIME', time.time()),
            'memory_usage': 0,  # Would be tracked by monitoring
            'device': model_manager.get_device()
        }
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
        
    except Exception as e:
        api_logger.log_exception("Error getting metrics", e)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    """Handle file too large errors"""
    return jsonify({
        'success': False,
        'error': f'File too large. Maximum size is {api_config.max_content_length / 1024 / 1024:.0f}MB'
    }), 413

@app.errorhandler(404)
def handle_not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def handle_internal_error(error):
    """Handle 500 errors"""
    api_logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Store start time for uptime calculation
    app.config['START_TIME'] = time.time()
    
    logger.info("Starting MVTec Anomaly Detection API")
    logger.info(f"Configuration: {api_config.__dict__}")
    
    app.run(
        host=api_config.host,
        port=api_config.port,
        debug=api_config.debug,
        threaded=True
    )