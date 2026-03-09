"""
Flask Backend API for Gesture2Speech
Provides REST API endpoints for the React frontend
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import config, setup_logger
from src.models.gesture_model import create_model
from src.preprocessing import KeypointExtractor
from loguru import logger

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Setup logging
setup_logger(log_dir=Path("logs"), log_level="INFO")

# Global variables
model = None
keypoint_extractor = None
device = None
label_mapping = None

def initialize_model():
    """Initialize the gesture recognition model"""
    global model, keypoint_extractor, device, label_mapping
    
    try:
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Initialize keypoint extractor
        keypoint_extractor = KeypointExtractor(
            max_num_hands=config.get('mediapipe.max_num_hands', 2),
            min_detection_confidence=config.get('mediapipe.min_detection_confidence', 0.7)
        )
        
        # Get label mapping
        label_mapping = config.get_label_mapping()
        num_classes = len(label_mapping)
        
        # Create model
        model = create_model(
            model_type=config.get('model.architecture', 'lstm'),
            input_size=63,  # 21 keypoints * 3 coordinates
            num_classes=num_classes,
            hidden_size=config.get('model.lstm.hidden_size', 256),
            num_layers=config.get('model.lstm.num_layers', 3),
            dropout=config.get('model.lstm.dropout', 0.3),
            bidirectional=config.get('model.lstm.bidirectional', True)
        )
        
        # Load trained model if exists
        model_path = Path('models/checkpoints/best_model.pth')
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning("No trained model found. Using untrained model.")
        
        model.to(device)
        model.eval()
        
        logger.info("Model initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        return False


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'not initialized'
    })


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get configuration information"""
    return jsonify({
        'project_name': config.get('project.name'),
        'version': config.get('project.version'),
        'num_classes': config.get_num_classes(),
        'gestures': config.get('dataset.categories'),
        'model_architecture': config.get('model.architecture')
    })


@app.route('/api/predict/image', methods=['POST'])
def predict_image():
    """Predict gesture from uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        # Extract keypoints
        keypoints_list, _ = keypoint_extractor.extract_from_image(image)
        
        if not keypoints_list:
            return jsonify({
                'prediction': 'No hands detected',
                'confidence': 0.0,
                'hands_detected': 0
            })
        
        # Prepare input for model (simplified - needs proper implementation)
        # This is a placeholder - full implementation would process keypoints properly
        
        return jsonify({
            'prediction': 'Model inference not fully implemented',
            'confidence': 0.0,
            'hands_detected': len(keypoints_list),
            'message': 'Keypoint extraction successful. Full inference coming soon.'
        })
        
    except Exception as e:
        logger.error(f"Error in predict_image: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/video', methods=['POST'])
def predict_video():
    """Predict gesture from uploaded video"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video provided'}), 400
        
        file = request.files['video']
        
        # Save temporarily
        temp_path = Path('temp_video.mp4')
        file.save(temp_path)
        
        # Process video (simplified)
        cap = cv2.VideoCapture(str(temp_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Clean up
        temp_path.unlink()
        
        return jsonify({
            'prediction': 'Video processing not fully implemented',
            'confidence': 0.0,
            'frames_processed': frame_count,
            'message': 'Video uploaded successfully. Full processing coming soon.'
        })
        
    except Exception as e:
        logger.error(f"Error in predict_video: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/train/status', methods=['GET'])
def training_status():
    """Get training status"""
    history_path = Path('models/checkpoints/training_history.json')
    
    if history_path.exists():
        import json
        with open(history_path, 'r') as f:
            history = json.load(f)
        return jsonify({
            'status': 'completed',
            'history': history
        })
    else:
        return jsonify({
            'status': 'not_started',
            'message': 'No training history found'
        })


@app.route('/api/dataset/info', methods=['GET'])
def dataset_info():
    """Get dataset information"""
    videos_path = Path('data/videos')
    frames_path = Path('data/frames')
    keypoints_path = Path('data/keypoints')
    
    def count_files(path, extensions):
        if not path.exists():
            return 0
        count = 0
        for ext in extensions:
            count += len(list(path.rglob(f'*{ext}')))
        return count
    
    return jsonify({
        'videos': count_files(videos_path, ['.mp4', '.avi', '.mov']),
        'frames': count_files(frames_path, ['.jpg', '.jpeg', '.png']),
        'keypoints': count_files(keypoints_path, ['.json']),
        'categories': len(list(videos_path.iterdir())) if videos_path.exists() else 0
    })


@app.route('/api/models/list', methods=['GET'])
def list_models():
    """List available models"""
    models_path = Path('models/checkpoints')
    
    if not models_path.exists():
        return jsonify({'models': []})
    
    models = []
    for model_file in models_path.glob('*.pth'):
        models.append({
            'name': model_file.name,
            'size': model_file.stat().st_size,
            'modified': model_file.stat().st_mtime
        })
    
    return jsonify({'models': models})


if __name__ == '__main__':
    logger.info("Starting Gesture2Speech Flask Backend...")
    
    # Initialize model
    if initialize_model():
        logger.info("Model initialized successfully")
    else:
        logger.warning("Model initialization failed. Some features may not work.")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )

# Made with Bob
