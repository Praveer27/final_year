"""
Flask Backend API for Gesture2Speech
Provides REST API endpoints for the React frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import config, setup_logger
from src.models.gesture_model import create_model
from src.preprocessing import KeypointExtractor

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300 MB

# Setup logging
setup_logger(log_dir=Path("logs"), log_level="INFO")

# Global variables
model = None
keypoint_extractor = None
device = None

# gesture->id mapping (preferred) and id->gesture for response
label_map = None
id_to_label = None

# avoid repeated init
_initialized = False


def _build_id_to_label(label_map_dict):
    if not label_map_dict:
        return None
    # label_map expected: {gesture_name: class_id}
    return {int(v): str(k) for k, v in label_map_dict.items()}


def _handlist_to_vec63(keypoints_list):
    """
    Convert MediaPipe hand output to a 63-dim vector (21*3).
    Uses first detected hand. If not available -> zeros.
    """
    if not keypoints_list:
        return np.zeros((63,), dtype=np.float32)

    kps = keypoints_list[0].get("keypoints", [])
    if len(kps) != 21:
        return np.zeros((63,), dtype=np.float32)

    arr = np.array([[p["x"], p["y"], p["z"]] for p in kps], dtype=np.float32)  # (21,3)
    return arr.reshape(-1)  # (63,)


def initialize_model():
    """Initialize the gesture recognition model + keypoint extractor"""
    global model, keypoint_extractor, device, label_map, id_to_label

    try:
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Initialize keypoint extractor
        keypoint_extractor = KeypointExtractor(
            max_num_hands=config.get("mediapipe.max_num_hands", 2),
            min_detection_confidence=config.get("mediapipe.min_detection_confidence", 0.3),
            min_tracking_confidence=config.get("mediapipe.min_tracking_confidence", 0.3),
            model_complexity=config.get("mediapipe.model_complexity", 1),
            static_image_mode=config.get("mediapipe.static_image_mode", False),
        )

        # Default label map from config (may be None / or not matching)
        try:
            label_map = config.get_label_mapping()
        except Exception:
            label_map = None

        # If config label_map exists, infer classes from it; otherwise fallback to config num_classes
        num_classes = len(label_map) if label_map else int(config.get("dataset.num_classes", config.get_num_classes()))

        model = create_model(
            model_type=config.get("model.architecture", "lstm"),
            input_size=63,
            num_classes=num_classes,
            hidden_size=config.get("model.lstm.hidden_size", 256),
            num_layers=config.get("model.lstm.num_layers", 3),
            dropout=config.get("model.lstm.dropout", 0.3),
            bidirectional=config.get("model.lstm.bidirectional", True),
        )

        # Load trained model if exists
        model_path = Path("models/checkpoints/best_model.pth")
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=device)

            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "model_state" in checkpoint:
                # tolerate alternative key name
                model.load_state_dict(checkpoint["model_state"])
            else:
                raise KeyError("Checkpoint missing model weights (expected model_state_dict or model_state)")

            # Prefer label_map from checkpoint if present
            if "label_map" in checkpoint and checkpoint["label_map"]:
                label_map = checkpoint["label_map"]

            id_to_label = _build_id_to_label(label_map)

            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning("No trained model found at models/checkpoints/best_model.pth. Using untrained model.")
            id_to_label = _build_id_to_label(label_map)

        model.to(device)
        model.eval()

        logger.info("Model initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        model = None
        keypoint_extractor = None
        id_to_label = None
        return False


@app.before_request
def ensure_initialized():
    """Ensures model/extractor are initialized even when running via `flask run`."""
    global _initialized
    if not _initialized:
        _initialized = initialize_model()


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": model is not None,
            "device": str(device) if device else "not initialized",
        }
    )


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify(
        {
            "project_name": config.get("project.name"),
            "version": config.get("project.version"),
            "num_classes": config.get_num_classes(),
            "gestures": config.get("dataset.categories"),
            "model_architecture": config.get("model.architecture"),
        }
    )


@app.route("/api/predict/image", methods=["POST"])
def predict_image():
    """
    Predict gesture from an uploaded image.
    Note: LSTM expects sequences; here we pad a (seq_len,63) sequence with zeros and put this frame at the end.
    """
    try:
        if model is None or keypoint_extractor is None or device is None:
            return jsonify({"error": "Model not initialized"}), 500

        if "image" not in request.files:
            return jsonify({"error": "No image provided (expected form field name: image)"}), 400

        file = request.files["image"]
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Invalid image"}), 400

        keypoints_list, _ = keypoint_extractor.extract_from_image(image)
        if not keypoints_list:
            return jsonify(
                {
                    "prediction": "No hands detected",
                    "confidence": 0.0,
                    "hands_detected": 0,
                }
            )

        # Build padded sequence
        seq_len = 45
        x0 = _handlist_to_vec63(keypoints_list)
        seq = np.zeros((seq_len, 63), dtype=np.float32)
        seq[-1] = x0

        xb = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1,45,63)

        with torch.no_grad():
            logits = model(xb)  # (1,C)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_id = int(np.argmax(probs))
        conf = float(probs[pred_id])

        pred_label = str(pred_id)
        if id_to_label and pred_id in id_to_label:
            pred_label = id_to_label[pred_id]

        return jsonify(
            {
                "prediction": pred_label,
                "confidence": conf,
                "hands_detected": len(keypoints_list),
            }
        )

    except Exception as e:
        logger.error(f"Error in predict_image: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/video", methods=["POST"])
def predict_video():
    """
    Placeholder: your frontend may call this for uploaded videos.
    Implementing full video->sequence inference is next step.
    """
    return jsonify(
        {
            "prediction": "Video inference not implemented yet",
            "confidence": 0.0,
            "message": "Use /api/predict/image for now or implement video sequence inference.",
        }
    )


@app.route("/api/train/status", methods=["GET"])
def training_status():
    history_path = Path("models/checkpoints/training_history.json")
    if history_path.exists():
        import json

        with open(history_path, "r") as f:
            history = json.load(f)
        return jsonify({"status": "completed", "history": history})

    return jsonify({"status": "not_started", "message": "No training history found"})


@app.route("/api/dataset/info", methods=["GET"])
def dataset_info():
    videos_path = Path("data/videos")
    frames_path = Path("data/frames")
    keypoints_path = Path("data/keypoints")

    def count_files(path, extensions):
        if not path.exists():
            return 0
        count = 0
        for ext in extensions:
            count += len(list(path.rglob(f"*{ext}")))
        return count

    return jsonify(
        {
            "videos": count_files(videos_path, [".mp4", ".avi", ".mov", ".MOV", ".mkv", ".MKV"]),
            "frames": count_files(frames_path, [".jpg", ".jpeg", ".png"]),
            "keypoints": count_files(keypoints_path, [".json"]),
            "categories": len(list(videos_path.iterdir())) if videos_path.exists() else 0,
        }
    )


@app.route("/api/models/list", methods=["GET"])
def list_models():
    models_path = Path("models/checkpoints")
    if not models_path.exists():
        return jsonify({"models": []})

    models = []
    for model_file in list(models_path.glob("*.pth")) + list(models_path.glob("*.pt")):
        models.append(
            {
                "name": model_file.name,
                "size": model_file.stat().st_size,
                "modified": model_file.stat().st_mtime,
            }
        )
    return jsonify({"models": models})


if __name__ == "__main__":
    logger.info("Starting Gesture2Speech Flask Backend...")
    ok = initialize_model()
    logger.info(f"Model init ok={ok}")

    app.run(host="0.0.0.0", port=5001, debug=True)