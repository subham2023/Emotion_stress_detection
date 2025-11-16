"""
Inference Server for Real-time Emotion Detection.

Provides Flask endpoints for image and video frame processing
with real-time emotion and stress analysis.
"""

import logging
import os
import base64
import json
from io import BytesIO
from typing import Dict, Any, Optional

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room

from config import (
    EMOTIONS,
    IMAGE_SIZE_SMALL,
    LOG_FILE,
    LOG_LEVEL,
    LOG_FORMAT,
)
from predict import EmotionPredictor, PredictionError
from stress_analyzer import StressAnalyzer

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================================
# FLASK APP SETUP
# ============================================================================

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ============================================================================
# GLOBAL STATE
# ============================================================================

# Load model on startup
MODEL_PATH = os.getenv("MODEL_PATH", "models/emotion_model.h5")
predictor: Optional[EmotionPredictor] = None
stress_analyzer = StressAnalyzer()

# Active sessions
active_sessions: Dict[str, Dict[str, Any]] = {}


def load_model():
    """Load emotion detection model."""
    global predictor
    try:
        if os.path.exists(MODEL_PATH):
            predictor = EmotionPredictor(MODEL_PATH)
            logger.info(f"Model loaded from {MODEL_PATH}")
        else:
            logger.warning(f"Model not found at {MODEL_PATH}")
            logger.info("Using mock predictions for demonstration")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.info("Using mock predictions for demonstration")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_mock_prediction():
    """Get mock prediction for demonstration."""
    emotions = ["happy", "neutral", "sad", "angry"]
    emotion = np.random.choice(emotions)
    confidence = 0.7 + np.random.random() * 0.3
    stress_score = np.random.random() * 100

    emotion_probs = {
        "angry": np.random.random(),
        "disgust": np.random.random(),
        "fear": np.random.random(),
        "happy": np.random.random(),
        "sad": np.random.random(),
        "surprise": np.random.random(),
        "neutral": np.random.random(),
    }
    # Normalize
    total = sum(emotion_probs.values())
    emotion_probs = {k: v / total for k, v in emotion_probs.items()}

    return {
        "dominant_emotion": emotion,
        "confidence": float(confidence),
        "stress_score": float(stress_score),
        "stress_level": (
            "low"
            if stress_score < 25
            else (
                "moderate"
                if stress_score < 50
                else "high" if stress_score < 75 else "critical"
            )
        ),
        "emotion_probabilities": {k: float(v) for k, v in emotion_probs.items()},
        "faces_detected": 1,
    }


def predict_frame(frame: np.ndarray) -> Dict[str, Any]:
    """
    Predict emotion from a frame.

    Args:
        frame: Image frame (BGR format)

    Returns:
        Prediction result
    """
    try:
        if predictor is None:
            # Use mock prediction if model not loaded
            return get_mock_prediction()

        result = predictor.predict_frame(frame)

        if result["num_faces"] == 0:
            return {
                "error": "No faces detected",
                "num_faces": 0,
            }

        # Get first face result
        face_result = result["results"][0]

        return {
            "dominant_emotion": face_result["dominant_emotion"],
            "confidence": face_result["confidence"],
            "stress_score": face_result["stress_score"],
            "stress_level": face_result["stress_level"],
            "emotion_probabilities": face_result["emotion_probabilities"],
            "faces_detected": result["num_faces"],
        }

    except PredictionError as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        return {"error": "Prediction failed"}


# ============================================================================
# REST API ENDPOINTS
# ============================================================================


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_loaded": predictor is not None})


@app.route("/api/predict/image", methods=["POST"])
def predict_image():
    """
    Predict emotion from uploaded image.

    Expected: multipart/form-data with 'image' file
    Returns: JSON with emotion predictions
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Read image
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Predict
        result = predict_frame(image)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Image prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/base64", methods=["POST"])
def predict_base64():
    """
    Predict emotion from base64 encoded image.

    Expected: JSON with 'image' field containing base64 data
    Returns: JSON with emotion predictions
    """
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode base64
        image_data = base64.b64decode(data["image"])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Predict
        result = predict_frame(image)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Base64 prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/model/info", methods=["GET"])
def model_info():
    """Get information about loaded model."""
    if predictor is None:
        return jsonify(
            {
                "loaded": False,
                "message": "Model not loaded, using mock predictions",
            }
        )

    return jsonify(
        {
            "loaded": True,
            "model_path": MODEL_PATH,
            "input_shape": predictor.input_shape,
            "emotions": EMOTIONS,
        }
    )


# ============================================================================
# WEBSOCKET EVENTS
# ============================================================================


@socketio.on("connect")
def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid}")
    emit("response", {"data": "Connected to inference server"})


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")
    # Clean up session
    if request.sid in active_sessions:
        del active_sessions[request.sid]


@socketio.on("start_session")
def handle_start_session(data):
    """
    Start a new detection session.

    Expected data:
    {
        "session_type": "webcam" | "video",
        "session_id": number (from backend)
    }
    """
    try:
        session_id = data.get("session_id")
        session_type = data.get("session_type", "webcam")

        # Initialize session
        active_sessions[request.sid] = {
            "session_id": session_id,
            "session_type": session_type,
            "frame_count": 0,
            "emotion_history": [],
            "stress_history": [],
            "start_time": None,
        }

        logger.info(f"Session started: {request.sid} (type: {session_type})")
        emit("session_started", {"session_id": session_id})

    except Exception as e:
        logger.error(f"Error starting session: {str(e)}")
        emit("error", {"message": str(e)})


@socketio.on("predict_frame")
def handle_predict_frame(data):
    """
    Process a single frame for emotion detection.

    Expected data:
    {
        "image": base64 encoded image,
        "timestamp": optional timestamp
    }
    """
    try:
        if request.sid not in active_sessions:
            emit("error", {"message": "No active session"})
            return

        # Decode image
        image_data = base64.b64decode(data["image"])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            emit("error", {"message": "Invalid image format"})
            return

        # Predict
        result = predict_frame(image)

        # Update session
        session = active_sessions[request.sid]
        session["frame_count"] += 1

        if "error" not in result:
            session["emotion_history"].append(result["dominant_emotion"])
            session["stress_history"].append(result["stress_score"])

        # Emit result
        emit(
            "prediction_result",
            {
                "frame_number": session["frame_count"],
                "result": result,
                "timestamp": data.get("timestamp"),
            },
        )

    except Exception as e:
        logger.error(f"Frame prediction error: {str(e)}")
        emit("error", {"message": str(e)})


@socketio.on("end_session")
def handle_end_session(data):
    """
    End detection session and return summary.

    Expected data:
    {
        "session_id": number
    }
    """
    try:
        if request.sid not in active_sessions:
            emit("error", {"message": "No active session"})
            return

        session = active_sessions[request.sid]

        # Calculate summary
        summary = {
            "session_id": session["session_id"],
            "total_frames": session["frame_count"],
            "dominant_emotion": (
                max(
                    set(session["emotion_history"]),
                    key=session["emotion_history"].count,
                )
                if session["emotion_history"]
                else None
            ),
            "average_stress": (
                np.mean(session["stress_history"])
                if session["stress_history"]
                else None
            ),
            "max_stress": (
                np.max(session["stress_history"]) if session["stress_history"] else None
            ),
            "min_stress": (
                np.min(session["stress_history"]) if session["stress_history"] else None
            ),
        }

        logger.info(f"Session ended: {request.sid}")
        emit("session_ended", summary)

        # Clean up
        del active_sessions[request.sid]

    except Exception as e:
        logger.error(f"Error ending session: {str(e)}")
        emit("error", {"message": str(e)})


@socketio.on("get_session_stats")
def handle_get_session_stats():
    """Get current session statistics."""
    try:
        if request.sid not in active_sessions:
            emit("error", {"message": "No active session"})
            return

        session = active_sessions[request.sid]

        stats = {
            "frame_count": session["frame_count"],
            "emotion_history": session["emotion_history"],
            "stress_history": session["stress_history"],
        }

        emit("session_stats", stats)

    except Exception as e:
        logger.error(f"Error getting session stats: {str(e)}")
        emit("error", {"message": str(e)})


# ============================================================================
# ERROR HANDLERS
# ============================================================================


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Start inference server."""
    # Load model
    load_model()

    # Start server
    port = int(os.getenv("INFERENCE_SERVER_PORT", 5000))
    debug = os.getenv("FLASK_ENV") == "development"

    logger.info(f"Starting inference server on port {port}")
    socketio.run(app, host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    main()
