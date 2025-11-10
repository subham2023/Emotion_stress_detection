"""
Configuration module for Emotion Recognition and Stress Detection System.

This module contains all configuration parameters including model architecture,
training hyperparameters, file paths, and feature flags.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Emotion classes
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = len(EMOTIONS)
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {idx: emotion for emotion, idx in EMOTION_TO_IDX.items()}

# Image preprocessing
IMAGE_SIZE_SMALL = (48, 48)  # For custom CNN
IMAGE_SIZE_LARGE = (224, 224)  # For transfer learning
IMAGE_CHANNELS = 1  # Grayscale for small model
IMAGE_CHANNELS_RGB = 3  # RGB for transfer learning

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

BATCH_SIZE = 64
LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.0001
EPOCHS = 100
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
TRAIN_SPLIT = 0.8

# Optimizer settings
OPTIMIZER = "adam"
LOSS_FUNCTION = "categorical_crossentropy"

# ============================================================================
# CALLBACKS CONFIGURATION
# ============================================================================

EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_MIN = 1e-7

# Model checkpoint
CHECKPOINT_MONITOR = "val_accuracy"
CHECKPOINT_MODE = "max"
CHECKPOINT_SAVE_BEST_ONLY = True

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

AUGMENTATION_CONFIG = {
    "rotation_range": 15,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "zoom_range": 0.2,
    "horizontal_flip": True,
    "brightness_range": [0.8, 1.2],
    "fill_mode": "nearest",
}

# ============================================================================
# FACE DETECTION
# ============================================================================

# Haar Cascade classifier path (will be loaded from OpenCV)
FACE_DETECTION_METHOD = "haar"  # Options: "haar", "mtcnn"
FACE_DETECTION_MIN_CONFIDENCE = 0.5
FACE_DETECTION_SCALE_FACTOR = 1.1
FACE_DETECTION_MIN_NEIGHBORS = 5

# ============================================================================
# STRESS LEVEL CONFIGURATION
# ============================================================================

# Stress level thresholds
STRESS_LEVELS = {
    "low": (0, 25),
    "moderate": (26, 50),
    "high": (51, 75),
    "critical": (76, 100),
}

# Emotion stress weights (higher = more stressful)
EMOTION_STRESS_WEIGHTS = {
    "angry": 0.9,
    "disgust": 0.7,
    "fear": 0.95,
    "happy": -0.8,  # Negative = reduces stress
    "sad": 0.8,
    "surprise": 0.3,
    "neutral": 0.0,
}

# ============================================================================
# PERFORMANCE TARGETS
# ============================================================================

TARGET_ACCURACY = 0.70  # 70% accuracy on test set
TARGET_INFERENCE_TIME = 0.1  # 100ms per frame
TARGET_UPTIME = 0.99  # 99% uptime

# ============================================================================
# FEATURE FLAGS
# ============================================================================

USE_GPU = True
MIXED_PRECISION_TRAINING = False
ENABLE_TENSORBOARD = True
ENABLE_LOGGING = True
DEBUG_MODE = False

# ============================================================================
# FILE PATHS FOR MODELS
# ============================================================================

CUSTOM_CNN_MODEL_PATH = MODELS_DIR / "custom_cnn_model.h5"
TRANSFER_LEARNING_MODEL_PATH = MODELS_DIR / "transfer_learning_model.h5"
BEST_MODEL_PATH = MODELS_DIR / "best_model.h5"
TRAINING_HISTORY_PATH = LOGS_DIR / "training_history.json"
CHECKPOINT_PATH = MODELS_DIR / "checkpoint"

# ============================================================================
# API CONFIGURATION
# ============================================================================

API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = False
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "gif"}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "emotion_detector.log"

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# FER2013 dataset paths (if using)
FER2013_TRAIN_PATH = RAW_DATA_DIR / "fer2013_train.csv"
FER2013_TEST_PATH = RAW_DATA_DIR / "fer2013_test.csv"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_model_path(model_type: str = "best") -> Path:
    """
    Get the path to a model file.

    Args:
        model_type: Type of model ("best", "custom_cnn", "transfer_learning")

    Returns:
        Path to the model file
    """
    paths = {
        "best": BEST_MODEL_PATH,
        "custom_cnn": CUSTOM_CNN_MODEL_PATH,
        "transfer_learning": TRANSFER_LEARNING_MODEL_PATH,
    }
    return paths.get(model_type, BEST_MODEL_PATH)


def get_stress_level(stress_score: float) -> str:
    """
    Get stress level category based on stress score.

    Args:
        stress_score: Stress score between 0 and 100

    Returns:
        Stress level category (low, moderate, high, critical)
    """
    for level, (min_val, max_val) in STRESS_LEVELS.items():
        if min_val <= stress_score <= max_val:
            return level
    return "critical"


def get_emotion_name(emotion_idx: int) -> str:
    """
    Get emotion name from index.

    Args:
        emotion_idx: Index of emotion

    Returns:
        Emotion name
    """
    return IDX_TO_EMOTION.get(emotion_idx, "unknown")


def get_emotion_index(emotion_name: str) -> int:
    """
    Get emotion index from name.

    Args:
        emotion_name: Name of emotion

    Returns:
        Index of emotion
    """
    return EMOTION_TO_IDX.get(emotion_name.lower(), -1)
