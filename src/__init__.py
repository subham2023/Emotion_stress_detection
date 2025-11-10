"""
Emotion Recognition and Stress Detection System

A complete, production-ready facial emotion recognition and stress level
detection system with web interface and model training pipeline.
"""

__version__ = "1.0.0"
__author__ = "Emotion Detection Team"

from .config import (
    EMOTIONS,
    NUM_CLASSES,
    IMAGE_SIZE_SMALL,
    IMAGE_SIZE_LARGE,
    get_emotion_name,
    get_emotion_index,
    get_stress_level,
)
from .data_preprocessing import (
    FaceDetector,
    ImagePreprocessor,
    DataAugmentor,
    DatasetManager,
    PreprocessingError,
    InvalidImageError,
    NoFaceDetectedError,
    DatasetError,
)

__all__ = [
    "EMOTIONS",
    "NUM_CLASSES",
    "IMAGE_SIZE_SMALL",
    "IMAGE_SIZE_LARGE",
    "get_emotion_name",
    "get_emotion_index",
    "get_stress_level",
    "FaceDetector",
    "ImagePreprocessor",
    "DataAugmentor",
    "DatasetManager",
    "PreprocessingError",
    "InvalidImageError",
    "NoFaceDetectedError",
    "DatasetError",
]
