"""
Prediction Module for Emotion Recognition System.

This module handles single image prediction, batch prediction,
and real-time video prediction with visualization.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import (
    EMOTIONS,
    IMAGE_SIZE_SMALL,
    IMAGE_SIZE_LARGE,
    IMAGE_CHANNELS,
    IMAGE_CHANNELS_RGB,
    LOG_FILE,
    LOG_LEVEL,
    LOG_FORMAT,
)
from data_preprocessing import (
    FaceDetector,
    ImagePreprocessor,
    InvalidImageError,
    NoFaceDetectedError,
)
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
# CUSTOM EXCEPTIONS
# ============================================================================


class PredictionError(Exception):
    """Base exception for prediction errors."""

    pass


class ModelLoadError(PredictionError):
    """Raised when model cannot be loaded."""

    pass


# ============================================================================
# PREDICTOR
# ============================================================================


class EmotionPredictor:
    """
    Predict emotions from images and video frames.
    """

    def __init__(
        self,
        model_path: str,
        face_detection_method: str = "haar",
        use_grayscale: bool = True,
    ):
        """
        Initialize emotion predictor.

        Args:
            model_path: Path to trained model
            face_detection_method: Face detection method ("haar" or "mtcnn")
            use_grayscale: Use grayscale images if True

        Raises:
            ModelLoadError: If model cannot be loaded
        """
        self.model_path = model_path
        self.use_grayscale = use_grayscale

        # Load model
        try:
            self.model = keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}")

        # Initialize components
        self.face_detector = FaceDetector(method=face_detection_method)
        self.image_preprocessor = ImagePreprocessor()
        self.stress_analyzer = StressAnalyzer()

        # Determine input size from model
        self.input_shape = self.model.input_shape[1:3]
        logger.info(f"Model input shape: {self.input_shape}")

    def predict_single_image(
        self,
        image_path: str,
        return_visualization: bool = False,
    ) -> Dict[str, Any]:
        """
        Predict emotion from a single image file.

        Args:
            image_path: Path to image file
            return_visualization: Return visualization image if True

        Returns:
            Dictionary with prediction results

        Raises:
            PredictionError: If prediction fails
        """
        try:
            # Load image
            image = self.image_preprocessor.load_image(image_path)

            # Detect faces
            faces = self.face_detector.detect_faces(image)

            # Process each face
            results = []
            for face_idx, face_bbox in enumerate(faces):
                try:
                    # Preprocess face
                    face_image = self.image_preprocessor.preprocess_face(
                        image,
                        face_bbox,
                        target_size=self.input_shape,
                        grayscale=self.use_grayscale,
                    )

                    # Add batch dimension
                    face_batch = np.expand_dims(face_image, axis=0)

                    # Predict
                    emotion_probs = self.model.predict(face_batch, verbose=0)[0]

                    # Analyze stress
                    stress_result = self.stress_analyzer.analyze_frame(emotion_probs)

                    # Get dominant emotion
                    dominant_emotion_idx = np.argmax(emotion_probs)
                    confidence = float(emotion_probs[dominant_emotion_idx])

                    result = {
                        "face_index": face_idx,
                        "bbox": face_bbox,
                        "emotion_probabilities": {
                            EMOTIONS[i]: float(emotion_probs[i]) for i in range(len(EMOTIONS))
                        },
                        "dominant_emotion": EMOTIONS[dominant_emotion_idx],
                        "confidence": confidence,
                        "stress_score": stress_result["combined_stress"],
                        "stress_level": stress_result["stress_level"],
                    }

                    results.append(result)

                except (InvalidImageError, NoFaceDetectedError) as e:
                    logger.warning(f"Error processing face {face_idx}: {str(e)}")

            if not results:
                raise PredictionError("No valid faces detected in image")

            output = {
                "image_path": image_path,
                "num_faces": len(results),
                "results": results,
            }

            # Add visualization if requested
            if return_visualization:
                vis_image = self._visualize_predictions(image, results)
                output["visualization"] = vis_image

            return output

        except Exception as e:
            raise PredictionError(f"Single image prediction failed: {str(e)}")

    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Predict emotions from multiple images.

        Args:
            image_paths: List of image file paths
            batch_size: Batch size for processing

        Returns:
            List of prediction results

        Raises:
            PredictionError: If batch prediction fails
        """
        try:
            results = []

            for i, image_path in enumerate(image_paths):
                try:
                    logger.info(f"Processing image {i + 1}/{len(image_paths)}")
                    result = self.predict_single_image(image_path)
                    results.append(result)
                except PredictionError as e:
                    logger.warning(f"Failed to process {image_path}: {str(e)}")
                    results.append({"image_path": image_path, "error": str(e)})

            return results

        except Exception as e:
            raise PredictionError(f"Batch prediction failed: {str(e)}")

    def predict_frame(
        self,
        frame: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Predict emotion from a video frame.

        Args:
            frame: Video frame (BGR format)

        Returns:
            Dictionary with prediction results

        Raises:
            PredictionError: If prediction fails
        """
        try:
            # Detect faces
            faces = self.face_detector.detect_faces(frame)

            # Process each face
            results = []
            for face_idx, face_bbox in enumerate(faces):
                try:
                    # Preprocess face
                    face_image = self.image_preprocessor.preprocess_face(
                        frame,
                        face_bbox,
                        target_size=self.input_shape,
                        grayscale=self.use_grayscale,
                    )

                    # Add batch dimension
                    face_batch = np.expand_dims(face_image, axis=0)

                    # Predict
                    emotion_probs = self.model.predict(face_batch, verbose=0)[0]

                    # Analyze stress
                    stress_result = self.stress_analyzer.analyze_frame(emotion_probs)

                    # Get dominant emotion
                    dominant_emotion_idx = np.argmax(emotion_probs)
                    confidence = float(emotion_probs[dominant_emotion_idx])

                    result = {
                        "face_index": face_idx,
                        "bbox": face_bbox,
                        "emotion_probabilities": {
                            EMOTIONS[i]: float(emotion_probs[i]) for i in range(len(EMOTIONS))
                        },
                        "dominant_emotion": EMOTIONS[dominant_emotion_idx],
                        "confidence": confidence,
                        "stress_score": stress_result["combined_stress"],
                        "stress_level": stress_result["stress_level"],
                    }

                    results.append(result)

                except (InvalidImageError, NoFaceDetectedError) as e:
                    logger.debug(f"Error processing face {face_idx}: {str(e)}")

            output = {
                "num_faces": len(results),
                "results": results,
                "timestamp": None,
            }

            return output

        except Exception as e:
            raise PredictionError(f"Frame prediction failed: {str(e)}")

    def predict_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        skip_frames: int = 1,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Predict emotions from video file.

        Args:
            video_path: Path to video file
            output_path: Path to save output video (optional)
            skip_frames: Process every nth frame
            max_frames: Maximum number of frames to process

        Returns:
            Dictionary with video analysis results

        Raises:
            PredictionError: If video prediction fails
        """
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise PredictionError(f"Failed to open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"Video: {frame_count} frames, {fps} FPS, {width}x{height}")

            # Setup video writer if output path provided
            out = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Process frames
            frame_results = []
            frame_idx = 0
            processed_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames
                if frame_idx % skip_frames != 0:
                    frame_idx += 1
                    continue

                # Check max frames
                if max_frames and processed_frames >= max_frames:
                    break

                try:
                    # Predict
                    result = self.predict_frame(frame)
                    frame_results.append(result)

                    # Visualize if output video requested
                    if out:
                        vis_frame = self._visualize_predictions(frame, result["results"])
                        out.write(vis_frame)

                    processed_frames += 1
                    if processed_frames % 10 == 0:
                        logger.info(f"Processed {processed_frames} frames")

                except PredictionError as e:
                    logger.debug(f"Frame {frame_idx} prediction failed: {str(e)}")

                frame_idx += 1

            # Release resources
            cap.release()
            if out:
                out.release()

            # Get temporal analysis
            temporal_analysis = self.stress_analyzer.get_temporal_analysis()

            output = {
                "video_path": video_path,
                "total_frames": frame_count,
                "processed_frames": processed_frames,
                "fps": fps,
                "frame_results": frame_results,
                "temporal_analysis": temporal_analysis,
            }

            if output_path:
                output["output_video"] = output_path

            return output

        except Exception as e:
            raise PredictionError(f"Video prediction failed: {str(e)}")

    def _visualize_predictions(
        self,
        image: np.ndarray,
        results: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Visualize predictions on image.

        Args:
            image: Input image (BGR format)
            results: List of prediction results

        Returns:
            Visualized image
        """
        vis_image = image.copy()

        for result in results:
            x, y, w, h = result["bbox"]

            # Draw bounding box
            color = self._get_emotion_color(result["dominant_emotion"])
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)

            # Draw label
            label = f"{result['dominant_emotion']} ({result['confidence']:.2f})"
            cv2.putText(
                vis_image,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            # Draw stress level
            stress_label = f"Stress: {result['stress_level']} ({result['stress_score']:.1f})"
            cv2.putText(
                vis_image,
                stress_label,
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        return vis_image

    @staticmethod
    def _get_emotion_color(emotion: str) -> Tuple[int, int, int]:
        """
        Get color for emotion visualization.

        Args:
            emotion: Emotion name

        Returns:
            BGR color tuple
        """
        colors = {
            "angry": (0, 0, 255),  # Red
            "disgust": (0, 165, 255),  # Orange
            "fear": (128, 0, 128),  # Purple
            "happy": (0, 255, 0),  # Green
            "sad": (255, 0, 0),  # Blue
            "surprise": (0, 255, 255),  # Yellow
            "neutral": (128, 128, 128),  # Gray
        }
        return colors.get(emotion, (255, 255, 255))  # White default


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def create_predictor(model_path: str) -> EmotionPredictor:
    """
    Create an emotion predictor instance.

    Args:
        model_path: Path to trained model

    Returns:
        EmotionPredictor instance

    Raises:
        ModelLoadError: If model cannot be loaded
    """
    return EmotionPredictor(model_path)
