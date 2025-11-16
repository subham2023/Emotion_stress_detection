"""
Data Preprocessing Module for Emotion Recognition System.

This module handles face detection, image preprocessing, data augmentation,
and dataset management for the emotion recognition pipeline.
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import (
    EMOTIONS,
    IMAGE_SIZE_SMALL,
    AUGMENTATION_CONFIG,
    FACE_DETECTION_MIN_CONFIDENCE,
    FACE_DETECTION_SCALE_FACTOR,
    FACE_DETECTION_MIN_NEIGHBORS,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    TEST_SPLIT,
    LOG_FILE,
    LOG_LEVEL,
    LOG_FORMAT,
)

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


class PreprocessingError(Exception):
    """Base exception for preprocessing errors."""


class InvalidImageError(PreprocessingError):
    """Raised when image is invalid or corrupted."""


class NoFaceDetectedError(PreprocessingError):
    """Raised when no face is detected in image."""


class DatasetError(PreprocessingError):
    """Raised when dataset has issues."""


# ============================================================================
# FACE DETECTION
# ============================================================================


class FaceDetector:
    """
    Face detection using Haar Cascade or MTCNN.

    Attributes:
        method: Detection method ("haar" or "mtcnn")
        cascade_path: Path to Haar Cascade XML file
        min_confidence: Minimum confidence for detection
    """

    def __init__(self, method: str = "haar"):
        """
        Initialize face detector.

        Args:
            method: Detection method ("haar" or "mtcnn")

        Raises:
            ValueError: If method is not supported
        """
        if method not in ["haar", "mtcnn"]:
            raise ValueError(f"Unsupported detection method: {method}")

        self.method = method
        self.min_confidence = FACE_DETECTION_MIN_CONFIDENCE

        if method == "haar":
            self._init_haar_cascade()
        elif method == "mtcnn":
            self._init_mtcnn()

    def _init_haar_cascade(self) -> None:
        """Initialize Haar Cascade classifier."""
        cascade_path = (
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.cascade = cv2.CascadeClassifier(cascade_path)

        if self.cascade.empty():
            raise PreprocessingError("Failed to load Haar Cascade classifier")

        logger.info("Haar Cascade classifier loaded successfully")

    def _init_mtcnn(self) -> None:
        """Initialize MTCNN detector."""
        try:
            from mtcnn import MTCNN

            self.detector = MTCNN()
            logger.info("MTCNN detector initialized successfully")
        except ImportError:
            raise PreprocessingError(
                "MTCNN not installed. Install with: pip install mtcnn"
            )

    def detect_faces(
        self, image: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image.

        Args:
            image: Input image (BGR format)

        Returns:
            List of face bounding boxes [(x, y, w, h), ...]

        Raises:
            InvalidImageError: If image is invalid
            NoFaceDetectedError: If no faces are detected
        """
        if image is None or image.size == 0:
            raise InvalidImageError("Invalid image provided")

        try:
            if self.method == "haar":
                return self._detect_haar(image)
            elif self.method == "mtcnn":
                return self._detect_mtcnn(image)
        except Exception as e:
            raise InvalidImageError(f"Face detection failed: {str(e)}")

    def _detect_haar(
        self, image: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar Cascade."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_DETECTION_SCALE_FACTOR,
            minNeighbors=FACE_DETECTION_MIN_NEIGHBORS,
            minSize=(30, 30),
        )

        if len(faces) == 0:
            raise NoFaceDetectedError("No faces detected in image")

        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

    def _detect_mtcnn(
        self, image: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MTCNN."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(rgb_image)

        if not detections:
            raise NoFaceDetectedError("No faces detected in image")

        faces = []
        for detection in detections:
            if detection["confidence"] >= self.min_confidence:
                x, y, w, h = detection["box"]
                faces.append((max(0, x), max(0, y), w, h))

        if not faces:
            raise NoFaceDetectedError("No confident faces detected in image")

        return faces


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================


class ImagePreprocessor:
    """
    Image preprocessing for emotion recognition.

    Handles resizing, normalization, and format conversion.
    """

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        Load image from file.

        Args:
            image_path: Path to image file

        Returns:
            Image as numpy array (BGR format)

        Raises:
            InvalidImageError: If image cannot be loaded
        """
        try:
            if not os.path.exists(image_path):
                raise InvalidImageError(f"Image file not found: {image_path}")

            image = cv2.imread(image_path)
            if image is None:
                raise InvalidImageError(f"Failed to load image: {image_path}")

            return image
        except Exception as e:
            raise InvalidImageError(f"Error loading image: {str(e)}")

    @staticmethod
    def preprocess_face(
        image: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        target_size: Tuple[int, int] = IMAGE_SIZE_SMALL,
        grayscale: bool = True,
    ) -> np.ndarray:
        """
        Extract and preprocess face from image.

        Args:
            image: Input image (BGR format)
            face_bbox: Face bounding box (x, y, w, h)
            target_size: Target image size (height, width)
            grayscale: Convert to grayscale if True

        Returns:
            Preprocessed face image

        Raises:
            InvalidImageError: If preprocessing fails
        """
        try:
            x, y, w, h = face_bbox

            # Extract face region with small margin
            margin = int(0.1 * max(w, h))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.shape[1], x + w + margin)
            y2 = min(image.shape[0], y + h + margin)

            face = image[y1:y2, x1:x2]

            if face.size == 0:
                raise InvalidImageError("Face region is empty")

            # Resize face
            face_resized = cv2.resize(face, target_size)

            # Convert to grayscale if needed
            if grayscale and len(face_resized.shape) == 3:
                face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

            # Normalize pixel values
            face_normalized = face_resized.astype(np.float32) / 255.0

            return face_normalized
        except Exception as e:
            raise InvalidImageError(f"Face preprocessing failed: {str(e)}")

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.

        Args:
            image: Input image

        Returns:
            Normalized image
        """
        if image.dtype != np.float32 and image.dtype != np.float64:
            image = image.astype(np.float32)

        if image.max() > 1.0:
            image = image / 255.0

        return image

    @staticmethod
    def equalize_histogram(image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization to improve contrast.

        Args:
            image: Input grayscale image

        Returns:
            Histogram equalized image
        """
        if len(image.shape) == 2:
            # Grayscale image
            image_uint8 = (image * 255).astype(np.uint8)
            equalized = cv2.equalizeHist(image_uint8)
            return equalized.astype(np.float32) / 255.0
        else:
            # Color image - equalize each channel
            image_uint8 = (image * 255).astype(np.uint8)
            channels = cv2.split(image_uint8)
            equalized_channels = [cv2.equalizeHist(ch) for ch in channels]
            equalized = cv2.merge(equalized_channels)
            return equalized.astype(np.float32) / 255.0


# ============================================================================
# DATA AUGMENTATION
# ============================================================================


class DataAugmentor:
    """
    Data augmentation for training dataset.

    Applies transformations like rotation, flip, zoom, brightness adjustment.
    """

    @staticmethod
    def rotate(image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by angle.

        Args:
            image: Input image
            angle: Rotation angle in degrees

        Returns:
            Rotated image
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        return rotated

    @staticmethod
    def flip(image: np.ndarray, horizontal: bool = True) -> np.ndarray:
        """
        Flip image horizontally or vertically.

        Args:
            image: Input image
            horizontal: Flip horizontally if True, vertically if False

        Returns:
            Flipped image
        """
        if horizontal:
            return cv2.flip(image, 1)
        else:
            return cv2.flip(image, 0)

    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image brightness.

        Args:
            image: Input image (float32, [0, 1])
            factor: Brightness factor (0.5 = darker, 1.5 = brighter)

        Returns:
            Brightness adjusted image
        """
        adjusted = np.clip(image * factor, 0, 1)
        return adjusted.astype(np.float32)

    @staticmethod
    def add_noise(image: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """
        Add Gaussian noise to image.

        Args:
            image: Input image (float32, [0, 1])
            noise_level: Standard deviation of noise

        Returns:
            Image with added noise
        """
        noise = np.random.normal(0, noise_level, image.shape)
        noisy = np.clip(image + noise, 0, 1)
        return noisy.astype(np.float32)

    @staticmethod
    def zoom(image: np.ndarray, zoom_factor: float) -> np.ndarray:
        """
        Zoom image in or out.

        Args:
            image: Input image
            zoom_factor: Zoom factor (1.0 = no zoom, 1.2 = 20% zoom in)

        Returns:
            Zoomed image
        """
        h, w = image.shape[:2]
        new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)

        # Resize to zoomed size
        zoomed = cv2.resize(image, (new_w, new_h))

        # Create output image
        output = np.zeros_like(image)
        y_offset = (h - new_h) // 2
        x_offset = (w - new_w) // 2

        output[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = (
            zoomed
        )

        return output

    @staticmethod
    def augment_image(
        image: np.ndarray, augmentation_config: Dict[str, Any]
    ) -> List[np.ndarray]:
        """
        Apply multiple augmentations to image.

        Args:
            image: Input image
            augmentation_config: Augmentation configuration

        Returns:
            List of augmented images
        """
        augmented_images = [image]  # Include original

        # Rotation
        for angle in np.linspace(
            -augmentation_config.get("rotation_range", 15),
            augmentation_config.get("rotation_range", 15),
            3,
        ):
            if angle != 0:
                augmented_images.append(DataAugmentor.rotate(image, angle))

        # Brightness
        brightness_range = augmentation_config.get(
            "brightness_range", [0.8, 1.2]
        )
        for factor in np.linspace(brightness_range[0], brightness_range[1], 2):
            if factor != 1.0:
                augmented_images.append(
                    DataAugmentor.adjust_brightness(image, factor)
                )

        # Flip
        if augmentation_config.get("horizontal_flip", True):
            augmented_images.append(DataAugmentor.flip(image, horizontal=True))

        # Zoom
        zoom_range = augmentation_config.get("zoom_range", 0.2)
        for zoom_factor in np.linspace(1.0 - zoom_range, 1.0 + zoom_range, 2):
            if zoom_factor != 1.0:
                augmented_images.append(DataAugmentor.zoom(image, zoom_factor))

        return augmented_images


# ============================================================================
# DATASET MANAGEMENT
# ============================================================================


class DatasetManager:
    """
    Manage dataset loading, splitting, and preprocessing.

    Handles loading images, detecting faces, preprocessing, and splitting
    into train/validation/test sets.
    """

    def __init__(self, face_detection_method: str = "haar"):
        """
        Initialize dataset manager.

        Args:
            face_detection_method: Face detection method ("haar" or "mtcnn")
        """
        self.face_detector = FaceDetector(method=face_detection_method)
        self.image_preprocessor = ImagePreprocessor()
        self.data_augmentor = DataAugmentor()

    def load_and_preprocess_dataset(
        self,
        data_dir: str,
        target_size: Tuple[int, int] = IMAGE_SIZE_SMALL,
        grayscale: bool = True,
        augment: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess dataset from directory structure.

        Directory structure expected:
        data_dir/
            emotion1/
                image1.jpg
                image2.jpg
            emotion2/
                image1.jpg
                ...

        Args:
            data_dir: Root directory containing emotion subdirectories
            target_size: Target image size
            grayscale: Convert to grayscale if True
            augment: Apply data augmentation if True

        Returns:
            Tuple of (images, labels) as numpy arrays

        Raises:
            DatasetError: If dataset is invalid or empty
        """
        images = []
        labels = []
        error_count = 0

        data_path = Path(data_dir)
        if not data_path.exists():
            raise DatasetError(f"Data directory not found: {data_dir}")

        # Get emotion directories
        emotion_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        if not emotion_dirs:
            raise DatasetError(
                f"No emotion subdirectories found in {data_dir}"
            )

        logger.info(f"Found {len(emotion_dirs)} emotion categories")

        # Load images from each emotion directory
        for emotion_dir in emotion_dirs:
            emotion_name = emotion_dir.name.lower()

            if emotion_name not in EMOTIONS:
                logger.warning(
                    f"Unknown emotion category: {emotion_name}, skipping"
                )
                continue

            emotion_label = EMOTIONS.index(emotion_name)
            image_files = list(emotion_dir.glob("*.[jJ][pP][gG]")) + list(
                emotion_dir.glob("*.[pP][nN][gG]")
            )

            logger.info(
                f"Processing {emotion_name}: {len(image_files)} images"
            )

            for image_file in tqdm(
                image_files, desc=f"Loading {emotion_name}"
            ):
                try:
                    # Load image
                    image = self.image_preprocessor.load_image(str(image_file))

                    # Detect faces
                    faces = self.face_detector.detect_faces(image)

                    # Process each detected face
                    for face_bbox in faces:
                        try:
                            # Preprocess face
                            face_image = (
                                self.image_preprocessor.preprocess_face(
                                    image,
                                    face_bbox,
                                    target_size=target_size,
                                    grayscale=grayscale,
                                )
                            )

                            images.append(face_image)
                            labels.append(emotion_label)

                            # Apply augmentation if enabled
                            if augment:
                                augmented_faces = (
                                    self.data_augmentor.augment_image(
                                        face_image, AUGMENTATION_CONFIG
                                    )
                                )
                                for aug_face in augmented_faces[
                                    1:
                                ]:  # Skip original
                                    images.append(aug_face)
                                    labels.append(emotion_label)

                        except InvalidImageError as e:
                            error_count += 1
                            logger.debug(f"Error processing face: {str(e)}")

                except (InvalidImageError, NoFaceDetectedError) as e:
                    error_count += 1
                    logger.debug(
                        f"Error processing image {image_file}: {str(e)}"
                    )

        if not images:
            raise DatasetError(f"No valid images found in {data_dir}")

        logger.info(
            f"Loaded {len(images)} images with {error_count} errors. "
            f"Errors: {error_count / (len(images) + error_count) * 100:.2f}%"
        )

        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)

        # Add channel dimension if grayscale
        if grayscale and len(X.shape) == 3:
            X = np.expand_dims(X, axis=-1)

        logger.info(f"Dataset shape: {X.shape}, Labels shape: {y.shape}")

        return X, y

    @staticmethod
    def split_dataset(
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = TRAIN_SPLIT,
        val_ratio: float = VALIDATION_SPLIT,
        test_ratio: float = TEST_SPLIT,
        random_state: int = 42,
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Split dataset into train, validation, and test sets.

        Args:
            X: Input features
            y: Labels
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)

        Raises:
            DatasetError: If ratios don't sum to 1.0
        """
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise DatasetError(f"Ratios must sum to 1.0, got {total_ratio}")

        # Split into train and temp (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            stratify=y,
        )

        # Split temp into val and test
        val_test_ratio = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=1 - val_test_ratio,
            random_state=random_state,
            stratify=y_temp,
        )

        logger.info(
            f"Dataset split: Train={X_train.shape[0]}, "
            f"Val={X_val.shape[0]}, Test={X_test.shape[0]}"
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def save_dataset(
        X: np.ndarray,
        y: np.ndarray,
        output_dir: str,
        name: str = "dataset",
    ) -> None:
        """
        Save dataset to disk.

        Args:
            X: Input features
            y: Labels
            output_dir: Output directory
            name: Dataset name prefix

        Raises:
            DatasetError: If save fails
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            np.save(output_path / f"{name}_X.npy", X)
            np.save(output_path / f"{name}_y.npy", y)

            logger.info(f"Dataset saved to {output_path}")
        except Exception as e:
            raise DatasetError(f"Failed to save dataset: {str(e)}")

    @staticmethod
    def load_dataset(
        input_dir: str,
        name: str = "dataset",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset from disk.

        Args:
            input_dir: Input directory
            name: Dataset name prefix

        Returns:
            Tuple of (X, y)

        Raises:
            DatasetError: If load fails
        """
        try:
            input_path = Path(input_dir)

            X = np.load(input_path / f"{name}_X.npy")
            y = np.load(input_path / f"{name}_y.npy")

            logger.info(f"Dataset loaded from {input_path}")
            return X, y
        except Exception as e:
            raise DatasetError(f"Failed to load dataset: {str(e)}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def create_sample_dataset(
    output_dir: str, num_images_per_emotion: int = 10
) -> None:
    """
    Create a sample dataset for testing.

    Args:
        output_dir: Output directory
        num_images_per_emotion: Number of sample images per emotion
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Creating sample dataset with {num_images_per_emotion} images per emotion"
    )

    for emotion in EMOTIONS:
        emotion_dir = output_path / emotion
        emotion_dir.mkdir(exist_ok=True)

        for i in range(num_images_per_emotion):
            # Create random image
            image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

            # Add a simple face-like pattern
            cv2.circle(image, (50, 40), 15, (255, 255, 255), -1)  # Face
            cv2.circle(image, (40, 35), 3, (0, 0, 0), -1)  # Left eye
            cv2.circle(image, (60, 35), 3, (0, 0, 0), -1)  # Right eye

            # Save image
            image_path = emotion_dir / f"{emotion}_{i:03d}.jpg"
            cv2.imwrite(str(image_path), image)

    logger.info(f"Sample dataset created at {output_dir}")
