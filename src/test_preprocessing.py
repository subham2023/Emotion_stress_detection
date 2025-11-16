"""
Test script for data preprocessing module.

This script tests all components of the data preprocessing pipeline
including face detection, image preprocessing, and data augmentation.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    IMAGE_SIZE_SMALL,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    AUGMENTATION_CONFIG,
)
from data_preprocessing import (
    FaceDetector,
    ImagePreprocessor,
    DataAugmentor,
    DatasetManager,
    create_sample_dataset,
    NoFaceDetectedError,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_face_detector():
    """Test face detection functionality."""
    logger.info("=" * 80)
    logger.info("Testing Face Detector")
    logger.info("=" * 80)

    try:
        # Test Haar Cascade
        logger.info("Testing Haar Cascade detector...")
        detector_haar = FaceDetector(method="haar")
        logger.info("✓ Haar Cascade detector initialized")

        # Create a simple test image with face-like pattern
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(test_image, (100, 100), 40, (255, 255, 255), -1)  # Face circle
        cv2.circle(test_image, (85, 85), 5, (0, 0, 0), -1)  # Left eye
        cv2.circle(test_image, (115, 85), 5, (0, 0, 0), -1)  # Right eye

        # Test detection
        try:
            faces = detector_haar.detect_faces(test_image)
            logger.info(f"✓ Detected {len(faces)} face(s)")
        except NoFaceDetectedError:
            logger.warning(
                "No faces detected in test image (expected for simple pattern)"
            )

        logger.info("✓ Face detector test passed\n")

    except Exception as e:
        logger.error(f"✗ Face detector test failed: {str(e)}\n")
        return False

    return True


def test_image_preprocessor():
    """Test image preprocessing functionality."""
    logger.info("=" * 80)
    logger.info("Testing Image Preprocessor")
    logger.info("=" * 80)

    try:
        preprocessor = ImagePreprocessor()

        # Create test image
        test_image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

        # Test normalization
        normalized = preprocessor.normalize_image(test_image)
        assert normalized.max() <= 1.0, "Normalized image should be <= 1.0"
        assert normalized.min() >= 0.0, "Normalized image should be >= 0.0"
        logger.info("✓ Image normalization works")

        # Test face preprocessing
        face_bbox = (50, 50, 100, 100)
        face = preprocessor.preprocess_face(
            test_image, face_bbox, target_size=IMAGE_SIZE_SMALL
        )
        assert face.shape == (48, 48), f"Expected shape (48, 48), got {face.shape}"
        logger.info(f"✓ Face preprocessing works (output shape: {face.shape})")

        # Test histogram equalization
        gray_image = (
            cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        )
        equalized = preprocessor.equalize_histogram(gray_image)
        assert equalized.shape == gray_image.shape, "Equalized image shape mismatch"
        logger.info("✓ Histogram equalization works")

        logger.info("✓ Image preprocessor test passed\n")

    except Exception as e:
        logger.error(f"✗ Image preprocessor test failed: {str(e)}\n")
        return False

    return True


def test_data_augmentor():
    """Test data augmentation functionality."""
    logger.info("=" * 80)
    logger.info("Testing Data Augmentor")
    logger.info("=" * 80)

    try:
        augmentor = DataAugmentor()

        # Create test image
        test_image = np.random.rand(48, 48).astype(np.float32)

        # Test rotation
        rotated = augmentor.rotate(test_image, 15)
        assert rotated.shape == test_image.shape, "Rotated image shape mismatch"
        logger.info("✓ Image rotation works")

        # Test flip
        flipped = augmentor.flip(test_image, horizontal=True)
        assert flipped.shape == test_image.shape, "Flipped image shape mismatch"
        logger.info("✓ Image flip works")

        # Test brightness adjustment
        brightened = augmentor.adjust_brightness(test_image, 1.2)
        assert brightened.max() <= 1.0, "Brightened image should be <= 1.0"
        logger.info("✓ Brightness adjustment works")

        # Test noise addition
        noisy = augmentor.add_noise(test_image, noise_level=0.01)
        assert noisy.shape == test_image.shape, "Noisy image shape mismatch"
        logger.info("✓ Noise addition works")

        # Test zoom
        zoomed = augmentor.zoom(test_image, 1.2)
        assert zoomed.shape == test_image.shape, "Zoomed image shape mismatch"
        logger.info("✓ Image zoom works")

        # Test augmentation pipeline
        augmented = augmentor.augment_image(test_image, AUGMENTATION_CONFIG)
        assert len(augmented) > 1, "Should produce multiple augmented images"
        logger.info(f"✓ Augmentation pipeline works (produced {len(augmented)} images)")

        logger.info("✓ Data augmentor test passed\n")

    except Exception as e:
        logger.error(f"✗ Data augmentor test failed: {str(e)}\n")
        return False

    return True


def test_dataset_manager():
    """Test dataset management functionality."""
    logger.info("=" * 80)
    logger.info("Testing Dataset Manager")
    logger.info("=" * 80)

    try:
        manager = DatasetManager(face_detection_method="haar")

        # Create sample dataset
        sample_dir = RAW_DATA_DIR / "test_sample"
        create_sample_dataset(str(sample_dir), num_images_per_emotion=5)
        logger.info(f"✓ Sample dataset created at {sample_dir}")

        # Test dataset splitting
        X = np.random.rand(100, 48, 48, 1).astype(np.float32)
        y = np.random.randint(0, 7, 100)

        X_train, X_val, X_test, y_train, y_val, y_test = DatasetManager.split_dataset(
            X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
        )

        assert (
            X_train.shape[0] == 80
        ), f"Expected 80 train samples, got {X_train.shape[0]}"
        assert X_val.shape[0] == 10, f"Expected 10 val samples, got {X_val.shape[0]}"
        assert X_test.shape[0] == 10, f"Expected 10 test samples, got {X_test.shape[0]}"
        logger.info(f"✓ Dataset splitting works")

        # Test dataset saving
        DatasetManager.save_dataset(X, y, str(PROCESSED_DATA_DIR), name="test")
        logger.info(f"✓ Dataset saved to {PROCESSED_DATA_DIR}")

        # Test dataset loading
        X_loaded, y_loaded = DatasetManager.load_dataset(
            str(PROCESSED_DATA_DIR), name="test"
        )
        assert X_loaded.shape == X.shape, "Loaded dataset shape mismatch"
        assert y_loaded.shape == y.shape, "Loaded labels shape mismatch"
        logger.info(f"✓ Dataset loading works")

        logger.info("✓ Dataset manager test passed\n")

    except Exception as e:
        logger.error(f"✗ Dataset manager test failed: {str(e)}\n")
        return False

    return True


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 80)
    logger.info("EMOTION RECOGNITION - DATA PREPROCESSING TEST SUITE")
    logger.info("=" * 80 + "\n")

    tests = [
        ("Face Detector", test_face_detector),
        ("Image Preprocessor", test_image_preprocessor),
        ("Data Augmentor", test_data_augmentor),
        ("Dataset Manager", test_dataset_manager),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Unexpected error in {test_name}: {str(e)}")
            results[test_name] = False

    # Summary
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")
    logger.info("=" * 80 + "\n")

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
