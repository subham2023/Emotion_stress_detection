"""
Training Module for Emotion Recognition System.

This module handles model training with callbacks, data generators,
class weight balancing, and training history export.
"""

import json
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
    Callback,
)

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    LEARNING_RATE_DECAY,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    REDUCE_LR_MIN,
    CHECKPOINT_MONITOR,
    CHECKPOINT_MODE,
    CHECKPOINT_SAVE_BEST_ONLY,
    AUGMENTATION_CONFIG,
    EMOTIONS,
    TRAINING_HISTORY_PATH,
    CHECKPOINT_PATH,
    LOGS_DIR,
    LOG_FILE,
    LOG_LEVEL,
    LOG_FORMAT,
)
from model import ModelManager, create_model

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


class TrainingError(Exception):
    """Base exception for training errors."""

    pass


# ============================================================================
# CUSTOM CALLBACKS
# ============================================================================


class MetricsLogger(Callback):
    """
    Custom callback to log training metrics.
    """

    def __init__(self, log_file: str = None):
        """
        Initialize metrics logger.

        Args:
            log_file: Path to log file
        """
        super().__init__()
        self.log_file = log_file or str(LOGS_DIR / "metrics.log")

    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None):
        """Log metrics at end of epoch."""
        logs = logs or {}
        message = (
            f"Epoch {epoch + 1}: "
            f"loss={logs.get('loss', 0):.4f}, "
            f"accuracy={logs.get('accuracy', 0):.4f}, "
            f"val_loss={logs.get('val_loss', 0):.4f}, "
            f"val_accuracy={logs.get('val_accuracy', 0):.4f}"
        )
        logger.info(message)


class PerformanceMonitor(Callback):
    """
    Monitor training performance and detect anomalies.
    """

    def __init__(self, patience: int = 5):
        """
        Initialize performance monitor.

        Args:
            patience: Number of epochs to wait for improvement
        """
        super().__init__()
        self.patience = patience
        self.wait = 0
        self.best_loss = float("inf")

    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None):
        """Check performance at end of epoch."""
        logs = logs or {}
        current_loss = logs.get("loss", 0)

        # Check for NaN or Inf
        if np.isnan(current_loss) or np.isinf(current_loss):
            logger.error(f"Epoch {epoch + 1}: Invalid loss value {current_loss}")
            self.model.stop_training = True
            return

        # Check for improvement
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logger.warning(
                    f"No improvement for {self.patience} epochs, stopping training"
                )
                self.model.stop_training = True


# ============================================================================
# DATA GENERATORS
# ============================================================================


class DataGeneratorManager:
    """
    Manage data generators for training and validation.
    """

    @staticmethod
    def create_train_generator(
        X_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int = BATCH_SIZE,
        augmentation_config: Dict[str, Any] = None,
    ) -> tf.keras.preprocessing.image.ImageDataGenerator:
        """
        Create training data generator with augmentation.

        Args:
            X_train: Training images
            y_train: Training labels
            batch_size: Batch size
            augmentation_config: Augmentation configuration

        Returns:
            Data generator
        """
        if augmentation_config is None:
            augmentation_config = AUGMENTATION_CONFIG

        logger.info("Creating training data generator with augmentation")

        train_datagen = ImageDataGenerator(
            rotation_range=augmentation_config.get("rotation_range", 15),
            width_shift_range=augmentation_config.get("width_shift_range", 0.1),
            height_shift_range=augmentation_config.get("height_shift_range", 0.1),
            zoom_range=augmentation_config.get("zoom_range", 0.2),
            horizontal_flip=augmentation_config.get("horizontal_flip", True),
            fill_mode=augmentation_config.get("fill_mode", "nearest"),
            rescale=1.0 / 255.0,
        )

        return train_datagen

    @staticmethod
    def create_val_generator() -> tf.keras.preprocessing.image.ImageDataGenerator:
        """
        Create validation data generator (no augmentation).

        Returns:
            Data generator
        """
        logger.info("Creating validation data generator (no augmentation)")

        val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

        return val_datagen

    @staticmethod
    def create_generators(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = BATCH_SIZE,
    ) -> Tuple[
        tf.keras.preprocessing.image.ImageDataGenerator,
        tf.keras.preprocessing.image.ImageDataGenerator,
    ]:
        """
        Create both training and validation generators.

        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            batch_size: Batch size

        Returns:
            Tuple of (train_generator, val_generator)
        """
        train_datagen = DataGeneratorManager.create_train_generator(
            X_train, y_train, batch_size=batch_size
        )
        val_datagen = DataGeneratorManager.create_val_generator()

        return train_datagen, val_datagen


# ============================================================================
# TRAINING MANAGER
# ============================================================================


class TrainingManager:
    """
    Manage model training process.
    """

    @staticmethod
    def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced dataset.

        Args:
            y: Labels array

        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        class_weights = {cls: weight for cls, weight in zip(classes, weights)}

        logger.info(f"Class weights: {class_weights}")
        return class_weights

    @staticmethod
    def create_callbacks(
        model_name: str = "emotion_model",
        enable_tensorboard: bool = True,
    ) -> List[Callback]:
        """
        Create training callbacks.

        Args:
            model_name: Name of the model
            enable_tensorboard: Enable TensorBoard logging

        Returns:
            List of callbacks
        """
        callbacks = []

        # Model checkpoint
        checkpoint_path = CHECKPOINT_PATH / f"{model_name}_best.h5"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=CHECKPOINT_MONITOR,
            mode=CHECKPOINT_MODE,
            save_best_only=CHECKPOINT_SAVE_BEST_ONLY,
            verbose=1,
        )
        callbacks.append(checkpoint_callback)
        logger.info(f"Checkpoint callback created: {checkpoint_path}")

        # Early stopping
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        )
        callbacks.append(early_stopping_callback)
        logger.info(
            f"Early stopping callback created (patience={EARLY_STOPPING_PATIENCE})"
        )

        # Reduce learning rate on plateau
        reduce_lr_callback = ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=REDUCE_LR_MIN,
            verbose=1,
        )
        callbacks.append(reduce_lr_callback)
        logger.info(
            f"ReduceLROnPlateau callback created (patience={REDUCE_LR_PATIENCE})"
        )

        # TensorBoard
        if enable_tensorboard:
            tensorboard_path = LOGS_DIR / "tensorboard" / model_name
            tensorboard_callback = TensorBoard(
                log_dir=str(tensorboard_path),
                histogram_freq=1,
                write_graph=True,
            )
            callbacks.append(tensorboard_callback)
            logger.info(f"TensorBoard callback created: {tensorboard_path}")

        # Metrics logger
        metrics_callback = MetricsLogger()
        callbacks.append(metrics_callback)

        # Performance monitor
        performance_callback = PerformanceMonitor(patience=5)
        callbacks.append(performance_callback)

        return callbacks

    @staticmethod
    def train(
        model: keras.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        class_weights: Optional[Dict[int, float]] = None,
        callbacks: Optional[List[Callback]] = None,
        use_data_augmentation: bool = True,
    ) -> keras.callbacks.History:
        """
        Train model.

        Args:
            model: Keras model to train
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            class_weights: Class weights for imbalanced data
            callbacks: List of callbacks
            use_data_augmentation: Use data augmentation

        Returns:
            Training history

        Raises:
            TrainingError: If training fails
        """
        try:
            logger.info("=" * 80)
            logger.info("STARTING MODEL TRAINING")
            logger.info("=" * 80)
            logger.info(f"Training samples: {X_train.shape[0]}")
            logger.info(f"Validation samples: {X_val.shape[0]}")
            logger.info(f"Batch size: {batch_size}")
            logger.info(f"Epochs: {epochs}")
            logger.info(f"Data augmentation: {use_data_augmentation}")

            if callbacks is None:
                callbacks = TrainingManager.create_callbacks()

            if class_weights is None:
                class_weights = TrainingManager.calculate_class_weights(y_train)

            if use_data_augmentation:
                # Use data generators
                train_datagen, val_datagen = DataGeneratorManager.create_generators(
                    X_train, y_train, X_val, y_val, batch_size=batch_size
                )

                train_generator = train_datagen.flow(
                    X_train, y_train, batch_size=batch_size, shuffle=True
                )
                val_generator = val_datagen.flow(
                    X_val, y_val, batch_size=batch_size, shuffle=False
                )

                history = model.fit(
                    train_generator,
                    steps_per_epoch=len(X_train) // batch_size,
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=len(X_val) // batch_size,
                    class_weight=class_weights,
                    callbacks=callbacks,
                    verbose=1,
                )
            else:
                # Direct training without augmentation
                history = model.fit(
                    X_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_val, y_val),
                    class_weight=class_weights,
                    callbacks=callbacks,
                    verbose=1,
                )

            logger.info("=" * 80)
            logger.info("TRAINING COMPLETED")
            logger.info("=" * 80)

            return history

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise TrainingError(f"Training failed: {str(e)}")

    @staticmethod
    def save_training_history(
        history: keras.callbacks.History,
        output_path: str = None,
    ) -> None:
        """
        Save training history to JSON file.

        Args:
            history: Training history object
            output_path: Path to save history

        Raises:
            TrainingError: If save fails
        """
        try:
            if output_path is None:
                output_path = TRAINING_HISTORY_PATH

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            history_dict = {
                "loss": [float(x) for x in history.history.get("loss", [])],
                "accuracy": [float(x) for x in history.history.get("accuracy", [])],
                "val_loss": [float(x) for x in history.history.get("val_loss", [])],
                "val_accuracy": [
                    float(x) for x in history.history.get("val_accuracy", [])
                ],
            }

            with open(output_path, "w") as f:
                json.dump(history_dict, f, indent=4)

            logger.info(f"Training history saved to {output_path}")

        except Exception as e:
            raise TrainingError(f"Failed to save training history: {str(e)}")

    @staticmethod
    def load_training_history(input_path: str = None) -> Dict[str, List[float]]:
        """
        Load training history from JSON file.

        Args:
            input_path: Path to history file

        Returns:
            Training history dictionary

        Raises:
            TrainingError: If load fails
        """
        try:
            if input_path is None:
                input_path = TRAINING_HISTORY_PATH

            input_path = Path(input_path)

            if not input_path.exists():
                raise TrainingError(f"History file not found: {input_path}")

            with open(input_path, "r") as f:
                history = json.load(f)

            logger.info(f"Training history loaded from {input_path}")
            return history

        except Exception as e:
            raise TrainingError(f"Failed to load training history: {str(e)}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def resume_training(
    model_path: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 10,
    batch_size: int = BATCH_SIZE,
) -> keras.callbacks.History:
    """
    Resume training from a saved model checkpoint.

    Args:
        model_path: Path to saved model
        X_train: Training images
        y_train: Training labels
        X_val: Validation images
        y_val: Validation labels
        epochs: Number of additional epochs
        batch_size: Batch size

    Returns:
        Training history

    Raises:
        TrainingError: If resume fails
    """
    try:
        logger.info(f"Resuming training from checkpoint: {model_path}")

        # Load model
        model = ModelManager.load_model(model_path)

        # Create callbacks
        callbacks = TrainingManager.create_callbacks(model_name="resumed_model")

        # Train
        history = TrainingManager.train(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
        )

        return history

    except Exception as e:
        raise TrainingError(f"Failed to resume training: {str(e)}")
