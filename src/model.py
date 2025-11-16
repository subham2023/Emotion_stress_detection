"""
Model Architecture Module for Emotion Recognition System.

This module provides two model architectures:
1. Custom CNN with multiple convolutional blocks
2. Transfer Learning models (ResNet50, MobileNetV2, VGG16)
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import ResNet50, MobileNetV2, VGG16

from config import (
    NUM_CLASSES,
    IMAGE_SIZE_SMALL,
    IMAGE_SIZE_LARGE,
    IMAGE_CHANNELS,
    IMAGE_CHANNELS_RGB,
    LOG_FILE,
    LOG_LEVEL,
    LOG_FORMAT,
    USE_GPU,
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


class ModelError(Exception):
    """Base exception for model errors."""



class InvalidModelError(ModelError):
    """Raised when model is invalid or incompatible."""



class ModelNotFoundError(ModelError):
    """Raised when model file is not found."""



# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================


def setup_device():
    """
    Setup GPU/CPU device configuration.

    Returns:
        Device name (e.g., '/GPU:0' or '/CPU:0')
    """
    if USE_GPU:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            logger.info(f"Found {len(gpus)} GPU(s)")
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("GPU memory growth enabled")
                return "/GPU:0"
            except Exception as e:
                logger.warning(f"Failed to configure GPU: {str(e)}")
                return "/CPU:0"
        else:
            logger.warning("No GPU found, using CPU")
            return "/CPU:0"
    else:
        logger.info("Using CPU")
        return "/CPU:0"


# ============================================================================
# CUSTOM CNN MODEL
# ============================================================================


class CustomCNNModel:
    """
    Custom Convolutional Neural Network for emotion recognition.

    Architecture:
    - Input: 48x48x1 (grayscale)
    - 4 Conv2D blocks with BatchNorm and Dropout
    - MaxPooling after each block
    - 2 Dense layers with Dropout
    - Output: 7 classes (softmax)
    """

    @staticmethod
    def build(
        input_shape: Tuple[int, int, int] = (
            IMAGE_SIZE_SMALL[0],
            IMAGE_SIZE_SMALL[1],
            IMAGE_CHANNELS,
        ),
        num_classes: int = NUM_CLASSES,
        l2_reg: float = 0.001,
    ) -> keras.Model:
        """
        Build custom CNN model.

        Args:
            input_shape: Input shape (height, width, channels)
            num_classes: Number of output classes
            l2_reg: L2 regularization factor

        Returns:
            Compiled Keras model
        """
        logger.info("Building Custom CNN Model")
        logger.info(f"Input shape: {input_shape}, Classes: {num_classes}")

        model = models.Sequential(
            [
                # Input layer
                layers.Input(shape=input_shape),
                # First Conv Block
                layers.Conv2D(
                    32,
                    (3, 3),
                    padding="same",
                    kernel_regularizer=regularizers.l2(l2_reg),
                ),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Conv2D(
                    32,
                    (3, 3),
                    padding="same",
                    kernel_regularizer=regularizers.l2(l2_reg),
                ),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Second Conv Block
                layers.Conv2D(
                    64,
                    (3, 3),
                    padding="same",
                    kernel_regularizer=regularizers.l2(l2_reg),
                ),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Conv2D(
                    64,
                    (3, 3),
                    padding="same",
                    kernel_regularizer=regularizers.l2(l2_reg),
                ),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Third Conv Block
                layers.Conv2D(
                    128,
                    (3, 3),
                    padding="same",
                    kernel_regularizer=regularizers.l2(l2_reg),
                ),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Conv2D(
                    128,
                    (3, 3),
                    padding="same",
                    kernel_regularizer=regularizers.l2(l2_reg),
                ),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Fourth Conv Block
                layers.Conv2D(
                    256,
                    (3, 3),
                    padding="same",
                    kernel_regularizer=regularizers.l2(l2_reg),
                ),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Conv2D(
                    256,
                    (3, 3),
                    padding="same",
                    kernel_regularizer=regularizers.l2(l2_reg),
                ),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Global Average Pooling
                layers.GlobalAveragePooling2D(),
                # Dense layers
                layers.Dense(512, kernel_regularizer=regularizers.l2(l2_reg)),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dropout(0.5),
                layers.Dense(256, kernel_regularizer=regularizers.l2(l2_reg)),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dropout(0.5),
                # Output layer
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        logger.info("Custom CNN Model built successfully")
        return model


# ============================================================================
# TRANSFER LEARNING MODELS
# ============================================================================


class TransferLearningModel:
    """
    Transfer Learning models for emotion recognition.

    Supports: ResNet50, MobileNetV2, VGG16
    """

    @staticmethod
    def build_resnet50(
        input_shape: Tuple[int, int, int] = (
            IMAGE_SIZE_LARGE[0],
            IMAGE_SIZE_LARGE[1],
            IMAGE_CHANNELS_RGB,
        ),
        num_classes: int = NUM_CLASSES,
        freeze_base: bool = True,
    ) -> keras.Model:
        """
        Build ResNet50 transfer learning model.

        Args:
            input_shape: Input shape (height, width, channels)
            num_classes: Number of output classes
            freeze_base: Freeze base model weights if True

        Returns:
            Compiled Keras model
        """
        logger.info("Building ResNet50 Transfer Learning Model")

        # Load pre-trained ResNet50
        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
        )

        if freeze_base:
            base_model.trainable = False
            logger.info("Base model layers frozen")
        else:
            logger.info("Base model layers trainable")

        # Build custom head
        model = models.Sequential(
            [
                layers.Input(shape=input_shape),
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(512, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        logger.info("ResNet50 model built successfully")
        return model

    @staticmethod
    def build_mobilenetv2(
        input_shape: Tuple[int, int, int] = (
            IMAGE_SIZE_LARGE[0],
            IMAGE_SIZE_LARGE[1],
            IMAGE_CHANNELS_RGB,
        ),
        num_classes: int = NUM_CLASSES,
        freeze_base: bool = True,
    ) -> keras.Model:
        """
        Build MobileNetV2 transfer learning model.

        Args:
            input_shape: Input shape (height, width, channels)
            num_classes: Number of output classes
            freeze_base: Freeze base model weights if True

        Returns:
            Compiled Keras model
        """
        logger.info("Building MobileNetV2 Transfer Learning Model")

        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
        )

        if freeze_base:
            base_model.trainable = False
            logger.info("Base model layers frozen")
        else:
            logger.info("Base model layers trainable")

        # Build custom head
        model = models.Sequential(
            [
                layers.Input(shape=input_shape),
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(128, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        logger.info("MobileNetV2 model built successfully")
        return model

    @staticmethod
    def build_vgg16(
        input_shape: Tuple[int, int, int] = (
            IMAGE_SIZE_LARGE[0],
            IMAGE_SIZE_LARGE[1],
            IMAGE_CHANNELS_RGB,
        ),
        num_classes: int = NUM_CLASSES,
        freeze_base: bool = True,
    ) -> keras.Model:
        """
        Build VGG16 transfer learning model.

        Args:
            input_shape: Input shape (height, width, channels)
            num_classes: Number of output classes
            freeze_base: Freeze base model weights if True

        Returns:
            Compiled Keras model
        """
        logger.info("Building VGG16 Transfer Learning Model")

        # Load pre-trained VGG16
        base_model = VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
        )

        if freeze_base:
            base_model.trainable = False
            logger.info("Base model layers frozen")
        else:
            logger.info("Base model layers trainable")

        # Build custom head
        model = models.Sequential(
            [
                layers.Input(shape=input_shape),
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(512, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        logger.info("VGG16 model built successfully")
        return model


# ============================================================================
# MODEL MANAGER
# ============================================================================


class ModelManager:
    """
    Manage model creation, saving, loading, and compilation.
    """

    @staticmethod
    def compile_model(
        model: keras.Model,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        loss: str = "categorical_crossentropy",
    ) -> keras.Model:
        """
        Compile model with specified optimizer and loss.

        Args:
            model: Keras model to compile
            optimizer: Optimizer name ("adam", "sgd", "rmsprop")
            learning_rate: Learning rate for optimizer
            loss: Loss function

        Returns:
            Compiled model

        Raises:
            ModelError: If compilation fails
        """
        try:
            if optimizer.lower() == "adam":
                opt = keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer.lower() == "sgd":
                opt = keras.optimizers.SGD(learning_rate=learning_rate)
            elif optimizer.lower() == "rmsprop":
                opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")

            model.compile(
                optimizer=opt,
                loss=loss,
                metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
            )

            logger.info(
                f"Model compiled with optimizer={optimizer}, lr={learning_rate}"
            )
            return model

        except Exception as e:
            raise ModelError(f"Failed to compile model: {str(e)}")

    @staticmethod
    def save_model(model: keras.Model, model_path: str) -> None:
        """
        Save model to disk.

        Args:
            model: Keras model to save
            model_path: Path to save model

        Raises:
            ModelError: If save fails
        """
        try:
            model_path = Path(model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)

            model.save(str(model_path))
            logger.info(f"Model saved to {model_path}")

        except Exception as e:
            raise ModelError(f"Failed to save model: {str(e)}")

    @staticmethod
    def load_model(model_path: str) -> keras.Model:
        """
        Load model from disk.

        Args:
            model_path: Path to model file

        Returns:
            Loaded Keras model

        Raises:
            ModelNotFoundError: If model file not found
            InvalidModelError: If model is invalid
        """
        try:
            model_path = Path(model_path)

            if not model_path.exists():
                raise ModelNotFoundError(f"Model file not found: {model_path}")

            model = keras.models.load_model(str(model_path))
            logger.info(f"Model loaded from {model_path}")

            return model

        except ModelNotFoundError:
            raise
        except Exception as e:
            raise InvalidModelError(f"Failed to load model: {str(e)}")

    @staticmethod
    def get_model_summary(model: keras.Model) -> str:
        """
        Get model summary as string.

        Args:
            model: Keras model

        Returns:
            Model summary string
        """
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            model.summary()
        return f.getvalue()

    @staticmethod
    def count_parameters(model: keras.Model) -> Dict[str, int]:
        """
        Count model parameters.

        Args:
            model: Keras model

        Returns:
            Dictionary with parameter counts
        """
        total_params = model.count_params()
        trainable_params = sum(
            [tf.keras.backend.count_params(w) for w in model.trainable_weights]
        )
        non_trainable_params = total_params - trainable_params

        return {
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": non_trainable_params,
        }

    @staticmethod
    def unfreeze_base_layers(model: keras.Model, num_layers: int = 50) -> None:
        """
        Unfreeze base model layers for fine-tuning.

        Args:
            model: Keras model
            num_layers: Number of layers to unfreeze from the end

        Raises:
            ModelError: If unfreezing fails
        """
        try:
            # Find the base model (usually the first layer)
            base_model = None
            for layer in model.layers:
                if isinstance(layer, keras.Model):
                    base_model = layer
                    break

            if base_model is None:
                raise ModelError("No base model found in the model")

            # Unfreeze last num_layers
            len(base_model.layers)
            for layer in base_model.layers[-num_layers:]:
                layer.trainable = True

            logger.info(f"Unfroze last {num_layers} layers of base model")

        except Exception as e:
            raise ModelError(f"Failed to unfreeze layers: {str(e)}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def create_model(
    model_type: str = "custom_cnn",
    input_shape: Optional[Tuple[int, int, int]] = None,
    num_classes: int = NUM_CLASSES,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Create and compile a model.

    Args:
        model_type: Model type ("custom_cnn", "resnet50", "mobilenetv2", "vgg16")
        input_shape: Input shape (if None, uses default for model type)
        num_classes: Number of output classes
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled Keras model

    Raises:
        ModelError: If model creation fails
    """
    try:
        if model_type == "custom_cnn":
            if input_shape is None:
                input_shape = (IMAGE_SIZE_SMALL[0], IMAGE_SIZE_SMALL[1], IMAGE_CHANNELS)
            model = CustomCNNModel.build(
                input_shape=input_shape, num_classes=num_classes
            )

        elif model_type == "resnet50":
            if input_shape is None:
                input_shape = (
                    IMAGE_SIZE_LARGE[0],
                    IMAGE_SIZE_LARGE[1],
                    IMAGE_CHANNELS_RGB,
                )
            model = TransferLearningModel.build_resnet50(
                input_shape=input_shape, num_classes=num_classes
            )

        elif model_type == "mobilenetv2":
            if input_shape is None:
                input_shape = (
                    IMAGE_SIZE_LARGE[0],
                    IMAGE_SIZE_LARGE[1],
                    IMAGE_CHANNELS_RGB,
                )
            model = TransferLearningModel.build_mobilenetv2(
                input_shape=input_shape, num_classes=num_classes
            )

        elif model_type == "vgg16":
            if input_shape is None:
                input_shape = (
                    IMAGE_SIZE_LARGE[0],
                    IMAGE_SIZE_LARGE[1],
                    IMAGE_CHANNELS_RGB,
                )
            model = TransferLearningModel.build_vgg16(
                input_shape=input_shape, num_classes=num_classes
            )

        else:
            raise ModelError(f"Unknown model type: {model_type}")

        # Compile model
        model = ModelManager.compile_model(model, learning_rate=learning_rate)

        # Log model info
        params = ModelManager.count_parameters(model)
        logger.info(
            f"Model parameters - Total: {params['total']:,}, "
            f"Trainable: {params['trainable']:,}, "
            f"Non-trainable: {params['non_trainable']:,}"
        )

        return model

    except Exception as e:
        raise ModelError(f"Failed to create model: {str(e)}")
