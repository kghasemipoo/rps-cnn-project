# src/models.py

import tensorflow as tf
from tensorflow.keras import layers, models, applications


def build_tinynet(input_shape=(150, 150, 3), n_classes=3):
    """
    TinyNet (~6K parameters):
    A lightweight CNN designed as a quick pipeline sanity check.
    
    Architecture:
        Conv2D(16) → ReLU → MaxPool →
        Conv2D(32) → ReLU → MaxPool →
        GlobalAvgPool →
        Dense(32) → ReLU → Dropout(0.5) →
        Output (Dense with softmax)

    Returns:
        Keras Model object (TinyNet)
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name="TinyNet")


def build_mediumnet(input_shape=(150, 150, 3), n_classes=3):
    """
    MediumNet (~300K parameters):
    A deeper CNN with batch normalization and regularization for balance between accuracy and speed.

    Architecture:
        [Conv2D(32) → Conv2D(32) → MaxPool] →
        [Conv2D(64) → Conv2D(64) → MaxPool] →
        [Conv2D(128) → Conv2D(128) → MaxPool] →
        GlobalAvgPool →
        Dense(128) → ReLU → Dropout(0.5) →
        Output (Dense with softmax)

    Returns:
        Keras Model object (MediumNet)
    """
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)

    # Block 2
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)

    # Block 3
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name="MediumNet")


def build_transfernet(input_shape=(150, 150, 3), n_classes=3):
    """
    TransferNet (~2.4M parameters):
    Uses MobileNetV2 as a frozen feature extractor, followed by a lightweight classification head.

    Architecture:
        MobileNetV2 (frozen) →
        GlobalAvgPool →
        Dense(128) → ReLU → Dropout(0.3) →
        Output (Dense with softmax)

    Returns:
        Keras Model object (TransferNet)
    """
    base = applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name="TransferNet")
