# src/model.py
import tensorflow as tf
from tensorflow.keras import layers, models

def create_model():
    """Creates a simple neural network model."""
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
