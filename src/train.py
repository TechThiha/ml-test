# src/train.py
import tensorflow as tf
from src.model import create_model
import numpy as np
import os

def train():
    # Generate dummy data (100 samples, 784 features)
    x_train = np.random.rand(100, 784)
    y_train = np.random.randint(0, 10, size=100)

    # Create the model
    model = create_model()

    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=32)

    # Save the trained model
    model_save_path = os.path.join(os.getcwd(), "model.h5")
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train()
