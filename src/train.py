# src/train.py
import tensorflow as tf
from src.model import create_model
import numpy as np
import os

def train():
    # Dummy data for training
    x_train = np.random.rand(100, 784)
    y_train = np.random.randint(0, 10, size=100)

    model = create_model()
    model.fit(x_train, y_train, epochs=5, batch_size=32)

    # Save the trained model
    model.save(os.path.join(os.getcwd(), "model.h5"))
    print("Model saved as model.h5")

if __name__ == "__main__":
    train()
