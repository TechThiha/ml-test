# tests/test_model.py
import pytest
import tensorflow as tf
from src.model import create_model

def test_create_model():
    model = create_model()
    assert model is not None, "Model should be created"
    assert len(model.layers) == 3, "Model should have 3 layers"
    
def test_model_compiled():
    model = create_model()
    assert model.optimizer is not None, "Model should be compiled"
    assert model.loss is not None, "Model should have a loss function"
