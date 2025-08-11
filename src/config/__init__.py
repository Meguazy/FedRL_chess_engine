"""
Configuration Management

This module handles model and training configurations for the chess engine.
"""

from .model_config import ModelConfig, get_model_config
#from .training_config import get_training_config

__all__ = [
    'ModelConfig',
    'get_model_config',
    #'get_training_config'
]