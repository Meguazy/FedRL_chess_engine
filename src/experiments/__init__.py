"""
Experiment Management for Federated Chess Learning

This module provides comprehensive experiment configuration and management
for diversity-preserving federated learning research.

Configuration files are stored in the configs/ subdirectory:
- model_sizes.json: Neural network architecture configurations
- training_configs.json: Training hyperparameter configurations
"""

# Currently only configuration files exist in this module
# The experiment management classes are planned for future implementation

__all__ = []

# Configuration files are accessed via:
# - src/experiments/configs/model_sizes.json  
# - src/experiments/configs/training_configs.json
# These are loaded by the ParallelTrainer's config loading system