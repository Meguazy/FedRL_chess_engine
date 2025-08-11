"""
Federated Chess Learning Framework

A comprehensive framework for diversity-preserving federated reinforcement learning
applied to chess, with physical robotic validation capabilities.

Author: Francesco Finucci
"""

__version__ = "1.0.0"
__author__ = "Francesco Finucci"

# Core modules
from . import core
from . import training
from . import experiments
from . import data

__all__ = [
    'core',
    'training', 
    'experiments',
    'data'
]