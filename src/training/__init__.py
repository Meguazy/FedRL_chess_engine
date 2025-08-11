"""
Training Components for Federated Chess Learning

This module provides parallel training capabilities and self-play generation
for style-diverse chess engine development.
"""

from .trainer import ParallelTrainer
from .self_play import StyleSpecificSelfPlay, SelfPlayGameResult

__all__ = [
    'ParallelTrainer',
    'StyleSpecificSelfPlay',
    'SelfPlayGameResult'
]