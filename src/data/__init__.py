"""
Data Components for Chess Training

This module provides chess-specific data structures and utilities,
including opening databases and style classifications.
"""

from .openings.openings import (
    ECO_OpeningDatabase, 
    OpeningTemplate,
    create_eco_opening_database
)

__all__ = [
    'ECO_OpeningDatabase',
    'OpeningTemplate', 
    'create_eco_opening_database'
]