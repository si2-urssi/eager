"""Stored datasets."""

from pathlib import Path

from .soft_search_2022 import (
    load_soft_search_2022_training,
    load_soft_search_2022_training_irr,
)

_DATA_DIR = Path(__file__).parent

__all__ = [
    "load_soft_search_2022_training",
    "load_soft_search_2022_training_irr",
    "_DATA_DIR",
]
