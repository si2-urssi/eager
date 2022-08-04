# -*- coding: utf-8 -*-

"""Stored datasets."""

from pathlib import Path

from .soft_search_2022 import load_joined_soft_search_2022, load_soft_search_2022

_DATA_DIR = Path(__file__).parent

__all__ = [
    "load_soft_search_2022",
    "load_joined_soft_search_2022",
    "_DATA_DIR",
]
