"""
ViT-to-gene predictors module initialization.

This module provides prediction utilities for Vision Transformer models
used in gene expression prediction tasks.
"""

from .simple_predictor import (
    SimpleViTPredictor,
    create_simple_predictor
)

__all__ = [
    'SimpleViTPredictor',
    'create_simple_predictor'
]