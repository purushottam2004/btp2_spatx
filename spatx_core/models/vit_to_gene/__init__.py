"""
ViT-to-gene prediction module initialization.

This module provides Vision Transformer-based models for predicting gene expression
from histology images in spatial transcriptomics analysis.
"""

from .vit import (
    VisionTransformer,
    create_vit_tiny,
    create_vit_small,
    create_vit_base
)

from .ViTGene import (
    ViTGenePredictor,
    create_vit_gene_model
)

from ._loss import (
    MSELoss,
    MAELoss,
    SmoothL1Loss,
    CombinedLoss,
    CorrelationLoss,
    create_loss_function
)

from ._utils import (
    init_weights,
    calculate_metrics,
    get_model_size,
    freeze_layers,
    unfreeze_layers,
    apply_gradient_clipping,
    warmup_cosine_schedule,
    EarlyStopping,
    log_model_info,
    save_checkpoint,
    load_checkpoint
)

__all__ = [
    # Core ViT models
    'VisionTransformer',
    'create_vit_tiny',
    'create_vit_small',
    'create_vit_base',
    
    # Gene prediction models
    'ViTGenePredictor',
    'create_vit_gene_model',
    
    # Loss functions
    'MSELoss',
    'MAELoss',
    'SmoothL1Loss',
    'CombinedLoss',
    'CorrelationLoss',
    'create_loss_function',
    
    # Utilities
    'init_weights',
    'calculate_metrics',
    'get_model_size',
    'freeze_layers',
    'unfreeze_layers',
    'apply_gradient_clipping',
    'warmup_cosine_schedule',
    'EarlyStopping',
    'log_model_info',
    'save_checkpoint',
    'load_checkpoint'
]