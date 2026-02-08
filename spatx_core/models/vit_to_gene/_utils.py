"""
Utility functions for ViT-to-gene prediction models.

This module provides helper functions for model initialization, metrics calculation,
and other utilities used across the ViT implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict, List, Optional
import numpy as np
from scipy.stats import pearsonr
import logging

def init_weights(module: nn.Module) -> None:
    """
    Initialize model weights using standard techniques.
    
    Args:
        module: PyTorch module to initialize
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)

def calculate_metrics(predictions: Tensor, targets: Tensor, mask: Tensor = None) -> Dict[str, float]:
    """
    Calculate evaluation metrics for gene expression prediction.
    
    Args:
        predictions: Predicted gene expression values [batch_size, num_genes]
        targets: Target gene expression values [batch_size, num_genes]
        mask: Optional mask for valid genes [batch_size, num_genes]
        
    Returns:
        Dictionary of metric values
    """
    # Convert to numpy for metric calculations
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    
    if mask is not None:
        mask_np = mask.detach().cpu().numpy()
        # Apply mask
        pred_np = pred_np * mask_np
        target_np = target_np * mask_np
        valid_mask = mask_np > 0
    else:
        valid_mask = np.ones_like(pred_np, dtype=bool)
    
    metrics = {}
    
    # MSE
    if valid_mask.any():
        mse = np.mean((pred_np[valid_mask] - target_np[valid_mask]) ** 2)
        metrics['mse'] = float(mse)
    else:
        metrics['mse'] = float('nan')
    
    # MAE
    if valid_mask.any():
        mae = np.mean(np.abs(pred_np[valid_mask] - target_np[valid_mask]))
        metrics['mae'] = float(mae)
    else:
        metrics['mae'] = float('nan')
    
    # Per-sample correlation
    correlations = []
    batch_size = pred_np.shape[0]
    
    for i in range(batch_size):
        pred_sample = pred_np[i]
        target_sample = target_np[i]
        
        if mask is not None:
            sample_mask = valid_mask[i]
            pred_sample = pred_sample[sample_mask]
            target_sample = target_sample[sample_mask]
        
        if len(pred_sample) > 1 and np.var(pred_sample) > 1e-8 and np.var(target_sample) > 1e-8:
            try:
                corr, _ = pearsonr(pred_sample, target_sample)
                if not np.isnan(corr):
                    correlations.append(corr)
            except:
                pass
    
    if correlations:
        metrics['correlation_mean'] = float(np.mean(correlations))
        metrics['correlation_std'] = float(np.std(correlations))
        metrics['correlation_median'] = float(np.median(correlations))
    else:
        metrics['correlation_mean'] = 0.0
        metrics['correlation_std'] = 0.0
        metrics['correlation_median'] = 0.0
    
    # R-squared
    if valid_mask.any():
        ss_res = np.sum((pred_np[valid_mask] - target_np[valid_mask]) ** 2)
        ss_tot = np.sum((target_np[valid_mask] - np.mean(target_np[valid_mask])) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        metrics['r2'] = float(r2)
    else:
        metrics['r2'] = float('nan')
    
    return metrics

def get_model_size(model: nn.Module) -> Dict[str, int]:
    """
    Get model size information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model size information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }

def freeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    """
    Freeze specified layers in a model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = False
                break

def unfreeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    """
    Unfreeze specified layers in a model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to unfreeze
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = True
                break

def apply_gradient_clipping(model: nn.Module, max_norm: float = 1.0) -> float:
    """
    Apply gradient clipping to model parameters.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
        
    Returns:
        Total gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def warmup_cosine_schedule(step: int, warmup_steps: int, total_steps: int, 
                          base_lr: float, min_lr: float = 0.0) -> float:
    """
    Cosine learning rate schedule with warmup.
    
    Args:
        step: Current training step
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        base_lr: Base learning rate
        min_lr: Minimum learning rate
        
    Returns:
        Learning rate for current step
    """
    if step < warmup_steps:
        # Linear warmup
        return base_lr * (step / warmup_steps)
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.
    """
    
    def __init__(self, patience: int = 7, min_delta: float = 0, restore_best: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False

def log_model_info(model: nn.Module, logger: logging.Logger = None) -> None:
    """
    Log information about a model.
    
    Args:
        model: PyTorch model
        logger: Logger to use (uses print if None)
    """
    size_info = get_model_size(model)
    
    log_func = logger.info if logger else print
    log_func(f"Model: {model.__class__.__name__}")
    log_func(f"Total parameters: {size_info['total_parameters']:,}")
    log_func(f"Trainable parameters: {size_info['trainable_parameters']:,}")
    log_func(f"Non-trainable parameters: {size_info['non_trainable_parameters']:,}")

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, filepath: str,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                   metrics: Dict[str, float] = None) -> None:
    """
    Save a training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
        scheduler: Optional learning rate scheduler
        metrics: Optional metrics dictionary
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath: str, model: nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                   device: torch.device = None) -> Dict:
    """
    Load a training checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to map tensors to
        
    Returns:
        Checkpoint information dictionary
    """
    if device is None:
        device = next(model.parameters()).device
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'metrics': checkpoint.get('metrics', {})
    }