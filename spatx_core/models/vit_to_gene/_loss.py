"""
Loss functions for ViT-to-gene prediction models.

This module provides various loss functions suitable for gene expression prediction
tasks, including regression losses and specialized losses for spatial transcriptomics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MSELoss(nn.Module):
    """
    Mean Squared Error loss for gene expression regression.
    
    Standard MSE loss with optional masking for missing gene values.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize MSE loss.
        
        Args:
            reduction: Specifies the reduction to apply ('mean', 'sum', 'none')
        """
        super().__init__()
        self.reduction = reduction
        
    def forward(self, predictions: Tensor, targets: Tensor, mask: Tensor = None) -> Tensor:
        """
        Compute MSE loss.
        
        Args:
            predictions: Predicted gene expression values [batch_size, num_genes]
            targets: Target gene expression values [batch_size, num_genes]
            mask: Optional mask for valid genes [batch_size, num_genes]
            
        Returns:
            MSE loss value
        """
        loss = F.mse_loss(predictions, targets, reduction='none')
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / mask.sum()
            elif self.reduction == 'sum':
                return loss.sum()
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        
        return loss

class MAELoss(nn.Module):
    """
    Mean Absolute Error loss for gene expression regression.
    
    MAE loss which can be more robust to outliers than MSE.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize MAE loss.
        
        Args:
            reduction: Specifies the reduction to apply ('mean', 'sum', 'none')
        """
        super().__init__()
        self.reduction = reduction
        
    def forward(self, predictions: Tensor, targets: Tensor, mask: Tensor = None) -> Tensor:
        """
        Compute MAE loss.
        
        Args:
            predictions: Predicted gene expression values [batch_size, num_genes]
            targets: Target gene expression values [batch_size, num_genes]
            mask: Optional mask for valid genes [batch_size, num_genes]
            
        Returns:
            MAE loss value
        """
        loss = F.l1_loss(predictions, targets, reduction='none')
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / mask.sum()
            elif self.reduction == 'sum':
                return loss.sum()
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        
        return loss

class SmoothL1Loss(nn.Module):
    """
    Smooth L1 (Huber) loss for gene expression regression.
    
    Combines the advantages of both MSE and MAE losses.
    """
    
    def __init__(self, beta: float = 1.0, reduction: str = 'mean'):
        """
        Initialize Smooth L1 loss.
        
        Args:
            beta: The threshold at which to change between L1 and L2 loss
            reduction: Specifies the reduction to apply ('mean', 'sum', 'none')
        """
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        
    def forward(self, predictions: Tensor, targets: Tensor, mask: Tensor = None) -> Tensor:
        """
        Compute Smooth L1 loss.
        
        Args:
            predictions: Predicted gene expression values [batch_size, num_genes]
            targets: Target gene expression values [batch_size, num_genes]
            mask: Optional mask for valid genes [batch_size, num_genes]
            
        Returns:
            Smooth L1 loss value
        """
        loss = F.smooth_l1_loss(predictions, targets, beta=self.beta, reduction='none')
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / mask.sum()
            elif self.reduction == 'sum':
                return loss.sum()
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        
        return loss

class CombinedLoss(nn.Module):
    """
    Combined loss function that mixes different loss types.
    
    Combines MSE and MAE losses with configurable weights, similar to
    the CombinedLoss in the CiT implementation.
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 1.0, reg_lambda: float = 0.0):
        """
        Initialize combined loss.
        
        Args:
            alpha: Weight for MSE loss (1-alpha weight for MAE)
            beta: Weight for total loss
            reg_lambda: L2 regularization coefficient
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reg_lambda = reg_lambda
        
        self.mse_loss = MSELoss()
        self.mae_loss = MAELoss()
        
    def forward(self, predictions: Tensor, targets: Tensor, 
                model_params: Tensor = None, mask: Tensor = None) -> Tensor:
        """
        Compute combined loss.
        
        Args:
            predictions: Predicted gene expression values [batch_size, num_genes]
            targets: Target gene expression values [batch_size, num_genes]
            model_params: Model parameters for regularization
            mask: Optional mask for valid genes [batch_size, num_genes]
            
        Returns:
            Combined loss value
        """
        # Compute primary losses
        mse = self.mse_loss(predictions, targets, mask)
        mae = self.mae_loss(predictions, targets, mask)
        
        # Combine losses
        primary_loss = self.alpha * mse + (1 - self.alpha) * mae
        total_loss = self.beta * primary_loss
        
        # Add L2 regularization if specified
        if self.reg_lambda > 0 and model_params is not None:
            l2_reg = torch.norm(model_params, p=2)
            total_loss = total_loss + self.reg_lambda * l2_reg
            
        return total_loss

class CorrelationLoss(nn.Module):
    """
    Pearson correlation-based loss for gene expression prediction.
    
    Encourages high correlation between predicted and target gene expressions.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize correlation loss.
        
        Args:
            reduction: Specifies the reduction to apply ('mean', 'sum', 'none')
        """
        super().__init__()
        self.reduction = reduction
        
    def forward(self, predictions: Tensor, targets: Tensor, mask: Tensor = None) -> Tensor:
        """
        Compute correlation loss (1 - correlation).
        
        Args:
            predictions: Predicted gene expression values [batch_size, num_genes]
            targets: Target gene expression values [batch_size, num_genes]
            mask: Optional mask for valid genes [batch_size, num_genes]
            
        Returns:
            Correlation loss value
        """
        if mask is not None:
            # Apply mask
            pred_masked = predictions * mask
            target_masked = targets * mask
        else:
            pred_masked = predictions
            target_masked = targets
        
        # Compute correlation for each sample in batch
        batch_size = predictions.shape[0]
        correlations = []
        
        for i in range(batch_size):
            pred_sample = pred_masked[i]
            target_sample = target_masked[i]
            
            if mask is not None:
                # Only use non-masked values
                valid_mask = mask[i] > 0
                pred_sample = pred_sample[valid_mask]
                target_sample = target_sample[valid_mask]
            
            # Compute Pearson correlation
            if pred_sample.numel() > 1:
                corr = torch.corrcoef(torch.stack([pred_sample, target_sample]))[0, 1]
                # Handle NaN case
                if torch.isnan(corr):
                    corr = torch.tensor(0.0, device=predictions.device)
            else:
                corr = torch.tensor(0.0, device=predictions.device)
            
            correlations.append(1 - corr)  # Convert to loss (higher correlation = lower loss)
        
        correlations_tensor = torch.stack(correlations)
        
        if self.reduction == 'mean':
            return correlations_tensor.mean()
        elif self.reduction == 'sum':
            return correlations_tensor.sum()
        
        return correlations_tensor

def create_loss_function(loss_type: str = "combined", **kwargs) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss function to create
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function instance
    """
    if loss_type == "mse":
        return MSELoss(**kwargs)
    elif loss_type == "mae":
        return MAELoss(**kwargs)
    elif loss_type == "smooth_l1":
        return SmoothL1Loss(**kwargs)
    elif loss_type == "combined":
        return CombinedLoss(**kwargs)
    elif loss_type == "correlation":
        return CorrelationLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")