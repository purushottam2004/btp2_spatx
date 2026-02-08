"""
Loss functions for gene expression prediction models.

This module provides specialized loss functions for gene expression prediction tasks,
including Spearman correlation loss and combined loss functions.
"""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SpearmanLoss(nn.Module):
    """
    Spearman rank correlation loss.
    
    This loss computes a differentiable approximation of the Spearman rank correlation
    and returns 1 minus the correlation as a loss (so perfect correlation yields zero loss).
    
    Attributes:
        reg (float): Regularization parameter that controls the softness of the ranking.
    """
    
    def __init__(self, reg: float = 1.0) -> None:
        """
        Initialize the Spearman rank correlation loss.
        
        Args:
            reg: Regularization parameter controlling the softness of the ranking.
                Higher values make the ranking more discrete.
        """
        super().__init__() #type: ignore
        self.reg = reg
        
    def forward(self, pred: Tensor, true: Tensor) -> Tensor:
        """
        Compute the Spearman rank correlation loss between predictions and targets.
        
        Args:
            pred: Predicted values [batch_size, num_genes]
            true: Target values [batch_size, num_genes]
            
        Returns:
            Loss value (1 - correlation)
        """
        # pred, true: [B, num_genes]
        pr = self._soft_rank(pred)
        tr = self._soft_rank(true)
        pr = F.normalize(pr, dim=1)
        tr = F.normalize(tr, dim=1)
        corr = (pr*tr).sum(dim=1)
        return cast(Tensor, 1 - corr.mean())

    def _soft_rank(self, x: Tensor) -> Tensor:
        """
        Compute a differentiable approximation of ranks.
        
        Args:
            x: Input tensor to be ranked [batch_size, num_genes]
            
        Returns:
            Soft ranks of input values
        """
        x2 = x.unsqueeze(-1)                     # [B,G,1]
        diff = x2 - x2.transpose(-1,-2)           # [B,G,G]
        P = torch.sigmoid(-self.reg*diff)         # [B,G,G]
        return P.sum(dim=-1)   
    

class CombinedLoss(nn.Module):
    """
    Combined loss function for gene expression prediction.
    
    This loss combines L1 loss (for absolute differences) and Spearman loss
    (for rank correlation), providing a balanced optimization target.
    
    Attributes:
        l1 (nn.L1Loss): L1 loss module
        sp (SpearmanLoss): Spearman loss module
        alpha (float): Weight for the Spearman loss component
    """
    
    def __init__(self, alpha: float = 0.5, reg: float = 1.0) -> None:
        """
        Initialize the combined loss function.
        
        Args:
            alpha: Weight for the Spearman loss component
            reg: Regularization parameter for the Spearman loss
        """
        super().__init__() #type: ignore
        self.l1 = nn.L1Loss()
        self.sp = SpearmanLoss(reg)
        self.alpha = alpha
        
    def forward(self, pred: Tensor, true: Tensor) -> Tensor:
        """
        Compute the combined loss.
        
        Args:
            pred: Predicted values [batch_size, num_genes]
            true: Target values [batch_size, num_genes]
            
        Returns:
            Combined loss value
        """
        return self.l1(pred, true) + self.alpha * self.sp(pred, true)