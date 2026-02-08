"""
Vision Transformer (ViT) implementation for gene expression prediction.

This module provides a standard Vision Transformer architecture adapted for 
spatial transcriptomics analysis, following the patterns established in the 
CiT-to-gene implementation.
"""

from typing import Tuple, Optional
import math

import torch
import torch.nn as nn
from torch import Tensor
from timm.layers.weight_init import trunc_normal_  # type: ignore

class PatchEmbedding(nn.Module):
    """
    Image patch embedding layer for Vision Transformer.
    
    Converts an input image into a sequence of patch embeddings by:
    1. Dividing the image into non-overlapping patches
    2. Flattening each patch
    3. Projecting to embedding dimension
    """
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        """
        Initialize patch embedding layer.
        
        Args:
            img_size: Input image size (assumed square)
            patch_size: Size of each patch (assumed square)
            in_chans: Number of input channels
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_chans * patch_size ** 2
        
        # Linear projection of flattened patches
        self.projection = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of patch embedding.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Patch embeddings of shape [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}x{W}) doesn't match expected size ({self.img_size}x{self.img_size})"
        
        # Apply convolution and reshape
        x = self.projection(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        return x

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self attention module for Vision Transformer.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape [B, N, embed_dim]
            
        Returns:
            Output tensor of shape [B, N, embed_dim]
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Final projection
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x

class MLP(nn.Module):
    """
    MLP block used in Vision Transformer.
    """
    
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        """
        Initialize MLP block.
        
        Args:
            embed_dim: Input/output embedding dimension
            mlp_ratio: Ratio of hidden dimension to embed_dim
            dropout: Dropout probability
        """
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of MLP block.
        
        Args:
            x: Input tensor of shape [B, N, embed_dim]
            
        Returns:
            Output tensor of shape [B, N, embed_dim]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class TransformerBlock(nn.Module):
    """
    Single transformer block consisting of multi-head attention and MLP.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, 
                 dropout: float = 0.0, drop_path: float = 0.0):
        """
        Initialize transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout probability
            drop_path: Drop path probability
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
        # Drop path for stochastic depth
        self.drop_path = nn.Identity() if drop_path <= 0.0 else nn.Dropout(drop_path)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of transformer block.
        
        Args:
            x: Input tensor of shape [B, N, embed_dim]
            
        Returns:
            Output tensor of shape [B, N, embed_dim]
        """
        # Multi-head attention with residual connection
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # MLP with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model for image feature extraction.
    
    This implementation follows the standard ViT architecture from 
    "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
    """
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.0, drop_path_rate: float = 0.0,
                 use_cls_token: bool = True):
        """
        Initialize Vision Transformer.
        
        Args:
            img_size: Input image size
            patch_size: Patch size
            in_chans: Number of input channels
            embed_dim: Embedding dimension
            depth: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP ratio
            dropout: Dropout probability
            drop_path_rate: Drop path rate
            use_cls_token: Whether to use classification token
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.use_cls_token = use_cls_token
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        
        # Class token (if used)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            
        # Position embedding
        num_tokens = self.num_patches + (1 if use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[i])
            for i in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m: nn.Module):
        """Initialize weights following ViT conventions."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of Vision Transformer.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Feature tensor of shape [B, N, embed_dim] where N is sequence length
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Add class token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Final normalization
        x = self.norm(x)
        
        return x

def create_vit_base(img_size: int = 224, num_classes: Optional[int] = None) -> VisionTransformer:
    """
    Create a ViT-Base model.
    
    Args:
        img_size: Input image size
        num_classes: Number of output classes (if None, returns feature extractor)
        
    Returns:
        ViT-Base model
    """
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        drop_path_rate=0.1
    )

def create_vit_small(img_size: int = 224, num_classes: Optional[int] = None) -> VisionTransformer:
    """
    Create a ViT-Small model.
    
    Args:
        img_size: Input image size  
        num_classes: Number of output classes (if None, returns feature extractor)
        
    Returns:
        ViT-Small model
    """
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        drop_path_rate=0.1
    )

def create_vit_tiny(img_size: int = 224, num_classes: Optional[int] = None) -> VisionTransformer:
    """
    Create a ViT-Tiny model.
    
    Args:
        img_size: Input image size
        num_classes: Number of output classes (if None, returns feature extractor)
        
    Returns:
        ViT-Tiny model
    """
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.1,
        drop_path_rate=0.1
    )