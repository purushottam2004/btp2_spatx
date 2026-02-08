"""
ViTGene model implementation for gene expression prediction from spatial transcriptomics images.

This module provides the architecture for predicting gene expression from image patches
using the Vision Transformer backbone combined with a specialized gene prediction head.
"""

import torch
import torch.nn as nn
from torch import Tensor

from spatx_core.models.vit_to_gene.vit import VisionTransformer

class GeneRegressionHead(nn.Module):
    """
    Simple regression head for gene expression prediction.
    
    Takes the CLS token output from ViT and predicts gene expression values
    through a series of linear layers with dropout.
    
    Attributes:
        head (nn.Sequential): The regression head layers
    """
    
    def __init__(self, feat_dim: int, num_genes: int, hidden_dim: int = 512, dropout: float = 0.1):
        """
        Initialize the gene regression head.
        
        Args:
            feat_dim: Dimension of input features from ViT
            num_genes: Number of genes to predict
            hidden_dim: Hidden dimension size
            dropout: Dropout probability
        """
        super().__init__()
        
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_genes)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the regression head.
        
        Args:
            x: Input features from ViT [batch_size, feat_dim]
            
        Returns:
            Gene expression predictions [batch_size, num_genes]
        """
        return self.head(x)

class GeneTransformerHead(nn.Module):
    """
    Transformer-based head for gene expression prediction.
    
    Each of `num_genes` learnable query vectors attends over the spatial tokens
    of the ViT output to predict gene expression levels. Similar to the CiTGene
    implementation but adapted for ViT features.
    
    Attributes:
        mem_proj (nn.Linear): Projection layer for spatial tokens
        query_embed (nn.Parameter): Learnable gene query embeddings
        decoder (nn.TransformerDecoder): Transformer decoder for cross-attention
        to_logits (nn.Linear): Final projection to scalar predictions
    """
    
    def __init__(
        self,
        feat_dim: int,
        num_genes: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
    ) -> None:
        """
        Initialize the Gene Transformer Head.
        
        Args:
            feat_dim: Dimension of the input features from ViT
            num_genes: Number of genes to predict
            d_model: Dimension of the transformer model
            nhead: Number of attention heads
            num_layers: Number of transformer decoder layers
        """
        super().__init__()
        
        # Project each spatial token from feat_dim → d_model
        self.mem_proj = nn.Linear(feat_dim, d_model)
        
        # Learnable gene queries: (num_genes, d_model)
        self.query_embed = nn.Parameter(torch.randn(num_genes, d_model))
        
        # Stack of Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final per-gene scalar projection
        self.to_logits = nn.Linear(d_model, 1)

    def forward(self, vit_features: Tensor) -> Tensor:
        """
        Process features through the gene transformer head.
        
        Args:
            vit_features: Input feature sequence from ViT [batch_size, seq_len, feat_dim]
                         where seq_len includes CLS token + patch tokens
            
        Returns:
            Gene expression predictions [batch_size, num_genes]
        """
        B, S, feat_dim = vit_features.shape
        
        # 1) Project spatial tokens to model dimension
        # Remove CLS token (first token) for spatial attention
        spatial_features = vit_features[:, 1:, :]  # [B, S-1, feat_dim]
        S_spatial = S - 1
        
        # Reshape for transformer: [seq_len, batch, feat_dim]
        mem = spatial_features.permute(1, 0, 2)  # [S_spatial, B, feat_dim]
        mem = self.mem_proj(mem)  # [S_spatial, B, d_model]

        # 2) Prepare queries: (T=num_genes, B, d_model) 
        T = self.query_embed.shape[0]
        q = self.query_embed.unsqueeze(1).expand(T, B, -1)

        # 3) Transformer decoder
        out = self.decoder(tgt=q, memory=mem)  # [T, B, d_model]

        # 4) To (B, T, d_model)
        out = out.permute(1, 0, 2)

        # 5) Project each gene-query to a scalar
        return self.to_logits(out).squeeze(-1)  # [B, num_genes]

class ViTGenePredictor(nn.Module):
    """
    ViT-based model for gene expression prediction.
    
    Wraps a Vision Transformer backbone for patch-level gene expression regression,
    taking a 224×224 RGB patch and predicting a vector of gene expression values.
    
    Attributes:
        vit (VisionTransformer): ViT backbone model
        head (nn.Module): Gene prediction head (regression or transformer)
    """
    
    def __init__(
        self, 
        vit_model: VisionTransformer, 
        num_genes: int,
        head_type: str = "simple",
        hidden_dim: int = 512,
        dropout: float = 0.1
    ) -> None:
        """
        Initialize the ViT Gene Predictor.
        
        Args:
            vit_model: Pre-configured ViT backbone model
            num_genes: Number of genes to predict
            head_type: Type of prediction head ("simple" or "transformer")
            hidden_dim: Hidden dimension for simple head
            dropout: Dropout probability
        """
        super().__init__()
        self.vit = vit_model
        self.head_type = head_type
        
        # Determine feature dimension from ViT
        feat_dim = self.vit.embed_dim
        
        # Create appropriate prediction head
        if head_type == "transformer":
            self.head = GeneTransformerHead(
                feat_dim=feat_dim,
                num_genes=num_genes,
                d_model=256,
                nhead=8,
                num_layers=2,
            )
        else:  # "simple"
            self.head = GeneRegressionHead(
                feat_dim=feat_dim,
                num_genes=num_genes,
                hidden_dim=hidden_dim,
                dropout=dropout
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the ViT Gene Predictor.
        
        Args:
            x: Input image tensor [batch_size, 3, 224, 224]
            
        Returns:
            Gene expression predictions [batch_size, num_genes]
        """
        # Extract features using ViT backbone
        vit_features = self.vit(x)  # [batch_size, seq_len, embed_dim]
        
        if self.head_type == "transformer":
            # Use all features for transformer head
            gene_predictions = self.head(vit_features)
        else:
            # Use CLS token for simple head
            cls_features = vit_features[:, 0]  # [batch_size, embed_dim]
            gene_predictions = self.head(cls_features)
            
        return gene_predictions
    
    def count_parameters(self) -> int:
        """
        Count total number of trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_vit_gene_model(
    model_size: str = "base",
    num_genes: int = 50,
    head_type: str = "simple",
    img_size: int = 224,
    pretrained: bool = False,
    dropout: float = 0.1
) -> ViTGenePredictor:
    """
    Factory function to create ViT gene predictor with different configurations.
    
    Args:
        model_size: Size of ViT backbone ("tiny", "small", "base")
        num_genes: Number of genes to predict
        head_type: Type of prediction head ("simple" or "transformer")
        img_size: Input image size
        pretrained: Whether to use pretrained weights (not implemented)
        
    Returns:
        Configured ViTGenePredictor model
    """
    from .vit import create_vit_tiny, create_vit_small, create_vit_base
    
    # Create backbone based on size
    if model_size == "tiny":
        vit_backbone = create_vit_tiny(img_size=img_size)
    elif model_size == "small":
        vit_backbone = create_vit_small(img_size=img_size)
    elif model_size == "base":
        vit_backbone = create_vit_base(img_size=img_size)
    else:
        raise ValueError(f"Unknown model size: {model_size}. Choose from 'tiny', 'small', 'base'")
    
    # Create gene predictor
    model = ViTGenePredictor(
        vit_model=vit_backbone,
        num_genes=num_genes,
        head_type=head_type
    )
    
    return model