"""
CiTGene model implementation for gene expression prediction from spatial transcriptomics images.

This module provides the architecture for predicting gene expression from image patches
using the CiT-Net backbone combined with a specialized gene prediction head.
"""

import torch
import torch.nn as nn
from torch import Tensor

from spatx_core.models.cit_to_gene.cit import CIT

class GeneTransformerHead(nn.Module):
    """
    Cross-attention head for gene expression prediction.
    
    Each of `num_genes` learnable query vectors attends over the H*W spatial tokens
    of the fused backbone output to predict gene expression levels.
    
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
            feat_dim: Dimension of the input features
            num_genes: Number of genes to predict
            d_model: Dimension of the transformer model
            nhead: Number of attention heads
            num_layers: Number of transformer decoder layers
        """
        super().__init__() #type: ignore
        # project each spatial token from feat_dim → d_model
        self.mem_proj = nn.Linear(feat_dim, d_model)
        # learnable gene‐queries: (num_genes, d_model)
        self.query_embed = nn.Parameter(torch.randn(num_genes, d_model))
        # stack of Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # final per‐gene scalar projection
        self.to_logits = nn.Linear(d_model, 1)

    def forward(self, bottleneck: Tensor) -> Tensor:
        """
        Process features through the gene transformer head.
        
        Args:
            bottleneck: Input feature map [batch_size, feat_dim, height, width]
            
        Returns:
            Gene expression predictions [batch_size, num_genes]
        """
        B, feat_dim, H, W = bottleneck.shape
        # 1) flatten spatial → (B, feat_dim, S) where S = H*W
        S = H * W
        mem = bottleneck.view(B, feat_dim, S).permute(2, 0, 1)  # [S, B, feat_dim]
        # 2) project to model-dim
        mem = self.mem_proj(mem)                                # [S, B, d_model]

        # 3) prepare queries: (T=num_genes, B, d_model) 
        T = self.query_embed.shape[0]
        q = self.query_embed.unsqueeze(1).expand(T, B, -1)

        # 4) Transformer decoder
        out = self.decoder(tgt=q, memory=mem)                   # [T, B, d_model]

        # 5) to (B, T, d_model)
        out = out.permute(1, 0, 2)

        # 6) project each gene‐query to a scalar
        return self.to_logits(out).squeeze(-1)                  # [B, num_genes]


class CITGenePredictor(nn.Module):
    """
    CIT-based model for gene expression prediction.
    
    Wraps a CiT-Net backbone for patch-level gene expression regression,
    taking a 224×224 RGB patch and predicting a vector of gene expression values.
    
    Attributes:
        cit (nn.Module): CiT-Net backbone model
        reg_head (nn.Sequential): Optional regression head
        flatten (nn.Flatten): Flattening layer
        head (GeneTransformerHead): Gene transformer prediction head
    """
    
    def __init__(self, cit_model: CIT, num_genes: int) -> None:
        """
        Initialize the CIT Gene Predictor.
        
        Args:
            cit_model: Pretrained CiT-Net model to use as backbone
            num_genes: Number of genes to predict
        """
        super().__init__() #type: ignore
        self.cit = cit_model
        
        # Define a new regression head: fuse Cnn4 & Swin4 features
        # embed_dim=96 => stage4 channels = embed_dim * 8 = 768 each
        fused_ch = self.cit.embed_dim * 8 * 2  # Cnn4 + Swin4
        self.reg_head = nn.Sequential(
             nn.AdaptiveAvgPool2d((1,1)),    # [B, fused_ch, 1,1]
             nn.Flatten(),                   # [B, fused_ch]
             nn.Linear(fused_ch, num_genes)   # [B, num_genes]
         )
        self.flatten = nn.Flatten()
        self.head = GeneTransformerHead(
            feat_dim=fused_ch,
            num_genes=num_genes,
            d_model=256,
            nhead=8,
            num_layers=2,
        )
       
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the CIT Gene Predictor.
        
        Args:
            x: Input image tensor [batch_size, 3, 224, 224]
            
        Returns:
            Gene expression predictions [batch_size, num_genes]
        """
        bottleneck = self.cit(x)
        expr = self.head(bottleneck)
        return expr