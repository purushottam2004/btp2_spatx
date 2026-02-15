# Vision Transformer (ViT) Model Details and Comparison with CiT

## Overview

The ViT implementation predicts gene expression from 224×224 H&E histology patches using a standard Vision Transformer architecture.

---

## ViT Architecture Components

### 1. Patch Embedding

- **Input**: 224×224 RGB image
- **Patch size**: 16×16 pixels
- Creates **(224/16)² = 196 patches**
- Each patch is flattened and projected to `embed_dim` dimensions via a Conv2d layer

### 2. Position Embedding

- Learnable positional embeddings added to patch tokens
- **CLS token**: A special learnable token prepended to the sequence (total: 197 tokens)

### 3. Transformer Encoder

Each layer contains:
- **Multi-Head Self-Attention (MHSA)**: Learns global relationships between all patches
- **MLP Block**: 2-layer feedforward with GELU activation and `mlp_ratio=4.0`
- **LayerNorm** applied before each sub-layer (Pre-LN)
- **Residual connections** around both sub-blocks
- **Drop Path** for stochastic depth regularization

### 4. Model Variants

| Variant | Embed Dim | Depth | Heads | Parameters (approx) |
|---------|-----------|-------|-------|---------------------|
| **ViT-Tiny** | 192 | 12 | 3 | ~5.5M |
| **ViT-Small** | 384 | 12 | 6 | ~22M |
| **ViT-Base** | 768 | 12 | 12 | ~86M |

**Current configuration**: ViT-Base (768 embed dim, 12 layers, 12 heads)

### 5. Gene Prediction Head

**Simple Head** (current implementation):
```
CLS token → LayerNorm → Linear(768→512) → GELU → Dropout 
         → Linear(512→256) → GELU → Dropout → Linear(256→num_genes)
```

**Transformer Head** (alternative):
- Learnable gene query vectors cross-attend to spatial patch tokens
- Similar to DETR-style object queries

---

## ViT vs CiT Comparison

### Architecture Overview

| Aspect | **ViT** | **CiT** |
|--------|---------|---------|
| **Architecture Type** | Pure Transformer | Hybrid CNN + Swin Transformer |
| **Patch Size** | 16×16 | 4×4 (finer patches) |
| **Attention Mechanism** | Global self-attention (all patches attend to all) | Windowed attention (7×7 windows) with shifted windows |
| **Spatial Processing** | Single-scale (all 196 patches at same resolution) | Multi-scale hierarchical (56→28→14→7 resolution) |
| **Feature Fusion** | Single transformer stream | Dual stream: CNN branch + Swin branch, fused at each scale |
| **Bottleneck Features** | CLS token (768-dim) or all patch tokens | Concatenated CNN4 + Swin4 features (768×2 = 1536 channels, 7×7 spatial) |
| **Parameter Efficiency** | More parameters for same receptive field | More efficient due to local windowed attention |
| **Inductive Bias** | None (learns everything from data) | Strong locality bias from CNN + windows |

---

## Key Architectural Differences

### CiT's Dual-Stream Design

```
Input Image
    ↓
ConvMixer (shared stem)
    ↓
┌───────────────┬───────────────┐
│   CNN Branch  │  Swin Branch  │
│   ConvBlock   │  BasicLayer   │
│   +MaxPool    │  +PatchMerge  │
└───────┬───────┴───────┬───────┘
        │               │
        └───Mid Fusion──┘  ← Fused at each scale
        ↓
   Bottleneck (1536 ch, 7×7)
        ↓
   GeneTransformerHead
```

### ViT's Single-Stream Design

```
Input Image (224×224)
    ↓
Patch Embedding (16×16 patches) → 196 tokens
    ↓
+ CLS Token + Position Embedding → 197 tokens
    ↓
12× [TransformerBlock: MHSA → MLP]
    ↓
CLS Token (768-dim)
    ↓
Gene Regression Head → num_genes predictions
```

---

## Computational Complexity

| Model | Attention Complexity |
|-------|---------------------|
| **ViT** | O(N²) where N = 196 patches (global attention) |
| **CiT (Swin)** | O(M²) where M = 49 patches per window (local attention) |

---

## Practical Considerations

| Factor | ViT | CiT |
|--------|-----|-----|
| **Data Efficiency** | Needs more data (no inductive bias) | Better with limited data (CNN bias helps) |
| **Resolution Flexibility** | Fixed patch count (must resize) | More flexible with hierarchical processing |
| **Feature Interpretation** | All patches equal importance | Multi-scale features capture both local + global patterns |
| **Training Stability** | May need careful LR scheduling | Often more stable due to locality |

---

## Configuration for Lung Dataset

### ViT-Base Setup
- **Number of genes**: 315
- **Loss function**: `CombinedLoss` with α=0.5 (L1 + Spearman correlation)
- **Optimizer**: AdamW with learning rate 3e-4, weight decay 0.01
- **Batch size**: 32

### How ViT Works for Gene Prediction
1. Learns global tissue patterns through self-attention
2. CLS token aggregates information from all 196 patches
3. Suitable when tissue patterns span entire patch

### When to Prefer CiT
- Local cellular patterns are important
- Limited training data available
- Multi-scale features matter (e.g., both cell-level and tissue-level patterns)

---

## Loss Function Details

### CombinedLoss
```python
loss = L1_loss(pred, target) + α * SpearmanLoss(pred, target)
```

- **L1 Loss**: Measures absolute differences in gene expression values
- **Spearman Loss**: Measures rank correlation (1 - correlation), ensuring predicted gene rankings match target rankings
- **α = 0.5**: Balances both objectives

The Spearman component uses a differentiable soft-ranking approximation via sigmoid functions.

---

## Summary

| | ViT | CiT |
|---|-----|-----|
| **Strength** | Global context modeling | Local + multi-scale features |
| **Weakness** | Data hungry, no local bias | More complex architecture |
| **Best for** | Large datasets, global patterns | Limited data, local patterns |
