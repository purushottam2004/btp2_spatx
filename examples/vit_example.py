"""
Example script demonstrating how to train a ViT model for gene expression prediction.

This script shows how to use the ViT implementation with the spatial transcriptomics
data following the same patterns as the existing CiT implementation.
"""

import os
import logging
from tqdm import tqdm
import torch
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

# Import ViT components - following CIT pattern
from spatx_core.data.data import TrainingData, PredictionData
from spatx_core.datasets.vit_to_gene import ViTTrainingDataset, ViTPredictionDataset
from spatx_core.models.vit_to_gene import ViTGenePredictor, create_vit_gene_model
from spatx_core.models.vit_to_gene._loss import CombinedLoss as ViTCombinedLoss
from spatx_core.data_adapters.breast_data_adapters import BreastTrainingDataAdapter, BreastPredictionDataAdapter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

gene_ids = [
        'ABCC11', 'ADH1B', 'ADIPOQ', 'ANKRD30A', 'AQP1', 'AQP3', 'CCR7', 'CD3E', 'CEACAM6', 'CEACAM8',
        'CLIC6', 'CYTIP', 'DST', 'ERBB2', 'ESR1', 'FASN', 'GATA3', 'IL2RG', 'IL7R', 'KIT', 'KLF5', 
        'KRT14', 'KRT5', 'KRT6B', 'MMP1', 'MMP12', 'MS4A1', 'MUC6', 'MYBPC1', 'MYH11', 'MYLK', 
        'OPRPN', 'OXTR', 'PIGR', 'PTGDS', 'PTN', 'PTPRC', 'SCD', 'SCGB2A1', 'SERHL2', 'SERPINA3', 
        'SFRP1', 'SLAMF7', 'TACSTD2', 'TCL1A', 'TENT5C', 'TOP2A', 'TPSAB1', 'TRAC', 'VWF'
    ]

def calculate_pearson_scores(predictions: np.ndarray, targets: np.ndarray) -> tuple:
    """
    Calculate per-gene Pearson correlation scores.
    
    Args:
        predictions: numpy array of shape (num_samples, num_genes)
        targets: numpy array of shape (num_samples, num_genes)
    
    Returns:
        tuple: (per_gene_pearson list, mean_pearson float)
    """
    num_genes = predictions.shape[1]
    per_gene_pearson = []
    
    for g in range(num_genes):
        try:
            result = pearsonr(predictions[:, g].tolist(), targets[:, g].tolist())
            pc = result.correlation
            per_gene_pearson.append(float(pc) if not np.isnan(pc) else 0.0)
        except Exception as e:
            logger.debug(f"Failed to calculate Pearson correlation for gene {g}: {e}")
            per_gene_pearson.append(0.0)
    
    mean_pearson = float(np.mean(per_gene_pearson))
    return per_gene_pearson, mean_pearson


def train_vit_model():
    """
    Example function showing how to train a ViT model.
    Follows the same pattern as SimpleCITTrainer.
    """
    from torch.utils.data import DataLoader
    import torch.optim as optim
    
    # Data configuration
    data_dir = Path("data")
    breast_csv = data_dir / "breast.csv"
    image_dir = data_dir / "20x"
    
    # Training hyperparameters
    num_epochs = 10
    learning_rate = 3e-4
    batch_size = 32
    num_workers = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results_dir = "saved_models/vit_to_gene/experiment_1"
    
    # Gene IDs to predict (example - adjust as needed)
    
    # Create data adapters - following CIT pattern
    train_adapter = BreastTrainingDataAdapter(
        breast_csv=str(breast_csv),
        image_dir=str(image_dir),
        wsi_ids=["TENX99", "TENX95", "NCBI785", "NCBI784"],  # Training WSIs
        gene_ids=gene_ids,
    )
    
    validation_adapter = BreastTrainingDataAdapter(
        breast_csv=str(breast_csv),
        image_dir=str(image_dir),
        wsi_ids=["NCBI783"],  # Validation WSIs (ideally use different WSIs)
        gene_ids=gene_ids,
    )
    
    # Validate adapters
    if len(train_adapter) == 0 or len(validation_adapter) == 0:
        raise ValueError("Empty dataset detected")
    if set(train_adapter.gene_ids) != set(validation_adapter.gene_ids):
        raise ValueError("Train and validation adapters must have the same gene IDs")
    
    # Create TrainingData wrappers
    train_data = TrainingData(train_adapter)
    validation_data = TrainingData(validation_adapter)
    num_genes = len(gene_ids)
    
    # Create datasets
    train_dataset = ViTTrainingDataset(train_data)
    validation_dataset = ViTTrainingDataset(validation_data)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers
    )
    validation_dataloader = DataLoader(
        dataset=validation_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers
    )
    
    # Log training initialization
    logger.info("=" * 60)
    logger.info("ViT Training Initialization")
    logger.info("=" * 60)
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Validation dataset: {len(validation_dataset)} samples")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Device: {device}")
    logger.info(f"Number of genes: {num_genes}")
    logger.info("=" * 60)
    
    # Create ViT model - using factory function
    model = create_vit_gene_model(
        model_size='base',      # 'tiny', 'small', 'base'
        num_genes=num_genes,
        head_type='simple',     # 'simple' or 'transformer'
        pretrained=False
    )
    model.to(device)
    
    # Initialize optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = ViTCombinedLoss(alpha=0.5, beta=1.0, reg_lambda=0.0).to(device)
    
    # Training tracking
    best_val_loss = float('inf')
    best_model_state = None
    os.makedirs(results_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        
        for batch_idx, (images, expressions, _) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")):
            images = images.to(device)
            expressions = expressions.to(device)
            
            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, expressions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Collect predictions and targets for Pearson calculation
            train_predictions.append(predictions.detach().cpu())
            train_targets.append(expressions.detach().cpu())
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Calculate training Pearson scores
        train_preds_np = torch.cat(train_predictions, dim=0).numpy()
        train_targets_np = torch.cat(train_targets, dim=0).numpy()
        train_per_gene_pearson, train_mean_pearson = calculate_pearson_scores(train_preds_np, train_targets_np)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for images, expressions, _ in tqdm(validation_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = images.to(device)
                expressions = expressions.to(device)
                
                predictions = model(images)
                loss = criterion(predictions, expressions)
                val_loss += loss.item()
                
                # Collect predictions and targets for Pearson calculation
                val_predictions.append(predictions.cpu())
                val_targets.append(expressions.cpu())
        
        avg_val_loss = val_loss / len(validation_dataloader)
        
        # Calculate validation Pearson scores
        val_preds_np = torch.cat(val_predictions, dim=0).numpy()
        val_targets_np = torch.cat(val_targets, dim=0).numpy()
        val_per_gene_pearson, val_mean_pearson = calculate_pearson_scores(val_preds_np, val_targets_np)
        
        # Epoch summary
        logger.info(f"  Train Loss: {avg_train_loss:.6f}, Train Pearson: {train_mean_pearson:.6f}")
        logger.info(f"  Val Loss: {avg_val_loss:.6f}, Val Pearson: {val_mean_pearson:.6f}")
        
        # Log per-gene Pearson scores
        for g, gene_id in enumerate(gene_ids):
            logger.info(f"    Gene {gene_id}: Train Pearson={train_per_gene_pearson[g]:.6f}, Val Pearson={val_per_gene_pearson[g]:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            logger.info(f"  New best validation loss!")
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, os.path.join(results_dir, "best_model.pth"))
    
    # Restore and save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    logger.info("=" * 60)
    logger.info("Training Completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Model saved to: {results_dir}")
    logger.info("=" * 60)
    
    return model


def predict_with_vit_model():
    """
    Example function showing how to use a trained ViT model for prediction.
    Follows the same pattern as SimpleCITPredictor.
    """
    from torch.utils.data import DataLoader
    
    # Configuration
    data_dir = Path("data")
    breast_csv = data_dir / "breast.csv"
    image_dir = data_dir / "20x"
    model_path = "saved_models/vit_to_gene/experiment_1/best_model.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 8
    num_workers = 4
    
    # Gene IDs - must match those used during training
    num_genes = len(gene_ids)
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Train a model first.")
        return None
    
    # Create prediction adapter
    prediction_adapter = BreastPredictionDataAdapter(
        prediction_csv=str(breast_csv),
        image_dir=str(image_dir),
        wsi_ids=["TENX99"],
    )
    
    if len(prediction_adapter) == 0:
        raise ValueError("Empty prediction dataset")
    
    # Create PredictionData wrapper and dataset
    prediction_data = PredictionData(prediction_adapter)
    prediction_dataset = ViTPredictionDataset(prediction_data)
    prediction_dataloader = DataLoader(
        dataset=prediction_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    logger.info("=" * 60)
    logger.info("ViT Prediction")
    logger.info("=" * 60)
    logger.info(f"Prediction dataset: {len(prediction_dataset)} samples")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Device: {device}")
    logger.info("=" * 60)
    
    # Create and load model
    model = create_vit_gene_model(
        model_size='base',
        num_genes=num_genes,
        head_type='simple'
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Make predictions
    all_predictions = []
    all_sample_info = []
    
    with torch.no_grad():
        for batch_idx, (image_tensor, sample_id, x, y, wsi_id, barcode) in enumerate(tqdm(prediction_dataloader, desc="Predicting")):
            image_tensor = image_tensor.to(device)
            predictions = model(image_tensor)
            
            for i in range(image_tensor.shape[0]):
                all_predictions.append(predictions[i].cpu().numpy())
                all_sample_info.append({
                    'sample_id': sample_id[i] if sample_id else f"sample_{batch_idx * batch_size + i}",
                    'x': x[i].item() if x[i] is not None else None,
                    'y': y[i].item() if y[i] is not None else None,
                })
    
    logger.info(f"Made predictions for {len(all_predictions)} samples")
    logger.info("=" * 60)
    
    return all_predictions, all_sample_info, gene_ids


def compare_model_sizes():
    """
    Example function to compare different ViT model sizes.
    """
    model_sizes = ['tiny', 'small', 'base']
    
    print("\nViT Model Size Comparison")
    print("=" * 50)
    
    for size in model_sizes:
        model = create_vit_gene_model(
            model_size=size,
            num_genes=50,
            head_type='simple'
        )
        num_params = model.count_parameters()
        print(f"ViT-{size}: {num_params:,} trainable parameters")
    
    print("=" * 50)


if __name__ == "__main__":
    print("ViT Gene Expression Prediction Example")
    print("=" * 50)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Example 1: Compare model sizes
    print("\n1. Comparing ViT model sizes...")
    compare_model_sizes()
    
    # Example 2: Train a ViT model
    print("\n2. Training ViT model...")
    # Uncomment to train:
    model = train_vit_model()
    
    # Example 3: Make predictions with trained model
    print("\n3. Making predictions...")
    # Uncomment to predict (requires trained model):
    # predictions, sample_info, gene_ids = predict_with_vit_model()
    
    print("\nDone!")