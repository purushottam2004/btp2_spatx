import os
import logging

import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr #type: ignore

from ...data.data import TrainingData
from ...data_adapters import BreastTrainingDataAdapter
from ...datasets.cit_to_gene.dataset import CITTrainingDataset
from ...models.cit_to_gene.cit import CIT
from ...models.cit_to_gene.CiTGene import CITGenePredictor
from ...models.cit_to_gene._loss import CombinedLoss

logger = logging.getLogger(__name__)

class Results:
    def __init__(self, save_path: str | None = None):
        # Storage for per-epoch metrics
        self.train_metrics = {  #type: ignore
            'per_gene_loss': [],      # List[List[float]] - [epoch][gene_idx]
            'per_gene_pearson': [],   # List[List[float]] - [epoch][gene_idx]
            'mean_loss': [],          # List[float] - [epoch]
            'mean_pearson': [],       # List[float] - [epoch]
            'loss_variance': [],      # List[float] - [epoch]
            'pearson_variance': []    # List[float] - [epoch]
        }
        
        self.val_metrics = { #type: ignore
            'per_gene_loss': [],
            'per_gene_pearson': [],
            'mean_loss': [],
            'mean_pearson': [],
            'loss_variance': [],
            'pearson_variance': []
        }
        
        # Best metrics tracking
        self.best_metrics = { #type: ignore
            'per_gene': {
                'best_loss': None,        # List[float] - best loss per gene
                'best_pearson': None,     # List[float] - best pearson per gene
                'best_loss_epoch': None,  # List[int] - epoch of best loss per gene
                'best_pearson_epoch': None # List[int] - epoch of best pearson per gene
            },
            'overall': {
                'best_mean_loss': float('inf'),
                'best_mean_pearson': -1.0,
                'best_loss_epoch': 0,
                'best_pearson_epoch': 0
            }
        }
        
        self.save_path = save_path
        self.best_epoch = 0
        self.best_val_loss = float('inf')
        
    def update_metrics(self, epoch: int, phase: str,
                      all_predictions: torch.Tensor, all_targets: torch.Tensor,
                      epoch_loss: float):
        """Update metrics for given epoch and phase (train/val)"""
        metrics = self.train_metrics if phase == 'train' else self.val_metrics #type: ignore
        
        # Convert to numpy for calculations
        preds = all_predictions.cpu().numpy()
        targets = all_targets.cpu().numpy()
        
        # Per-gene metrics
        num_genes = preds.shape[1]
        per_gene_loss = []
        per_gene_pearson = []
        
        for g in range(num_genes):
            # MSE per gene
            gene_loss = np.mean((preds[:, g] - targets[:, g]) ** 2)
            per_gene_loss.append(float(gene_loss)) #type: ignore
            
            # Pearson correlation per gene
            try:
                result = pearsonr(preds[:, g].tolist(), targets[:, g].tolist())
                pc = result.correlation  # This is the correlation coefficient #type: ignore
                per_gene_pearson.append(float(pc) if not np.isnan(pc) else 0.0) #type: ignore
            except Exception as e:
                logger.debug(f"Failed to calculate Pearson correlation: {e}")
                per_gene_pearson.append(0.0) #type: ignore
        
        # Update metrics
        metrics['per_gene_loss'].append(per_gene_loss) #type: ignore
        metrics['per_gene_pearson'].append(per_gene_pearson) #type: ignore
        metrics['mean_loss'].append(float(epoch_loss)) #type: ignore
        mean_pearson = float(np.mean(per_gene_pearson)) #type: ignore
        metrics['mean_pearson'].append(mean_pearson) #type: ignore
        metrics['loss_variance'].append(float(np.var(per_gene_loss))) #type: ignore
        metrics['pearson_variance'].append(float(np.var(per_gene_pearson))) #type: ignore
        
        # Update best metrics if this is validation phase
        if phase == 'val':
            # Initialize best metrics arrays if not done yet
            if self.best_metrics['per_gene']['best_loss'] is None: #type: ignore
                num_genes = len(per_gene_loss) #type: ignore
                self.best_metrics['per_gene']['best_loss'] = [float('inf')] * num_genes #type: ignore
                self.best_metrics['per_gene']['best_pearson'] = [-1.0] * num_genes #type: ignore
                self.best_metrics['per_gene']['best_loss_epoch'] = [0] * num_genes #type: ignore
                self.best_metrics['per_gene']['best_pearson_epoch'] = [0] * num_genes #type: ignore
            
            # Update per-gene best metrics
            for g in range(len(per_gene_loss)): #type: ignore
                if per_gene_loss[g] < self.best_metrics['per_gene']['best_loss'][g]: #type: ignore
                    self.best_metrics['per_gene']['best_loss'][g] = per_gene_loss[g] #type: ignore
                    self.best_metrics['per_gene']['best_loss_epoch'][g] = epoch #type: ignore
                
                if per_gene_pearson[g] > self.best_metrics['per_gene']['best_pearson'][g]: #type: ignore
                    self.best_metrics['per_gene']['best_pearson'][g] = per_gene_pearson[g] #type: ignore
                    self.best_metrics['per_gene']['best_pearson_epoch'][g] = epoch #type: ignore
            
            # Update overall best metrics
            if epoch_loss < self.best_metrics['overall']['best_mean_loss']: #type: ignore
                self.best_metrics['overall']['best_mean_loss'] = epoch_loss #type: ignore
                self.best_metrics['overall']['best_loss_epoch'] = epoch #type: ignore
            
            if mean_pearson > self.best_metrics['overall']['best_mean_pearson']: #type: ignore
                self.best_metrics['overall']['best_mean_pearson'] = mean_pearson #type: ignore
                self.best_metrics['overall']['best_pearson_epoch'] = epoch #type: ignore
            
            # Update best validation loss for model saving
            if epoch_loss < self.best_val_loss:
                self.best_val_loss = epoch_loss
                self.best_epoch = epoch
                self._save_metrics()
    
    def _save_metrics(self):
        """Save metrics to file if save_path is provided"""
        if not self.save_path:
            return
            
        os.makedirs(self.save_path, exist_ok=True)
        
        # Save metrics to files
        for phase in ['train', 'val']:
            metrics = self.train_metrics if phase == 'train' else self.val_metrics #type: ignore
            
            # Save detailed results
            with open(os.path.join(self.save_path, f'{phase}_metrics.txt'), 'w') as f:
                # Write best overall metrics
                f.write("Best Overall Metrics:\n")
                f.write("=" * 50 + "\n")
                f.write(f"Best Mean Loss: {self.best_metrics['overall']['best_mean_loss']:.6f} " #type: ignore
                       f"(Epoch {self.best_metrics['overall']['best_loss_epoch']})\n") #type: ignore
                f.write(f"Best Mean Pearson: {self.best_metrics['overall']['best_mean_pearson']:.6f} " #type: ignore
                       f"(Epoch {self.best_metrics['overall']['best_pearson_epoch']})\n") #type: ignore
                f.write("\nBest Per-Gene Metrics:\n")
                f.write("-" * 40 + "\n")
                
                # Write best per-gene metrics
                for g in range(len(self.best_metrics['per_gene']['best_loss'])): #type: ignore
                    f.write(f"Gene {g}:\n")
                    f.write(f"  Best Loss: {self.best_metrics['per_gene']['best_loss'][g]:.6f} " #type: ignore
                           f"(Epoch {self.best_metrics['per_gene']['best_loss_epoch'][g]})\n") #type: ignore
                    f.write(f"  Best Pearson: {self.best_metrics['per_gene']['best_pearson'][g]:.6f} " #type: ignore
                           f"(Epoch {self.best_metrics['per_gene']['best_pearson_epoch'][g]})\n") #type: ignore
                
                f.write("\nPer-Epoch Metrics:\n")
                f.write("=" * 50 + "\n\n")
                
                # Write per-epoch metrics
                for epoch in range(len(metrics['mean_loss'])): #type: ignore
                    f.write(f"Epoch {epoch}:\n")
                    f.write(f"Mean Loss: {metrics['mean_loss'][epoch]:.6f}\n")
                    f.write(f"Mean Pearson: {metrics['mean_pearson'][epoch]:.6f}\n")
                    f.write(f"Loss Variance: {metrics['loss_variance'][epoch]:.6f}\n")
                    f.write(f"Pearson Variance: {metrics['pearson_variance'][epoch]:.6f}\n")
                    f.write("\nPer-gene metrics:\n")
                    for g in range(len(metrics['per_gene_loss'][epoch])): #type: ignore
                        f.write(f"Gene {g}: Loss={metrics['per_gene_loss'][epoch][g]:.6f}, "
                               f"Pearson={metrics['per_gene_pearson'][epoch][g]:.6f}\n")
                    f.write("\n" + "-"*40 + "\n")

    
class SimpleCITTrainer:
    def __init__(self, 
                 train_adapter : BreastTrainingDataAdapter,
                 validation_adapter : BreastTrainingDataAdapter,
                 num_epochs: int = 100, 
                 learning_rate: float = 0.001, 
                 device: str  = 'cuda',
                 results_dir: str | None = None,
                 batch_size : int = 8,
                 num_workers : int = 4,
                 logging_interval: int = 1,
                 ):
        if len(train_adapter) == 0 or len(validation_adapter) == 0:
            raise ValueError("Empty dataset detected")
        if batch_size > len(train_adapter):
            raise ValueError("Batch size cannot be larger than train dataset size")
        if batch_size > len(validation_adapter):
            raise ValueError("Batch size cannot be larger than validation dataset size")
        if set(train_adapter.gene_ids) != set(validation_adapter.gene_ids):
            raise ValueError("Train and validation adapters must have the same gene IDs")
        
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.logging_interval = logging_interval
        if device not in ['cpu', 'cuda', None]:
            raise ValueError(f"The device must be cpu or cuda, currently it is {device}")
        if device == 'cuda' and (not torch.cuda.is_available()):
            raise ValueError("Cuda is not available, please don't use")
        self.device = device
        self.results = Results(save_path=results_dir)
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
        if results_dir and not os.access(os.path.dirname(results_dir), os.W_OK):
            raise ValueError(f"Results path {results_dir} is not writable")
        self.train_data = TrainingData(train_adapter)
        self.validation_data = TrainingData(validation_adapter)
        self.gene_ids = train_adapter.gene_ids
        self.num_genes = len(self.gene_ids)

    def train(self):
        # Initialize datasets and dataloaders
        train_dataset = CITTrainingDataset(self.train_data)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, 
                                    shuffle=True, num_workers=self.num_workers)
        validation_dataset = CITTrainingDataset(self.validation_data)
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=self.batch_size, 
                                         shuffle=False, num_workers=self.num_workers)
        
        # Log training initialization
        logger.info("=" * 60)
        logger.info("Training Initialization")
        logger.info("=" * 60)
        logger.info(f"Starting training with {self.num_epochs} epochs")
        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Validation dataset: {len(validation_dataset)} samples")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Number of genes: {self.num_genes}")
        logger.info(f"Log interval: {self.logging_interval} batch(es)")
        if self.results.save_path:
            logger.info(f"Results will be saved to: {self.results.save_path}")
        else:
            logger.info("Results will not be saved to file")
        logger.info("=" * 60)
        
        # Initialize model and training components
        cit_backbone = CIT(img_size=224, in_chans=3,
                          embed_dim=96,
                          depths=[2,2,6,2],
                          num_heads=[3,6,12,24],
                          device= self.device)
        model = CITGenePredictor(cit_backbone, num_genes=self.num_genes)
        model.to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr= self.learning_rate)
        criterion = CombinedLoss(alpha=0.5, reg=1.0).to(self.device)
        best_val_loss = float('inf')
        best_model_state = None
        
        # Training loop
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs} started")
            
            # Training phase
            logger.info("Training phase started")
            model.train()
            train_loss = 0.0
            train_predictions = []
            train_targets = []
            
            for batch_idx, (images, expressions, _) in enumerate(train_dataloader):
                # Move batch to device
                images = images.to(self.device)
                expressions = expressions.to(self.device)
                logger.debug(f"Batch {batch_idx+1}/{len(train_dataloader)} - Image tensors shape: {images.shape}, Expression tensors shape: {expressions.shape}")
                # Forward pass
                predictions = model(images)
                loss = criterion(predictions, expressions)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() #type: ignore
                
                # Collect predictions and targets for metrics
                train_predictions.append(predictions.detach()) #type: ignore
                train_targets.append(expressions.detach()) #type: ignore
                train_loss += loss.item()
                
                # Log batch progress
                if (batch_idx + 1) % self.logging_interval == 0:
                    progress = (batch_idx + 1) / len(train_dataloader) * 100
                    logger.info(f"Epoch {epoch + 1}: Batch {batch_idx + 1}/{len(train_dataloader)} processed ({progress:.1f}%) - Loss: {loss.item():.6f}")
            
            # Compute epoch metrics for training
            avg_train_loss = train_loss / len(train_dataloader)
            train_predictions = torch.cat(train_predictions, dim=0) #type: ignore
            train_targets = torch.cat(train_targets, dim=0) #type: ignore
            self.results.update_metrics(epoch, 'train', 
                                     train_predictions, train_targets, 
                                     avg_train_loss)
            
            logger.info(f"Training phase completed - Average Loss: {avg_train_loss:.6f}")
            
            # Validation phase
            logger.info("Validation phase started")
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for images, expressions, _ in validation_dataloader:
                    images = images.to(self.device)
                    expressions = expressions.to(self.device)
                    
                    predictions = model(images)
                    loss = criterion(predictions, expressions)
                    
                    # Collect predictions and targets for metrics
                    val_predictions.append(predictions) #type: ignore
                    val_targets.append(expressions) #type: ignore
                    val_loss += loss.item()
            
            # Compute epoch metrics for validation
            avg_val_loss = val_loss / len(validation_dataloader)
            val_predictions = torch.cat(val_predictions, dim=0) #type: ignore
            val_targets = torch.cat(val_targets, dim=0) #type: ignore
            self.results.update_metrics(epoch, 'val',
                                     val_predictions, val_targets,
                                     avg_val_loss)
            
            logger.info(f"Validation phase completed - Average Loss: {avg_val_loss:.6f}")
            
            # Calculate validation metrics for logging
            val_mean_pearson = self.results.val_metrics['mean_pearson'][-1] #type: ignore
            
            # Epoch summary
            logger.info(f"Epoch {epoch + 1} Summary:")
            logger.info(f"  Train Loss: {avg_train_loss:.6f}")
            logger.info(f"  Val Loss: {avg_val_loss:.6f}")
            logger.info(f"  Val Pearson: {val_mean_pearson:.6f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                logger.info(f"New best validation loss: {avg_val_loss:.6f} (improved from {best_val_loss:.6f})")
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
            torch.cuda.empty_cache()
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info("Best model restored")
        
        # Training completion logging
        logger.info("=" * 60)
        logger.info("Training Completed Successfully")
        logger.info("=" * 60)
        logger.info(f"Best validation loss achieved: {best_val_loss:.6f}")
        logger.info(f"Best validation loss epoch: {self.results.best_epoch}")
        logger.info(f"Best overall mean loss: {self.results.best_metrics['overall']['best_mean_loss']:.6f}") #type: ignore
        logger.info(f"Best overall mean Pearson: {self.results.best_metrics['overall']['best_mean_pearson']:.6f}") #type: ignore
        if self.results.save_path:
            logger.info(f"Detailed results saved to: {self.results.save_path}")
        logger.info("=" * 60)
        
        return model, self.results