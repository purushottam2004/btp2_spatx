"""
Training implementation for ViT-to-gene prediction models.

This module provides training functionality following the same pattern
as the CiT trainer implementation.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from typing import Dict, Optional, List, Tuple, Any
import wandb

from spatx_core.models.vit_to_gene.ViTGene import create_vit_gene_model
from spatx_core.models.vit_to_gene._loss import create_loss_function
from spatx_core.models.vit_to_gene._utils import (
    calculate_metrics, get_model_size, apply_gradient_clipping,
    warmup_cosine_schedule, EarlyStopping, log_model_info,
    save_checkpoint, load_checkpoint
)
from spatx_core.datasets.vit_to_gene.dataset import create_training_dataset, create_prediction_dataset

class SimpleViTTrainer:
    """
    Simple trainer for ViT-to-gene prediction models.
    
    Follows the same pattern as SimpleCITTrainer but adapted for Vision Transformers.
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        device: str = "auto"
    ):
        """
        Initialize the ViT trainer.
        
        Args:
            model_config: Configuration for the model
            training_config: Configuration for training
            device: Device to use for training
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Store configurations
        self.model_config = model_config
        self.training_config = training_config
        
        # Initialize model
        self.model = create_vit_gene_model(**model_config)
        self.model.to(self.device)
        
        # Initialize loss function
        loss_config = training_config.get("loss", {})
        self.loss_fn = create_loss_function(**loss_config)
        
        # Initialize optimizer
        optimizer_config = training_config.get("optimizer", {})
        self.optimizer = self._create_optimizer(optimizer_config)
        
        # Initialize scheduler
        scheduler_config = training_config.get("scheduler", {})
        self.scheduler = self._create_scheduler(scheduler_config)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {'train_loss': [], 'val_loss': [], 'metrics': []}
        
        # Early stopping
        early_stopping_config = training_config.get("early_stopping", {})
        if early_stopping_config.get("enabled", False):
            self.early_stopping = EarlyStopping(**{k: v for k, v in early_stopping_config.items() if k != "enabled"})
        else:
            self.early_stopping = None
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Log model information
        log_model_info(self.model, self.logger)
        
    def _create_optimizer(self, config: Dict[str, Any]) -> optim.Optimizer:
        """Create optimizer from configuration."""
        optimizer_type = config.get("type", "adamw")
        lr = config.get("learning_rate", 3e-4)
        weight_decay = config.get("weight_decay", 0.01)
        
        if optimizer_type.lower() == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == "sgd":
            momentum = config.get("momentum", 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def _create_scheduler(self, config: Dict[str, Any]) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler from configuration."""
        if not config or not config.get("enabled", False):
            return None
        
        scheduler_type = config.get("type", "cosine")
        
        if scheduler_type.lower() == "cosine":
            T_max = config.get("T_max", 100)
            eta_min = config.get("eta_min", 0)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
        elif scheduler_type.lower() == "step":
            step_size = config.get("step_size", 30)
            gamma = config.get("gamma", 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type.lower() == "reduce_on_plateau":
            mode = config.get("mode", "min")
            factor = config.get("factor", 0.5)
            patience = config.get("patience", 10)
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode=mode, factor=factor, patience=patience)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # Get gradient clipping config
        grad_clip = self.training_config.get("gradient_clipping", {})
        max_norm = grad_clip.get("max_norm", None)
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {self.current_epoch + 1}")
        
        for batch_idx, (images, gene_expressions, _) in enumerate(pbar):
            # Move batch to device
            images = images.to(self.device)
            gene_expressions = gene_expressions.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            
            # Calculate loss
            loss = self.loss_fn(predictions, gene_expressions)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if max_norm is not None:
                grad_norm = apply_gradient_clipping(self.model, max_norm)
            
            self.optimizer.step()
            
            # Update running loss
            total_loss += loss.item()
            
            # Store predictions for metrics
            all_predictions.append(predictions.detach())
            all_targets.append(gene_expressions.detach())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate epoch metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = calculate_metrics(all_predictions, all_targets, None)
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(val_loader, desc=f"Validation Epoch {self.current_epoch + 1}")
        
        with torch.no_grad():
            for batch_idx, (images, gene_expressions, _) in enumerate(pbar):
                # Move batch to device
                images = images.to(self.device)
                gene_expressions = gene_expressions.to(self.device)
                
                # Forward pass
                predictions = self.model(images)
                
                # Calculate loss
                loss = self.loss_fn(predictions, gene_expressions)
                
                # Update running loss
                total_loss += loss.item()
                
                # Store predictions for metrics
                all_predictions.append(predictions)
                all_targets.append(gene_expressions)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate epoch metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = calculate_metrics(all_predictions, all_targets, None)
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, metrics
    
    def train(
        self,
        train_data,
        val_data=None,
        epochs: int = None,
        save_dir: str = None,
        use_wandb: bool = False,
        wandb_config: Dict = None
    ) -> Dict[str, List]:
        """
        Train the model.
        
        Args:
            train_data: Training data
            val_data: Validation data (optional)
            epochs: Number of epochs to train
            save_dir: Directory to save model checkpoints
            use_wandb: Whether to use Weights & Biases logging
            wandb_config: Configuration for wandb
            
        Returns:
            Training history dictionary
        """
        if epochs is None:
            epochs = self.training_config.get("epochs", 100)
        
        # Create data loaders
        batch_size = self.training_config.get("batch_size", 32)
        num_workers = self.training_config.get("num_workers", 4)
        
        train_dataset = create_training_dataset(train_data)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        
        val_loader = None
        if val_data is not None:
            val_dataset = create_training_dataset(val_data)  # Use same format for validation
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True
            )
        
        # Initialize wandb if requested
        if use_wandb:
            wandb_config = wandb_config or {}
            wandb.init(
                project=wandb_config.get("project", "vit-gene-prediction"),
                config={**self.model_config, **self.training_config},
                **{k: v for k, v in wandb_config.items() if k != "project"}
            )
        
        # Training loop
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validation epoch
            val_loss = None
            val_metrics = None
            if val_loader is not None:
                val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loss is not None else train_loss)
                else:
                    self.scheduler.step()
            
            # Log metrics
            self.training_history['train_loss'].append(train_loss)
            if val_loss is not None:
                self.training_history['val_loss'].append(val_loss)
            
            epoch_metrics = {'epoch': epoch + 1, 'train': train_metrics}
            if val_metrics is not None:
                epoch_metrics['val'] = val_metrics
            self.training_history['metrics'].append(epoch_metrics)
            
            # Print epoch results
            log_str = f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}"
            if val_loss is not None:
                log_str += f", Val Loss: {val_loss:.4f}"
            if train_metrics:
                log_str += f", Train Corr: {train_metrics.get('correlation_mean', 0):.3f}"
            if val_metrics:
                log_str += f", Val Corr: {val_metrics.get('correlation_mean', 0):.3f}"
            
            self.logger.info(log_str)
            
            # Log to wandb
            if use_wandb:
                log_dict = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                
                # Add training metrics
                for key, value in train_metrics.items():
                    log_dict[f'train_{key}'] = value
                
                # Add validation metrics
                if val_loss is not None:
                    log_dict['val_loss'] = val_loss
                if val_metrics is not None:
                    for key, value in val_metrics.items():
                        log_dict[f'val_{key}'] = value
                
                wandb.log(log_dict)
            
            # Save checkpoint
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                
                # Save regular checkpoint
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
                save_checkpoint(
                    self.model, self.optimizer, epoch + 1, train_loss,
                    checkpoint_path, self.scheduler, epoch_metrics
                )
                
                # Save best model
                current_val_loss = val_loss if val_loss is not None else train_loss
                if current_val_loss < self.best_val_loss:
                    self.best_val_loss = current_val_loss
                    best_path = os.path.join(save_dir, 'best_model.pth')
                    save_checkpoint(
                        self.model, self.optimizer, epoch + 1, current_val_loss,
                        best_path, self.scheduler, epoch_metrics
                    )
                    self.logger.info(f"Saved new best model with validation loss: {current_val_loss:.4f}")
            
            # Check early stopping
            if self.early_stopping is not None:
                early_stop_loss = val_loss if val_loss is not None else train_loss
                if self.early_stopping(early_stop_loss, self.model):
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
        
        if use_wandb:
            wandb.finish()
        
        return self.training_history
    
    def load_model(self, checkpoint_path: str) -> None:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_info = load_checkpoint(checkpoint_path, self.model, self.optimizer, self.scheduler, self.device)
        self.current_epoch = checkpoint_info['epoch']
        self.best_val_loss = checkpoint_info['loss']
        self.logger.info(f"Loaded model from {checkpoint_path} (epoch {self.current_epoch})")
    
    def save_model(self, filepath: str) -> None:
        """
        Save current model state.
        
        Args:
            filepath: Path to save model
        """
        save_checkpoint(
            self.model, self.optimizer, self.current_epoch, self.best_val_loss,
            filepath, self.scheduler
        )
        self.logger.info(f"Saved model to {filepath}")

def create_simple_trainer(model_config: Dict[str, Any], training_config: Dict[str, Any], device: str = "auto") -> SimpleViTTrainer:
    """
    Factory function to create a simple ViT trainer.
    
    Args:
        model_config: Configuration for the model
        training_config: Configuration for training
        device: Device to use for training
        
    Returns:
        SimpleViTTrainer instance
    """
    return SimpleViTTrainer(model_config, training_config, device)