"""
Prediction implementation for ViT-to-gene prediction models.

This module provides prediction functionality following the same pattern
as the CiT predictor implementation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, List, Optional, Any, Union
import pandas as pd

from spatx_core.models.vit_to_gene.ViTGene import create_vit_gene_model
from spatx_core.models.vit_to_gene._utils import calculate_metrics, log_model_info, load_checkpoint
from spatx_core.datasets.vit_to_gene.dataset import create_prediction_dataset
from spatx_core.data.data import PredictionData

class SimpleViTPredictor:
    """
    Simple predictor for ViT-to-gene prediction models.
    
    Follows the same pattern as SimpleCITPredictor but adapted for Vision Transformers.
    """
    
    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize the ViT predictor.
        
        Args:
            model_config: Configuration for the model (required if model_path not provided)
            model_path: Path to trained model checkpoint
            device: Device to use for prediction
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize model
        if model_path is not None:
            # Load model from checkpoint
            self.model = self._load_model_from_checkpoint(model_path, model_config)
        elif model_config is not None:
            # Create new model from config
            self.model = create_vit_gene_model(**model_config)
            self.model.to(self.device)
        else:
            raise ValueError("Either model_config or model_path must be provided")
        
        self.model.eval()
        
        # Log model information
        log_model_info(self.model, self.logger)
    
    def _load_model_from_checkpoint(self, model_path: str, model_config: Optional[Dict[str, Any]] = None) -> nn.Module:
        """
        Load model from checkpoint file.
        
        Args:
            model_path: Path to model checkpoint
            model_config: Optional model configuration
            
        Returns:
            Loaded model
        """
        # Load checkpoint to get model configuration if not provided
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if model_config is None:
            # Try to extract model config from checkpoint
            if 'model_config' in checkpoint:
                model_config = checkpoint['model_config']
            else:
                # Use default config
                self.logger.warning("No model config found in checkpoint, using default ViT-base configuration")
                model_config = {
                    'model_size': 'base',
                    'num_genes': 50,  # Default for breast cancer dataset
                    'head_type': 'simple'
                }
        
        # Create model - model_config is guaranteed to be a dict at this point
        assert model_config is not None
        model = create_vit_gene_model(**model_config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume the checkpoint is just the state dict
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        self.logger.info(f"Loaded model from {model_path}")
        
        return model
    
    def predict(
        self,
        data: PredictionData,
        batch_size: int = 32,
        num_workers: int = 4,
        return_confidence: bool = False,
        return_raw: bool = False
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Make predictions on data.
        
        Args:
            data: PredictionData object containing prediction samples
            batch_size: Batch size for prediction
            num_workers: Number of data loader workers
            return_confidence: Whether to return confidence intervals (not implemented for ViT)
            return_raw: Whether to return raw predictions without post-processing
            
        Returns:
            Predictions as numpy array or dictionary with additional info
        """
        # Create dataset and data loader
        dataset = create_prediction_dataset(data)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        # Make predictions
        all_predictions = []
        all_sample_ids = []
        
        self.model.eval()
        with torch.no_grad():
            for image_tensor, sample_id, x, y, wsi_id, barcode in tqdm(data_loader, desc="Predicting"):
                images = image_tensor.to(self.device)
                
                # Forward pass
                predictions = self.model(images)
                
                # Move to CPU and convert to numpy
                predictions_np = predictions.cpu().numpy()
                all_predictions.append(predictions_np)
                all_sample_ids.extend(sample_id)
        
        # Concatenate all predictions
        predictions = np.concatenate(all_predictions, axis=0)
        
        # Return results
        if return_raw or not return_confidence:
            if len(all_sample_ids) > 0:
                return {
                    'predictions': predictions,
                    'sample_ids': all_sample_ids
                }
            else:
                return predictions
        else:
            # Confidence intervals not implemented for ViT
            self.logger.warning("Confidence intervals not implemented for ViT predictor")
            if len(all_sample_ids) > 0:
                return {
                    'predictions': predictions,
                    'sample_ids': all_sample_ids,
                    'confidence_lower': None,
                    'confidence_upper': None
                }
            else:
                return {
                    'predictions': predictions,
                    'confidence_lower': None,
                    'confidence_upper': None
                }
    
    def predict_single(
        self,
        data: PredictionData,
        return_confidence: bool = False
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Make prediction on a single sample.
        
        Args:
            data: PredictionData instance (should contain single sample)
            return_confidence: Whether to return confidence intervals
            
        Returns:
            Prediction as numpy array or dictionary
        """
        results = self.predict(data, batch_size=1, return_confidence=return_confidence)
        
        if isinstance(results, dict):
            # Extract single sample results
            single_results = {}
            for key, value in results.items():
                if key == 'sample_ids':
                    single_results[key] = value[0] if value else None
                elif value is not None:
                    single_results[key] = value[0]
                else:
                    single_results[key] = None
            return single_results
        else:
            return results[0]
    
    def evaluate(
        self,
        data: PredictionData,
        targets: np.ndarray,
        batch_size: int = 32,
        num_workers: int = 4,
        masks: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance on data with known targets.
        
        Args:
            data: PredictionData object containing prediction samples
            targets: True gene expression values
            batch_size: Batch size for prediction
            num_workers: Number of data loader workers
            masks: Optional masks for valid genes
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        predictions = self.predict(data, batch_size=batch_size, num_workers=num_workers)
        
        if isinstance(predictions, dict):
            predictions = predictions['predictions']
        
        # Convert to tensors for metrics calculation
        pred_tensor = torch.from_numpy(predictions)
        target_tensor = torch.from_numpy(targets)
        
        # Calculate metrics
        if masks is not None:
            mask_tensor = torch.from_numpy(masks)
            metrics = calculate_metrics(pred_tensor, target_tensor, mask_tensor)
        else:
            # Call without mask - passing a dummy all-ones mask
            metrics = calculate_metrics(pred_tensor, target_tensor, torch.ones_like(pred_tensor))
        
        return metrics
    
    def predict_to_dataframe(
        self,
        data: PredictionData,
        gene_names: Optional[List[str]] = None,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> pd.DataFrame:
        """
        Make predictions and return as pandas DataFrame.
        
        Args:
            data: PredictionData object containing prediction samples
            gene_names: Names of genes (for column names)
            batch_size: Batch size for prediction
            num_workers: Number of data loader workers
            
        Returns:
            DataFrame with predictions
        """
        results = self.predict(data, batch_size=batch_size, num_workers=num_workers)
        
        if isinstance(results, dict):
            predictions = results['predictions']
            sample_ids = results.get('sample_ids', None)
        else:
            predictions = results
            sample_ids = None
        
        # Create column names
        if gene_names is not None:
            columns = gene_names
        else:
            columns = [f'gene_{i}' for i in range(predictions.shape[1])]
        
        # Create DataFrame
        df = pd.DataFrame(predictions, columns=columns)
        
        # Add sample IDs if available
        if sample_ids is not None:
            df.insert(0, 'sample_id', sample_ids)
        
        return df
    
    def save_predictions(
        self,
        data: PredictionData,
        output_path: str,
        gene_names: Optional[List[str]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        file_format: str = 'csv'
    ) -> None:
        """
        Make predictions and save to file.
        
        Args:
            data: PredictionData object containing prediction samples
            output_path: Path to save predictions
            gene_names: Names of genes
            batch_size: Batch size for prediction
            num_workers: Number of data loader workers
            file_format: File format ('csv', 'parquet', 'pickle')
        """
        df = self.predict_to_dataframe(
            data, gene_names=gene_names, batch_size=batch_size, num_workers=num_workers
        )
        
        if file_format.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif file_format.lower() == 'parquet':
            df.to_parquet(output_path, index=False)
        elif file_format.lower() == 'pickle':
            df.to_pickle(output_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        self.logger.info(f"Saved predictions to {output_path}")
    
    def get_feature_importance(self, data: PredictionData, method: str = "attention") -> Optional[np.ndarray]:
        """
        Get feature importance for a prediction.
        
        Args:
            data: PredictionData instance for computing importance
            method: Method for computing importance ("attention", "gradient")
            
        Returns:
            Feature importance array or None if not implemented
        """
        if method == "attention":
            return self._get_attention_importance(data)
        elif method == "gradient":
            return self._get_gradient_importance(data)
        else:
            raise ValueError(f"Unknown importance method: {method}")
    
    def _get_attention_importance(self, data: PredictionData) -> Optional[np.ndarray]:
        """Get attention-based feature importance."""
        # This would require modifying the model to return attention weights
        self.logger.warning("Attention-based importance not implemented")
        return None
    
    def _get_gradient_importance(self, data: PredictionData) -> np.ndarray:
        """Get gradient-based feature importance."""
        dataset = create_prediction_dataset(data)
        data_loader = DataLoader(dataset, batch_size=1)
        image_tensor, sample_id, x, y, wsi_id, barcode = next(iter(data_loader))
        
        image = image_tensor.to(self.device)
        image.requires_grad_(True)
        
        self.model.eval()
        prediction = self.model(image)
        
        # Compute gradients with respect to input
        grad_outputs = torch.ones_like(prediction)
        gradients = torch.autograd.grad(
            outputs=prediction,
            inputs=image,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False
        )[0]
        
        # Compute importance as absolute gradient values
        importance = torch.abs(gradients).mean(dim=1).squeeze().cpu().numpy()
        
        return importance

def create_simple_predictor(
    model_config: Optional[Dict[str, Any]] = None,
    model_path: Optional[str] = None,
    device: str = "auto"
) -> SimpleViTPredictor:
    """
    Factory function to create a simple ViT predictor.
    
    Args:
        model_config: Configuration for the model
        model_path: Path to trained model checkpoint
        device: Device to use for prediction
        
    Returns:
        SimpleViTPredictor instance
    """
    return SimpleViTPredictor(model_config, model_path, device)