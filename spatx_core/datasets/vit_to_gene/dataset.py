"""
Module providing PyTorch Dataset implementations for ViT-to-gene prediction models.

This module contains dataset classes that transform raw data into the format
required by ViT-to-gene models, handling both training data (with gene expression values)
and prediction data (without gene expression values).
"""

import os
from typing import Tuple, cast

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms  # type: ignore
from torchvision.transforms import functional as F  # type: ignore
from PIL import Image

from ...data.data import TrainingData, PredictionData

class ViTTrainingDataset(Dataset[Tuple[Tensor, Tensor, str]]):
    """
    PyTorch Dataset for training ViT-to-gene models.
    
    This class transforms raw data into the format required by the ViTGene model,
    including image preprocessing and gene expression tensor creation.
    
    Attributes:
        data (TrainingData): The underlying data object providing access to training data points.
        transform (transforms.Compose): Image transformation pipeline for ViT.
    """

    def __init__(self, data: TrainingData) -> None:
        """
        Initialize the ViT training dataset.
        
        Args:
            data: TrainingData object providing access to training data points.
        """
        self.data = data
        
        # Define standard image transformations for ViT model
        # Using ImageNet preprocessing as is standard for ViT
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self) -> int:
        """
        Get the number of samples in the training dataset.
        
        Returns:
            Number of samples in the training dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, str]:
        """
        Get a single sample from the training dataset.
        
        Args:
            idx: Index of the sample to fetch.
            
        Returns:
            A tuple containing:
                - image_tensor: Tensor of shape [3, 224, 224], normalized with ImageNet stats.
                - expression_tensor: Tensor of shape [num_genes], containing gene expression values.
                - sample_id: Unique identifier for the sample.
                
        Raises:
            IOError: If the image file cannot be loaded.
        """
        # Get data point from the Data class
        datapoint = self.data[idx]
        
        # Load and transform image
        try:
            image = Image.open(datapoint.img_patch_path).convert('RGB')
            image = F.to_tensor(image)
            
            # Apply augmentations if present
            if datapoint.aug_seq:
                for aug in datapoint.aug_seq:
                    image = aug.apply(image)
            
            # Convert back to PIL for standard transforms
            image = F.to_pil_image(image)
            image_tensor = self.transform(image)
            
        except Exception as e:
            raise IOError(f"Could not load image at {datapoint.img_patch_path}: {str(e)}")
        
        # Convert gene expression dictionary to tensor
        # Ensure consistent ordering by sorting keys
        sorted_genes = sorted(datapoint.gene_expression.keys())
        expression_values = [datapoint.gene_expression[gene] for gene in sorted_genes]
        expression_tensor = torch.tensor(expression_values, dtype=torch.float32)
        
        # Create unique sample identifier
        sample_id = f"{datapoint.barcode}_{datapoint.wsi_id}" if datapoint.barcode and datapoint.wsi_id else f"sample_{idx}"
        
        return image_tensor, expression_tensor, sample_id

class ViTPredictionDataset(Dataset[Tuple[Tensor, str, int | None, int | None, str | None, str | None]]):
    """
    PyTorch Dataset for ViT-to-gene prediction.
    
    This class transforms raw data into the format required for prediction
    with the ViTGene model, focusing on image preprocessing.
    
    Attributes:
        prediction_data (PredictionData): The underlying data object providing access to prediction data points.
        transform (transforms.Compose): Image transformation pipeline for ViT.
    """
    
    def __init__(self, prediction_data: PredictionData) -> None:
        """
        Initialize the ViT prediction dataset.
        
        Args:
            prediction_data: PredictionData object providing access to prediction data points.
        """
        self.prediction_data = prediction_data
        
        # Define standard image transformations for ViT model
        # Using ImageNet preprocessing as is standard for ViT
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self) -> int:
        """
        Get the number of samples in the prediction dataset.
        
        Returns:
            Number of samples in the prediction dataset.
        """
        return len(self.prediction_data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, str, int | None, int | None, str | None, str | None]:
        """
        Get a single sample from the prediction dataset.
        
        Args:
            idx: Index of the sample to fetch.
            
        Returns:
            A tuple containing:
                - image_tensor: Tensor of shape [3, 224, 224], normalized with ImageNet stats.
                - sample_id: Unique identifier for the sample.
                - x: X coordinate of the patch (if available).
                - y: Y coordinate of the patch (if available).
                - wsi_id: Whole slide image ID (if available).
                - barcode: Barcode identifier (if available).
                
        Raises:
            IOError: If the image file cannot be loaded.
        """
        # Get data point from the PredictionData class
        datapoint = self.prediction_data[idx]
        
        # Load and transform image
        try:
            image = Image.open(datapoint.img_patch_path).convert('RGB')
            image_tensor = self.transform(image)
            
        except Exception as e:
            raise IOError(f"Could not load image at {datapoint.img_patch_path}: {str(e)}")
        
        # Create unique sample identifier
        sample_id = f"{datapoint.barcode}_{datapoint.wsi_id}" if datapoint.barcode and datapoint.wsi_id else f"sample_{idx}"
        
        # Cast coordinates to int or None
        x_coord = int(datapoint.x) if datapoint.x is not None else None
        y_coord = int(datapoint.y) if datapoint.y is not None else None
        
        return (
            image_tensor, 
            sample_id, 
            x_coord, 
            y_coord, 
            datapoint.wsi_id, 
            datapoint.barcode
        )

def create_training_dataset(data: TrainingData) -> ViTTrainingDataset:
    """
    Factory function to create ViT training dataset.
    
    Args:
        data: TrainingData object containing the training data
        
    Returns:
        ViTTrainingDataset instance
    """
    return ViTTrainingDataset(data)

def create_prediction_dataset(prediction_data: PredictionData) -> ViTPredictionDataset:
    """
    Factory function to create ViT prediction dataset.
    
    Args:
        prediction_data: PredictionData object containing the prediction data
        
    Returns:
        ViTPredictionDataset instance
    """
    return ViTPredictionDataset(prediction_data)