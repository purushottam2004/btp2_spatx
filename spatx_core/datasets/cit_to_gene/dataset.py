"""
Module providing PyTorch Dataset implementations for CIT-to-gene prediction models.

This module contains dataset classes that transform raw data into the format
required by CIT-to-gene models, handling both training data (with gene expression values)
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

class CITTrainingDataset(Dataset[Tuple[Tensor, Tensor, str]]):
    """
    PyTorch Dataset for training CIT-to-gene models.
    
    This class transforms raw data into the format required by the CitToGene model,
    including image preprocessing and gene expression tensor creation.
    
    Attributes:
        data (Data): The underlying data object providing access to data points.
        transform (transforms.Compose): Image transformation pipeline.
    """

    def __init__(self, data: TrainingData) -> None:
        """
        Initialize the CIT dataset.
        
        Args:
            data: Data object providing access to training data points.
        """
        self.data = data
        # Define standard image transformations for CIT model
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            Number of samples in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, str]:
        """
        Get a single sample from the dataset.
        
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
            if datapoint.aug_seq:
                for aug in datapoint.aug_seq:
                    image = aug.apply(image)
            image = cast(Tensor, image)
            image_tensor = self.transform(image)
        except (IOError, OSError) as e:
            from ...constants.error_msg import ErrorMessage
            raise IOError(ErrorMessage.failed_to_load_image(datapoint.img_patch_path)) from e
        
        # Convert gene expression dictionary to tensor
        # Ensure consistent gene ordering by sorting keys
        gene_names = sorted(datapoint.gene_expression.keys())
        expression_values = [datapoint.gene_expression[gene] for gene in gene_names]
        expression_tensor = torch.tensor(expression_values, dtype=torch.float32)
        
        # Create sample ID from coordinates
        sample_id = f"{datapoint.barcode}_{datapoint.wsi_id}"
        return image_tensor, expression_tensor, sample_id


class CITPredictionDataset(Dataset[Tuple[Tensor, str, int | None, int | None, str | None, str | None]]):
    """
    PyTorch Dataset for CIT-to-gene prediction.
    
    This class transforms raw data into the format required for prediction
    with the CitToGene model, focusing on image preprocessing.
    
    Attributes:
        prediction_data (PredictionData): The underlying data object providing access to prediction data points.
        transform (transforms.Compose): Image transformation pipeline.
    """
    
    def __init__(self, prediction_data: PredictionData) -> None:
        """
        Initialize the CIT prediction dataset.
        
        Args:
            prediction_data: PredictionData object providing access to prediction data points.
        """
        self.prediction_data = prediction_data
        # Define standard image transformations for CIT model
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
                - sample_info: Dictionary containing metadata about the sample.
                  Keys: 'image_name', 'x', 'y', 'barcode', 'wsi_id'.
        Raises:
            IOError: If the image file cannot be loaded.
        """
        # Get data point from the PredictionData class
        datapoint = self.prediction_data[idx]
        
        # Load and transform image
        try:
            image = Image.open(datapoint.img_patch_path).convert('RGB')
            image_tensor: Tensor = self.transform(image) #type:ignore
        except (IOError, OSError) as e:
            from ...constants.error_msg import ErrorMessage
            raise IOError(ErrorMessage.failed_to_load_image(datapoint.img_patch_path)) from e
        
        # Extract image name from path
        image_name = os.path.basename(datapoint.img_patch_path)

        return image_tensor, image_name, datapoint.x, datapoint.y, datapoint.barcode, datapoint.wsi_id