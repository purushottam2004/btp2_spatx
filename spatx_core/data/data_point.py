"""
Module for handling data points in spatial transcriptomics analysis.

This module provides classes for representing data points with gene expression
and prediction data points for machine learning models.
"""

import warnings
import os
from typing import Dict, Optional, List

from ..constants.error_msg import ErrorMessage
from ..constants.warning_msg import WarningMessages
from ..augmentation.augmentation import BaseAugmentation

class BaseDataPoint:
    """
    Base class for spatial transcriptomics data points.
    
    Contains common attributes and validation logic shared between training
    and prediction data points.
    
    Attributes:
        x (Optional[int]): X-coordinate in the spatial reference frame.
        y (Optional[int]): Y-coordinate in the spatial reference frame.
        img_patch_path (str): Path to the image patch associated with this data point.
        wsi_id (Optional[str]): Whole slide image ID, if applicable.
        barcode (Optional[str]): Unique barcode for this data point, if applicable.
    """
    
    def __init__(self, x: Optional[int], y: Optional[int], img_patch_path: str,
                 wsi_id: Optional[str] = None, 
                 barcode: Optional[str] = None) -> None:
        """
        Initialize a BaseDataPoint instance.
        
        Args:
            x (Optional[int]): X-coordinate in the spatial reference frame.
            y (Optional[int]): Y-coordinate in the spatial reference frame.
            img_patch_path (str): Path to the image patch associated with this data point.
            wsi_id (Optional[str], optional): Whole slide image ID. Defaults to None.
            barcode (Optional[str], optional): Unique barcode for this data point. Defaults to None.
        """
        self.x = x
        self.y = y
        self.img_patch_path = img_patch_path
        self.wsi_id = wsi_id
        self.barcode = barcode
    
    def _validate_common_attributes(self, adapter_name: str) -> None:
        """
        Validate common attributes shared by all data point types.
        
        Args:
            adapter_name (str): Name of the adapter providing this data point, for error reporting.
            
        Raises:
            ValueError: If the image path does not exist.
        """
        if self.wsi_id is None:
            warnings.warn(WarningMessages.wsi_missing_in_adapter(adapter_name))
        if self.barcode is None:
            warnings.warn(WarningMessages.barcode_missing_in_adapter(adapter_name))
        if not os.path.exists(self.img_patch_path):
            raise ValueError(ErrorMessage.image_missing_at_path_of_adapter(self.img_patch_path, adapter_name))
        if not isinstance(self.x, (float, int)) or not isinstance(self.y, (float, int)): #type: ignore
            warnings.warn(WarningMessages.adapter_not_returning_x_y_coordinates(adapter_name, self.x, self.y))

class TrainingDataPoint(BaseDataPoint):
    """
    Represents a single data point in spatial transcriptomics data.
    
    A data point contains spatial coordinates, an image patch, gene expression values,
    and optional metadata such as whole slide image ID and barcode.
    
    Attributes:
        x (int): X-coordinate in the spatial reference frame.
        y (int): Y-coordinate in the spatial reference frame.
        img_patch_path (str): Path to the image patch associated with this data point.
        gene_expression (Dict[str, float]): Dictionary mapping gene IDs to expression values.
        wsi_id (Optional[str]): Whole slide image ID, if applicable.
        barcode (Optional[str]): Unique barcode for this data point, if applicable.
        aug_seq (List[BaseAugmentation] | None): List of augmentations to apply, if any.
    """
    
    def __init__(self, x: Optional[int], y: Optional[int], img_patch_path: str, 
                 gene_expression: Dict[str, float], 
                 wsi_id: Optional[str] = None, 
                 barcode: Optional[str] = None,
                 aug_seq : List[BaseAugmentation] | None = None) -> None:
        """
        Initialize a TrainingDataPoint instance.
        
        Args:
            x (Optional[int]): X-coordinate in the spatial reference frame.
            y (Optional[int]): Y-coordinate in the spatial reference frame.
            img_patch_path (str): Path to the image patch associated with this data point.
            gene_expression (Dict[str, float]): Dictionary mapping gene IDs to expression values.
            wsi_id (Optional[str], optional): Whole slide image ID. Defaults to None.
            barcode (Optional[str], optional): Unique barcode for this data point. Defaults to None.
            aug_seq (List[BaseAugmentation] | None, optional): List of augmentations to apply. Defaults to None.
        """
        super().__init__(x, y, img_patch_path, wsi_id, barcode)
        self.gene_expression = gene_expression  # dict of gene id, float
        self.aug_seq = aug_seq

    def validate_TrainingDataPoint(self, adapter_name: str) -> bool:
        """
        Validate the data point for consistency and correctness.
        
        Checks for presence of required attributes and appropriate data types.
        Issues warnings or raises errors for invalid configurations.
        
        Args:
            adapter_name (str): Name of the adapter providing this data point, for error reporting.
            
        Returns:
            bool: True if the data point is valid.
            
        Raises:
            ValueError: If the image path does not exist or gene expression is not in the expected format.
        """
        # Validate common attributes
        self._validate_common_attributes(adapter_name)
        
        # Validate training-specific attributes
        if not isinstance(self.gene_expression, dict): #type: ignore
            raise ValueError(ErrorMessage.adapter_not_returning_gene_expression(adapter_name, type(self.gene_expression)))
        
        return True


class PredictionDataPoint(BaseDataPoint):
    """
    Represents a data point for making predictions in spatial transcriptomics.
    
    Similar to TrainingDataPoint but without gene expression data, typically used for
    generating predictions from images.
    
    Attributes:
        x (int): X-coordinate in the spatial reference frame.
        y (int): Y-coordinate in the spatial reference frame.
        img_patch_path (str): Path to the image patch associated with this data point.
        wsi_id (Optional[str]): Whole slide image ID, if applicable.
        barcode (Optional[str]): Unique barcode for this data point, if applicable.
    """
    
    def __init__(self, x: Optional[int], y: Optional[int], img_patch_path: str, 
                 wsi_id: Optional[str] = None, 
                 barcode: Optional[str] = None,
                 ) -> None:
        """
        Initialize a PredictionDataPoint instance.
        
        Args:
            x (Optional[int]): X-coordinate in the spatial reference frame.
            y (Optional[int]): Y-coordinate in the spatial reference frame.
            img_patch_path (str): Path to the image patch associated with this data point.
            wsi_id (Optional[str], optional): Whole slide image ID. Defaults to None.
            barcode (Optional[str], optional): Unique barcode for this data point. Defaults to None.
        """
        super().__init__(x, y, img_patch_path, wsi_id, barcode)

    def validate_TrainingDataPoint(self, adapter_name: str) -> bool:
        """
        Validate the prediction data point for consistency and correctness.
        
        Checks for presence of required attributes and appropriate data types.
        Issues warnings or raises errors for invalid configurations.
        
        Args:
            adapter_name (str): Name of the adapter providing this data point, for error reporting.
            
        Returns:
            bool: True if the data point is valid.

        Raises:
            ValueError: If the image path does not exist.
        """
        # Validate common attributes
        self._validate_common_attributes(adapter_name)
        
        return True