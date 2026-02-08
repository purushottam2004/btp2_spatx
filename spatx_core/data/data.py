"""
Module for standardizing data access in spatial transcriptomics analysis.

This module provides classes for working with spatial transcriptomics data,
ensuring a consistent interface for accessing data points regardless of the
underlying data source.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_adapters.base_data_adapter import BaseTrainingDataAdapter, BasePredictionDataAdapter
from .data_point import TrainingDataPoint, PredictionDataPoint

class TrainingData:
    """
    Standard data class for spatial transcriptomics data.
    
    This class standardizes the data format of spatial transcriptomics data
    for use with models. It acts as a wrapper around a data adapter,
    ensuring that data is provided in a consistent format.
    
    Classes which inherit from torch.utils.data.Dataset can use this class
    to retrieve data in a standard format and provide it to models.
    
    The valid data format expected from the Adapter is:
        - length: an integer
        - data point dictionary:
            - image patch path
            - x coordinate (optional)
            - y coordinate (optional)
            - gene expression dictionary mapping gene IDs to expression levels
    
    Attributes:
        adapter (BaseTrainingDataAdapter): The data adapter providing the data.
    """
    
    def __init__(self, adapter: BaseTrainingDataAdapter) -> None:
        """
        Initialize a Data instance with a data adapter.
        
        Args:
            adapter: A data adapter that provides access to the underlying data.
            
        Raises:
            ValueError: If the adapter does not return data in the expected format.
        """
        self.adapter = adapter
        if not self._validate_adapter():
            raise ValueError(f"The adapter: {self.adapter.name} is not returning a standard data point")
    
    def __getitem__(self, idx: int) -> TrainingDataPoint:
        """
        Get a data point by index.
        
        Args:
            idx: Index of the data point to retrieve.
            
        Returns:
            A Training DataPoint instance.
        """
        # Delegate to adapter for actual data loading
        return self.adapter[idx]
    
    def __len__(self) -> int:
        """
        Get the number of data points available.
        
        Returns:
            Number of available data points.
        """
        return len(self.adapter)
    
    def _validate_adapter(self) -> bool:
        """
        Check whether the adapter returns data in the standardized format.
        
        Returns:
            True if the adapter returns valid data points, False otherwise.
            
        Raises:
            RuntimeError: If the adapter provides zero data points.
        """
        if len(self.adapter) > 0:
            if isinstance(self.adapter[0], TrainingDataPoint): #type: ignore
                return self.adapter[0].validate_TrainingDataPoint(adapter_name=self.adapter.name)
            return False
        raise RuntimeError(f"The Current Data adapter: {self.adapter.name} is providing zero data length")

        
class PredictionData:
    """
    Standard data class for prediction data in spatial transcriptomics.
    
    This class standardizes the data format for prediction purposes,
    ensuring a consistent interface for accessing prediction data points.
    
    Dataset classes which inherit from torch.utils.data.Dataset can use 
    this class to retrieve data in a standard format.
    
    The valid data format expected from the Adapter is:
        - length: an integer
        - prediction data point:
            - image patch path
            - x coordinate (optional)
            - y coordinate (optional)
    
    Attributes:
        adapter (BaseTrainingDataAdapter): The data adapter providing the prediction data.
    """
    
    def __init__(self, adapter: BasePredictionDataAdapter) -> None:
        """
        Initialize a PredictionData instance with a data adapter.
        
        Args:
            adapter: A data adapter that provides access to the underlying prediction data.
            
        Raises:
            ValueError: If the adapter does not return data in the expected format.
        """
        self.adapter = adapter
        if not self._validate_adapter():
            raise ValueError(f"The adapter: {self.adapter.name} is not returning a standard prediction data point")
    
    def __getitem__(self, idx: int) -> PredictionDataPoint:
        """
        Get a prediction data point by index.
        
        Args:
            idx: Index of the prediction data point to retrieve.
            
        Returns:
            A PredictionDataPoint instance.
        """
        # Delegate to adapter for actual data loading
        return self.adapter[idx]
    
    def __len__(self) -> int:
        """
        Get the number of prediction data points available.
        
        Returns:
            Number of available prediction data points.
        """
        return len(self.adapter)
    
    def _validate_adapter(self) -> bool:
        """
        Check whether the adapter returns prediction data in the standardized format.
        
        Returns:
            True if the adapter returns valid prediction data points, False otherwise.
            
        Raises:
            RuntimeError: If the adapter provides zero prediction data points.
        """
        if len(self.adapter) > 0:
            if isinstance(self.adapter[0], PredictionDataPoint): #type: ignore
                return self.adapter[0].validate_TrainingDataPoint(adapter_name=self.adapter.name)
            return False
        raise RuntimeError(f"The Current Prediction Data adapter: {self.adapter.name} is providing zero data length")