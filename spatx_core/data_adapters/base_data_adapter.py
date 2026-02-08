"""
Module defining base interfaces for data adapters in the system.

Data adapters are responsible for loading and providing access to datasets
in a standardized format for the rest of the system.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from ..data.data_point import TrainingDataPoint, PredictionDataPoint

class BaseTrainingDataAdapter(ABC):
    """
    Abstract base class for all training data adapters in the system.
    
    Training data adapters are responsible for loading training data from various sources
    and providing a standardized interface for accessing training data points that include
    gene expression values.
    
    Attributes:
        name (str): Name identifier for the adapter.
    """
    
    name = 'Unnamed Training Adapter'

    @abstractmethod
    def __init__(self, gene_ids: list[str], *args, **kwargs) -> None: #type: ignore
        """
        Initialize a training data adapter.
        
        Args:
            gene_ids: List of gene IDs that this adapter should provide data for.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> TrainingDataPoint:
        """
        Get a training data point by index.
        
        Should be implemented to return a training data point in a format that
        the standard Data class requires.
        
        Args:
            idx: Index of the training data point to retrieve.
            
        Returns:
            A TrainingDataPoint instance.
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Get the number of training data points available through this adapter.
        
        Should be implemented to return the length of training data in a format 
        that the standard Data class requires.
        
        Returns:
            Number of available training data points.
        """
        pass

class BasePredictionDataAdapter(ABC):
    """
    Abstract base class for all prediction data adapters in the system.
    
    Prediction data adapters are responsible for loading data for prediction
    from various sources and providing a standardized interface for accessing
    prediction data points (which do not include gene expression values).
    
    Attributes:
        name (str): Name identifier for the adapter.
    """
    
    name = 'Unnamed Prediction Data Adapter'

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None: # type: ignore
        """
        Initialize a prediction data adapter.
        
        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> PredictionDataPoint:
        """
        Get a prediction data point by index.
        
        Should be implemented to return a prediction data point in a format that
        the standard PredictionData class requires.
        
        Args:
            idx: Index of the prediction data point to retrieve.
            
        Returns:
            A PredictionDataPoint instance.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Get the number of prediction data points available through this adapter.
        
        Should be implemented to return the length of prediction data in a format 
        that the standard PredictionData class requires.
        
        Returns:
            Number of available prediction data points.
        """
        pass