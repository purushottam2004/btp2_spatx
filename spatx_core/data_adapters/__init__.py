'''
The Data Adapter classes, are responsible to convert the unorganised or randomly downloaded data from some website, into Standard Data Class Objects
A Data Adapter class should be created when, a new data source is to be incorporated into the ecosystem.

While doing training of the model, simply use this classes to tell where is the data located to the model, and give this as input to the Trainer Classes
'''
from .breast_data_adapters import (
    BreastTrainingDataAdapter, BreastPredictionDataAdapter,
)

from .hest_data_adapter import (
    HestTrainingDataAdapter, HestPredictionDataAdapter,
)

from .base_data_adapter import BaseTrainingDataAdapter, BasePredictionDataAdapter

__all__ = [
    'BreastTrainingDataAdapter',
    'BreastPredictionDataAdapter',
    'HestTrainingDataAdapter',
    'HestPredictionDataAdapter',
    'BaseTrainingDataAdapter',
    'BasePredictionDataAdapter',
]