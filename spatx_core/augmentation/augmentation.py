from abc import ABC, abstractmethod

from PIL.ImageFile import ImageFile
from torch import Tensor

class BaseAugmentation(ABC):
    name : str = "BaseAugmentation"

    @abstractmethod
    def apply(self, image : ImageFile | Tensor) -> ImageFile | Tensor:
        pass
