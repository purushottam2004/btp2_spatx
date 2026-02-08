import random
from typing import List, Type

from PIL.ImageFile import ImageFile
import torchvision.transforms.functional as F #type: ignore
from torch import Tensor

from spatx_core.augmentation.augmentation import BaseAugmentation

class _BaseD8Augmentation(BaseAugmentation):
    name  = "_BaseD8Augmentation"

class Identity(_BaseD8Augmentation):
    name = "Identity"
    def apply(self, image : ImageFile | Tensor) -> ImageFile | Tensor:
        return image

class Rotate90(_BaseD8Augmentation):
    name = "Rotate90"
    def apply(self, image : ImageFile | Tensor) -> ImageFile | Tensor:
        if isinstance(image, Tensor):           
            return F.rotate(image, 90)
        else:
            return F.rotate(F.to_tensor(image), 90)


class Rotate180(_BaseD8Augmentation):
    name = "Rotate180"
    def apply(self, image : ImageFile | Tensor) -> ImageFile | Tensor:
        if isinstance(image, Tensor):
            return F.rotate(image, 180)
        else:   
            return F.rotate(F.to_tensor(image), 180)

class Rotate270(_BaseD8Augmentation):
    name = "Rotate270"
    def apply(self, image : ImageFile | Tensor) -> ImageFile | Tensor:
        if isinstance(image, Tensor):
            return F.rotate(image, 270)
        else:   
            return F.rotate(F.to_tensor(image), 270)

class FlipHorizontal(_BaseD8Augmentation):
    name = "FlipHorizontal"
    def apply(self, image : ImageFile | Tensor) -> ImageFile | Tensor:
        if isinstance(image, Tensor):
            return F.hflip(image)
        else:   
            return F.hflip(F.to_tensor(image))

class FlipVertical(_BaseD8Augmentation):
    name = "FlipVertical"
    def apply(self, image : ImageFile | Tensor) -> ImageFile | Tensor:
        if isinstance(image, Tensor):
            return F.vflip(image)
        else:   
            return F.vflip(F.to_tensor(image))

class FlipDiagonal1(_BaseD8Augmentation):
    name = "FlipDiagonal1"
    def apply(self, image : ImageFile | Tensor) -> ImageFile | Tensor:
        if isinstance(image, Tensor):
            return F.hflip(F.rotate(image, 90))
        else:
            return F.hflip(F.rotate(F.to_tensor(image), 90))

class FlipDiagonal2(_BaseD8Augmentation):
    name = "FlipDiagonal2"
    def apply(self, image : ImageFile | Tensor) -> ImageFile | Tensor:
        if isinstance(image, Tensor):
            return F.hflip(F.rotate(image, 270))
        else:
            return F.hflip(F.rotate(F.to_tensor(image), 270))

class RandomD8(_BaseD8Augmentation):
    name = "RandomD8"
    def apply(self, image : ImageFile | Tensor) -> ImageFile | Tensor:
        ops : List[Type[BaseAugmentation]]  = [
            Identity,
            Rotate90,
            Rotate180,
            Rotate270,
            FlipHorizontal,
            FlipVertical,
            FlipDiagonal1,
            FlipDiagonal2,
        ]
        op = random.choice(ops)
        if isinstance(image, Tensor):
            return op().apply(image)
        else:
            return (op().apply(image))
