from __future__ import annotations
import os
import numpy as np
import cv2
from typing import List
from glob import glob

from dataclasses import dataclass


@dataclass
class Sample:
    id: int
    name: str
    mask: np.ndarray
    image: np.ndarray


class Dataset:
    def __init__(self, path_images: str, path_masks: str, name: str = "default") -> None:
        self.name = name
        mask_paths = sorted(glob(os.path.join(path_masks, "*.png"))) + sorted(glob(os.path.join(path_masks, "*.jpg")))
        image_paths = sorted(glob(os.path.join(path_images, "*.png"))) + sorted(glob(os.path.join(path_images, "*.jpg")))

        assert len(mask_paths) > 0, f"No masks were found on {path_masks}."
        assert len(image_paths) > 0, f"No images were found on {path_images}."

        self.samples: List[Sample] = []

        for i, (mask_path, image_path) in enumerate(zip(mask_paths, image_paths)):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) != 0
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            basename = os.path.basename(mask_path).split('.')[0]
            self.samples.append(Sample(
                id=i,
                name=basename,
                mask=mask,
                image=image
            ))

    def size(self) -> int:
        return len(self.samples)

    def get_item(self, id: int) -> Sample:
        return self.__getitem__(id)

    def __iter__(self):
        return (self.__getitem__(id) for id in range(self.size()))

    def __getitem__(self, id: int) -> Sample:
        return self.samples[id]