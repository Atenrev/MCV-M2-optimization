from __future__ import annotations
import os
import numpy as np
import cv2
from typing import List
from glob import glob

from dataclasses import dataclass


@dataclass
class SampleDefault:
    id: int
    name: str
    image: np.ndarray


class DatasetDefault:
    def __init__(self, path_images: str, name: str = "default") -> None:
        self.name = name
        image_paths = sorted(glob(os.path.join(path_images, "*.png")) + glob(os.path.join(path_images, "*.jpg")))

        assert len(image_paths) > 0, f"No images were found on {path_images}."

        self.samples: List[SampleDefault] = []

        for i, image_path in enumerate(image_paths):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            basename = os.path.basename(image_path).split('.')[0]
            self.samples.append(SampleDefault(
                id=i,
                name=basename,
                image=image
            ))

    def size(self) -> int:
        return len(self.samples)

    def get_item(self, id: int) -> SampleDefault:
        return self.__getitem__(id)

    def __iter__(self):
        return (self.__getitem__(id) for id in range(self.size()))

    def __getitem__(self, id: int) -> SampleDefault:
        return self.samples[id]


@dataclass
class SampleInpainting:
    id: int
    name: str
    mask: np.ndarray
    image: np.ndarray


class DatasetInpainting:
    def __init__(self, path_images: str, path_masks: str, name: str = "default") -> None:
        self.name = name
        mask_paths = sorted(glob(os.path.join(path_masks, "*.png")) + glob(os.path.join(path_masks, "*.jpg")))
        image_paths = sorted(glob(os.path.join(path_images, "*.png")) + glob(os.path.join(path_images, "*.jpg")))

        assert len(mask_paths) > 0, f"No masks were found on {path_masks}."
        assert len(image_paths) > 0, f"No images were found on {path_images}."

        self.samples: List[SampleInpainting] = []

        for i, (mask_path, image_path) in enumerate(zip(mask_paths, image_paths)):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) != 0
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            basename = os.path.basename(mask_path).split('.')[0]
            self.samples.append(SampleInpainting(
                id=i,
                name=basename,
                mask=mask,
                image=image
            ))

    def size(self) -> int:
        return len(self.samples)

    def get_item(self, id: int) -> SampleInpainting:
        return self.__getitem__(id)

    def __iter__(self):
        return (self.__getitem__(id) for id in range(self.size()))

    def __getitem__(self, id: int) -> SampleInpainting:
        return self.samples[id]


@dataclass
class SamplePoissonEdit:
    id: int
    name: str
    src_mask: np.ndarray
    src_image: np.ndarray
    dst_mask: np.ndarray
    dst_image: np.ndarray


class DatasetPoissonEdit:
    def __init__(self, path_images: str, path_masks: str, name: str = "default") -> None:
        self.name = name
        mask_paths = sorted(glob(os.path.join(path_masks, "*")))
        image_paths = sorted(glob(os.path.join(path_images, "*")))

        assert len(mask_paths) > 0, f"No masks were found on {path_masks}."
        assert len(image_paths) > 0, f"No images were found on {path_images}."

        self.samples: List[SamplePoissonEdit] = []

        for i, (mask_path, image_path) in enumerate(zip(mask_paths, image_paths)):
            src_image = cv2.imread(os.path.join(image_path, "src.png"), cv2.IMREAD_COLOR).astype(np.float64)
            # src_image = src_image / 255
            src_mask = cv2.imread(os.path.join(mask_path, "src.png"), cv2.IMREAD_GRAYSCALE) != 0

            dst_image = cv2.imread(os.path.join(image_path, "dst.png"), cv2.IMREAD_COLOR).astype(np.float64)
            # dst_image = dst_image / 255
            dst_mask = cv2.imread(os.path.join(mask_path, "dst.png"), cv2.IMREAD_GRAYSCALE) != 0

            self.samples.append(SamplePoissonEdit(
                id=i,
                name=str(i),
                src_mask=src_mask,
                src_image=src_image,
                dst_mask=dst_mask,
                dst_image=dst_image
            ))

    def __len__(self) -> int:
        return len(self.samples)

    def get_item(self, id: int) -> SamplePoissonEdit:
        return self.__getitem__(id)

    def __iter__(self):
        return (self.__getitem__(id) for id in range(len(self)))

    def __getitem__(self, id: int) -> SamplePoissonEdit:
        return self.samples[id]


        