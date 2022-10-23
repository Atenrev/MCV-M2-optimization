import argparse
import os
from typing import Callable
import cv2
import numpy as np
from tqdm import tqdm

from src.model import solve_equation
from src.dataset import DatasetInpainting, DatasetPoissonEdit, SamplePoissonEdit
from src.utils import derivative, fast_laplacian


def do_inpainting(args: argparse.Namespace):
    dataset = DatasetInpainting(args.images_dir, args.masks_dir)

    for n, sample in tqdm(enumerate(dataset), total=dataset.size()):
        image = sample.image
        mask = sample.mask

        image -= np.min(image)
        image = image / np.max(image)

        if len(image.shape) > 2:
            image_channels = [image[:,:,0], image[:,:,1], image[:,:,2]]
        else:
            image_channels = image

        V = np.zeros_like(image)

        for c, img in enumerate(image_channels):
            u = np.array(img)
            img[mask] = 0

            u = solve_equation(u, mask)

            if len(image_channels) > 2:
                V[:,:,c] = u * 255
            else:
                V = u * 255
        
        cv2.imwrite(os.path.join(args.output_dir, f"{sample.name}.jpg"), V.astype(np.uint8))


def get_importing_gradients(sample: SamplePoissonEdit) -> np.ndarray:
    dst_image = sample.dst_image
    dst_mask = sample.dst_mask
    src_image = sample.src_image
    src_mask = sample.src_mask

    src_gradient = derivative(src_image)
    gradient = np.zeros_like(dst_image)
    gradient[dst_mask] = src_gradient[src_mask]
    return gradient


def get_mixed_gradients(sample: SamplePoissonEdit) -> np.ndarray:
    dst_image = sample.dst_image
    dst_mask = sample.dst_mask
    dst_gradient = derivative(dst_image)

    src_image = sample.src_image
    src_mask = sample.src_mask
    src_gradient_in_src = derivative(src_image)
    src_gradient = np.zeros_like(dst_image)
    src_gradient[dst_mask] = src_gradient_in_src[src_mask]

    mask = (src_gradient ** 2 - dst_gradient ** 2) < 0
    gradient = src_gradient
    gradient[mask] = dst_gradient[mask]
    return gradient 


def get_weighted_gradients(sample: SamplePoissonEdit, a: float = 0.5) -> np.ndarray:
    dst_image = sample.dst_image
    dst_mask = sample.dst_mask
    dst_gradient = derivative(dst_image)

    src_image = sample.src_image
    src_mask = sample.src_mask
    src_gradient_in_src = derivative(src_image)
    src_gradient = np.zeros_like(dst_image)
    src_gradient[dst_mask] = src_gradient_in_src[src_mask]

    gradient = a * src_gradient + (1-a) * dst_gradient
    return gradient


def do_poisson_edit(args: argparse.Namespace, get_gradient: Callable, **kwargs):
    dataset = DatasetPoissonEdit(args.images_dir, args.masks_dir)

    for n, sample in tqdm(enumerate(dataset)):
        dst_image = sample.dst_image
        dst_mask = sample.dst_mask

        gradient = get_gradient(sample, **kwargs)

        # gradient = gradient[1:-1, 1:-1]
        # dst_image = dst_image[1:-1, 1:-1]
        # dst_mask = dst_mask[1:-1, 1:-1]

        if len(dst_image.shape) > 2:
            image_channels = [dst_image[:,:,0], dst_image[:,:,1], dst_image[:,:,2]]
            gradient_channels = [gradient[:,:,0], gradient[:,:,1], gradient[:,:,2]]
        else:
            image_channels = [dst_image]
            gradient_channels = [gradient]

        V = np.zeros_like(dst_image)

        for c, (img, grad) in enumerate(zip(image_channels, gradient_channels)):
            u = np.array(img)
            img[dst_mask] = 0

            u = solve_equation(u, dst_mask, grad)

            if len(image_channels) > 2:
                V[:,:,c] = u # * 255
            else:
                V = u # * 255
                
        # V -= np.min(V)
        # V = V / np.max(V) * 255
        V = np.clip(V, 0, 255)
        cv2.imwrite(os.path.join(args.output_dir, f"{sample.name}.jpg"), V.astype(np.uint8))