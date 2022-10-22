import argparse
import os
from typing import Callable
import cv2
import numpy as np
from tqdm import tqdm

from src.model import solve_equation
from src.dataset import DatasetInpainting, DatasetPoissonEdit, SamplePoissonEdit
from src.utils import fast_laplacian


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
    dst_mask = sample.dst_mask
    src_image = sample.src_image
    src_image -= np.min(src_image)
    src_image = src_image / np.max(src_image)
    src_mask = sample.src_mask

    gradient = fast_laplacian(src_image)
    gradient[dst_mask == 1] = gradient[src_mask == 1]
    return gradient


def get_mixed_gradients(sample: SamplePoissonEdit) -> np.ndarray:
    dst_image = sample.dst_image
    dst_mask = sample.dst_mask

    dst_image -= np.min(dst_image)
    dst_image = dst_image / np.max(dst_image)
    gradient_dst = fast_laplacian(dst_image)

    src_image = sample.src_image
    src_mask = sample.src_mask
    
    src_image -= np.min(src_image)
    src_image = src_image / np.max(src_image)
    gradient_src = fast_laplacian(src_image)
    gradient_src[dst_mask == 1] = gradient_src[src_mask == 1]

    gradient = np.maximum(gradient_src, gradient_dst)
    return gradient


def get_weighted_gradients(sample: SamplePoissonEdit, a: float = 0.5) -> np.ndarray:
    dst_image = sample.dst_image
    dst_mask = sample.dst_mask

    dst_image -= np.min(dst_image)
    dst_image = dst_image / np.max(dst_image)
    gradient_dst = fast_laplacian(dst_image)

    src_image = sample.src_image
    src_image -= np.min(src_image)
    src_image = src_image / np.max(src_image)
    src_mask = sample.src_mask

    gradient_src = fast_laplacian(src_image)
    gradient_src[dst_mask == 1] = gradient_src[src_mask == 1]

    gradient = a * gradient_src + (1-a) * gradient_dst
    return gradient


def do_poisson_edit(args: argparse.Namespace, get_gradient: Callable, **kwargs):
    dataset = DatasetPoissonEdit(args.images_dir, args.masks_dir)

    for n, sample in tqdm(enumerate(dataset)):
        dst_image = sample.dst_image
        dst_mask = sample.dst_mask

        dst_image -= np.min(dst_image)
        dst_image = dst_image / np.max(dst_image)

        gradient = get_gradient(sample, **kwargs)

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
                V[:,:,c] = u * 255
            else:
                V = u * 255
        
        V -= np.min(V)
        V = V / np.max(V) * 255
        cv2.imwrite(os.path.join(args.output_dir, f"{sample.name}.jpg"), V.astype(np.uint8))