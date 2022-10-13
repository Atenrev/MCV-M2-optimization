import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.model import get_laplacian
from src.dataset import Dataset


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Steganography trainer parser')
    parser.add_argument('--images_dir', type=str, default='./dataset/images',
                        help='location of the dataset')
    parser.add_argument('--masks_dir', type=str, default='./dataset/masks',
                        help='location of the dataset')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='location of the dataset')
    args = parser.parse_args()
    return args


def provisional_laplacian(image, mask): # This should be a laplacian, it's not tho
    im1 = np.roll(image, -1, 0)
    im2 = np.roll(image, 1, 0)
    im3 = np.roll(image, -1, 1)
    im4 = np.roll(image, 1, 1)
    dx = 4*image - (im1 + im2 + im3 + im4)
    dx[~mask] = 1
    return dx


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = Dataset(args.images_dir, args.masks_dir)

    max_iter = 1
    alpha = 1.0
    theta = 1.0
    conv = 0.01

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

            laplacian = get_laplacian(u, mask)
            u = laplacian

            if len(image_channels) > 2:
                V[:,:,c] = u * 255
            else:
                V = u * 255
        
        cv2.imwrite(os.path.join(args.output_dir, f"{sample.name}.jpg"), V.astype(np.uint8))


if __name__ == "__main__":
    args = __parse_args()
    main(args)