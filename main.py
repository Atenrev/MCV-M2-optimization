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
    dx[~mask] = 0
    return dx

def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = Dataset(args.images_dir, args.masks_dir)

    max_iter = 200
    alpha = 1
    theta = 1
    conv = 0.01

    for n, sample in enumerate(dataset):
        image = sample.image
        mask = sample.mask

        # TODO: Do this for each channel
        if len(image.shape) > 2:
            image = np.mean(image, axis=-1).squeeze()

        #u = np.zeros_like(image)
        u = np.array(image)
        image[mask] = 0
    
        for _ in tqdm(range(max_iter)):
            
            laplacian = get_laplacian(u, mask)
            lagrange_gradient = theta * (u-image) - laplacian
            u = u - alpha * lagrange_gradient
        
        plt.imshow(u, cmap = 'gray')
        plt.savefig(os.path.join(args.output_dir, f"{sample.name}.jpg"))
        plt.clf()
        #cv2.imwrite(os.path.join(args.output_dir, f"{sample.name}.jpg"), u)


if __name__ == "__main__":
    args = __parse_args()
    main(args)