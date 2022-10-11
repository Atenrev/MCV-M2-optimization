import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.model import inpaint_image
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


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = Dataset(args.images_dir, args.masks_dir)

    max_iter = 50
    alpha = 1e-2
    conv = 0.01

    for n, sample in enumerate(dataset):
        image = sample.image
        mask = sample.mask

        # TODO: Do this for each channel
        if len(image.shape) > 2:
            image = np.mean(image, axis=-1).squeeze()

        u = np.zeros_like(image)
    
        for _ in tqdm(range(max_iter)):
            u_new = inpaint_image(u, mask)
            lagrange = 0.001 * (u-image) - u_new
            u = u - alpha * lagrange
        plt.imshow(u, cmap = 'gray')
        plt.show()
            
        cv2.imwrite(os.path.join(args.output_dir, f"{sample.name}.jpg"), u)


if __name__ == "__main__":
    args = __parse_args()
    main(args)