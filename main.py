import argparse
import os

import cv2

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

    for sample in dataset:
        image = sample.image
        mask = sample.mask
        image_inpainted = inpaint_image(image, mask)
        cv2.imwrite(os.path.join(args.output_dir, f"{sample.name}.jpg"), image_inpainted)


if __name__ == "__main__":
    args = __parse_args()
    main(args)