import argparse

from src.model import inpaint_image
from src.dataset import Dataset


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Steganography trainer parser')
    parser.add_argument('--images_dir', type=str, default='./dataset/images',
                        help='location of the dataset')
    parser.add_argument('--masks_dir', type=str, default='./dataset/masks',
                        help='location of the dataset')
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    dataset = Dataset(args.images_dir, args.masks_dir)

    for sample in dataset:
        image = sample.image
        mask = sample.mask
        inpaint_image(image, mask)


if __name__ == "__main__":
    args = __parse_args()
    main(args)