import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.model import solve_equation, poisson_edit
from src.dataset import DatasetInpainting, DatasetPoissonEdit


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Steganography trainer parser')
    parser.add_argument('--images_dir', type=str, default='./dataset_w2/images',
                        help='location of the dataset')
    parser.add_argument('--masks_dir', type=str, default='./dataset_w2/masks',
                        help='location of the dataset')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='location of the dataset')
    parser.add_argument('--task', type=str, default='poisson_edit',
                        help='task')
    args = parser.parse_args()
    return args


def fast_laplacian(image):
    im1 = np.roll(image, -1, 0)
    im2 = np.roll(image, 1, 0)
    im3 = np.roll(image, -1, 1)
    im4 = np.roll(image, 1, 1)
    dx = 4*image - (im1 + im2 + im3 + im4)
    return dx


def derivative(image): 
    im1 = np.roll(image, -1, 0)
    im2 = np.roll(image, 1, 0)
    im3 = np.roll(image, -1, 1)
    im4 = np.roll(image, 1, 1)
    dx = np.sqrt(np.square(im1-im2) + np.square(im3-im4))
    return dx


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


def do_poisson_edit(args: argparse.Namespace):
    dataset = DatasetPoissonEdit(args.images_dir, args.masks_dir)

    for n, sample in tqdm(enumerate(dataset)):
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

        # Mixing gradient, move to function
        p = 0.5
        gradient = p * gradient_src + (1-p) * gradient_dst

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

            u = poisson_edit(u, dst_mask, grad)

            if len(image_channels) > 2:
                V[:,:,c] = u * 255
            else:
                V = u * 255
        
        V -= np.min(V)
        V = V / np.max(V) * 255
        cv2.imwrite(os.path.join(args.output_dir, f"{sample.name}.jpg"), V.astype(np.uint8))


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.task == "inpaint":
        do_inpainting(args)
    elif args.task == "poisson_edit":
        do_poisson_edit(args)


if __name__ == "__main__":
    args = __parse_args()
    main(args)