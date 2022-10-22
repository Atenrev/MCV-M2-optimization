import numpy as np


def get_flat_index(row: int, col: int, cols:int) -> int:
    return row * (cols) + col


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