import numpy as np


def get_flat_index(row: int, col: int, cols:int) -> int:
    return row * (cols) + col


def fast_laplacian(image):
    # im1 = np.roll(image, -1, 0)
    # im2 = np.roll(image, 1, 0)
    # im3 = np.roll(image, -1, 1)
    # im4 = np.roll(image, 1, 1)
    # dx = 4*image - (im1 + im2 + im3 + im4)
    # return dx
    G1_DiBwd = np.array(image)
    G1_DiBwd[1:, :] = image[1:, :] - image[:-1, :]

    G1_DiFwd = np.array(image)
    G1_DiFwd[:-1, :] = image[1:, :] - image[:-1, :]

    G1_DjBwd = np.array(image)
    G1_DjBwd[:, 1:] = image[:, 1:] - image[:, :-1]

    G1_DjFwd = np.array(image)
    G1_DjFwd[:, :-1] = image[:, 1:] - image[:, :-1]

    drivingGrad_i = G1_DiBwd - G1_DiFwd
    drivingGrad_j = G1_DjBwd - G1_DjFwd
    return drivingGrad_i + drivingGrad_j


def derivative(image): 
    im1 = np.roll(image, -1, 0)
    im2 = np.roll(image, 1, 0)
    im3 = np.roll(image, -1, 1)
    im4 = np.roll(image, 1, 1)
    dx = np.sqrt(np.square(im1-im2) + np.square(im3-im4))
    return dx