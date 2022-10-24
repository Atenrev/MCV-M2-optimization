import numpy as np


def get_flat_index(row: int, col: int, cols:int) -> int:
    return row * (cols) + col


def fast_laplacian(image):
    im1 = np.roll(image, -1, 0)
    im2 = np.roll(image, 1, 0)
    im3 = np.roll(image, -1, 1)
    im4 = np.roll(image, 1, 1)
    dx = 4*image - (im1 + im2 + im3 + im4)
    dx[0, :] = 0
    dx[-1,:] = 0
    dx[0, :] = 0
    dx[:, -1] = 0
    return dx
    

def derivative(image): 
    G1_DiBwd = np.zeros_like(image)
    G1_DiBwd[1:, :] = image[1:, :] - image[:-1, :]

    G1_DiFwd = np.zeros_like(image)
    G1_DiFwd[:-1, :] = image[1:, :] - image[:-1, :]

    G1_DjBwd = np.zeros_like(image)
    G1_DjBwd[:, 1:] = image[:, 1:] - image[:, :-1]

    G1_DjFwd = np.zeros_like(image)
    G1_DjFwd[:, :-1] = image[:, 1:] - image[:, :-1]

    drivingGrad_i = G1_DiBwd - G1_DiFwd
    drivingGrad_j = G1_DjBwd - G1_DjFwd
    return drivingGrad_i + drivingGrad_j