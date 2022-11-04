import numpy as np


def get_flat_index(row: int, col: int, cols: int) -> int:
    return row * (cols) + col


def fast_laplacian(image):
    im1 = np.roll(image, -1, 0)
    im2 = np.roll(image, 1, 0)
    im3 = np.roll(image, -1, 1)
    im4 = np.roll(image, 1, 1)
    dx = 4*image - (im1 + im2 + im3 + im4)
    dx[0, :] = 0
    dx[-1, :] = 0
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


def init_surface_cone(img, radius=1,) -> np.ndarray:
    npixels = img.shape[0] * img.shape[1]
    xcoords = np.arange(npixels) // img.shape[1]
    xcoords -= max(xcoords) // 2  # center
    ycoords = np.arange(npixels) // img.shape[0]
    ycoords -= max(ycoords) // 2

    cone = radius - (xcoords.reshape(img.shape)**2 + ycoords.reshape((img.shape[1], img.shape[0])).T**2)**.5
    # Normalization
    return 255 * (cone - cone.min()) / (cone.max() - cone.min())


def init_surface_sine(img, freq=3) -> np.ndarray:
    npixels = img.shape[0] * img.shape[1]
    xcoords = np.arange(npixels) // img.shape[1]
    ycoords = np.arange(npixels) // img.shape[0]

    waves = np.sin(freq * xcoords.reshape(img.shape)) + np.sin(freq * ycoords.reshape((img.shape[1], img.shape[0])).T)

    return 255 * (waves - waves.min()) / (waves.max() - waves.min())


def init_surface_xavier(img, ) -> np.ndarray:
    x, y = img.shape
    scale = 1/max(1., (x+y)/2.)
    limit = (3.0 * scale) ** .5
    weights = np.random.uniform(-limit, limit, size=(x, y))

    return 255 * (weights - weights.min()) / (weights.max() - weights.min())


def init_surface_random_normal(img, ) -> np.ndarray:
    normal = np.random.normal(0, 1, size=img.shape)
    return 255 * (normal - normal.min()) / (normal.max() - normal.min())


PHI_INIT_FUNC = {
    "cone": init_surface_cone,
    "sine": init_surface_sine, 
    "normal": init_surface_random_normal,
    "xavier": init_surface_xavier,
}