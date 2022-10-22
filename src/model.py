import numpy as np
from scipy.sparse import csr_matrix
from pypardiso import spsolve
# from scipy.sparse.linalg import spsolve

from src.utils import get_flat_index


def solve_equation(image: np.ndarray, mask: np.ndarray, gradient_img: np.ndarray = None) -> np.ndarray:
    """
    Solves the equation of the given image.

    Args:
        image: The image to inpaint
        mask: The mask with the region to inpaint

    Returns:
        The inpainted image.
    """
    ni, nj = image.shape[:2]
    mask_ext = np.zeros((mask.shape[0]+2, mask.shape[1]+2))
    mask_ext[1:-1, 1:-1] = mask
    image_ext = np.zeros((image.shape[0]+2, image.shape[1]+2))
    image_ext[1:-1, 1:-1] = image

    if gradient_img is not None:
        gradient_img_ext = np.zeros((gradient_img.shape[0]+2, gradient_img.shape[1]+2))
        gradient_img_ext[1:-1, 1:-1] = gradient_img
    else:
        gradient_img_ext = None

    n_pixels = (ni+2) * (nj+2)
    b = np.zeros((n_pixels, ), dtype=np.float64)

    idx_Ai = list()
    idx_Aj = list()
    a_ij = list()

    # North Side boundary conditions
    j = 0
    for i in range(ni+2):
        idx = get_flat_index(i, j, nj+2)

        idx_Ai.append(idx)
        idx_Aj.append(idx)
        a_ij.append(1)

        idx_Ai.append(idx)
        idx_Aj.append(idx+1)
        a_ij.append(-1)

        b[idx] = 0

    # South Side boundary conditions
    j = nj+1
    for i in range(ni+2):
        idx = get_flat_index(i, j, nj+2)

        idx_Ai.append(idx)
        idx_Aj.append(idx)
        a_ij.append(1)

        idx_Ai.append(idx)
        idx_Aj.append(idx-1)
        a_ij.append(-1)

        b[idx] = 0

    # West Side boundary conditions
    i = 0
    for j in range(nj+2):
        idx = get_flat_index(i, j, nj+2)
        idx_next = get_flat_index(i+1, j, nj+2)

        idx_Ai.append(idx)
        idx_Aj.append(idx)
        a_ij.append(1)

        idx_Ai.append(idx)
        idx_Aj.append(idx_next)
        a_ij.append(-1)

        b[idx] = 0

    # East Side boundary conditions
    i = ni+1
    for j in range(nj+2):
        idx = get_flat_index(i, j, nj+2)
        idx_prev = get_flat_index(i-1, j, nj+2)

        idx_Ai.append(idx)
        idx_Aj.append(idx)
        a_ij.append(1)

        idx_Ai.append(idx)
        idx_Aj.append(idx_prev)
        a_ij.append(-1)

        b[idx] = 0

    # Image points
    for i in range(1, ni+1):
        for j in range(1, nj+1):
            idx = get_flat_index(i, j, nj+2)

            if mask_ext[i, j]:
                # If the pixel falls in region B
                # 4V(i,j)
                idx_Ai.append(idx)
                idx_Aj.append(idx)
                a_ij.append(4)
                # -V(i-1,j)
                col = idx-1
                idx_Ai.append(idx)
                idx_Aj.append(col)
                a_ij.append(-1)
                # -V(i+1,j)
                col = idx+1
                idx_Ai.append(idx)
                idx_Aj.append(col)
                a_ij.append(-1)
                # -V(i,j+1)
                col = get_flat_index(i+1, j, nj+2)
                idx_Ai.append(idx)
                idx_Aj.append(col)
                a_ij.append(-1)
                # -V(i,j-1)
                col = get_flat_index(i-1, j, nj+2)
                idx_Ai.append(idx)
                idx_Aj.append(col)
                a_ij.append(-1)

                if gradient_img_ext is None:
                    b[idx] = 0
                else:
                    b[idx] = gradient_img_ext[i, j]
            else:
                # If the pixel falls in region A
                idx_Ai.append(idx)
                idx_Aj.append(idx)
                a_ij.append(1)
                b[idx] = image_ext[i, j]

    A = csr_matrix((a_ij, (idx_Ai, idx_Aj)), shape=(
        n_pixels, n_pixels), dtype=np.float64)

    # Solve equation
    x = spsolve(A, b)
    u_ext = np.reshape(x, (image.shape[0]+2, image.shape[1]+2))[1:-1, 1:-1]
    return u_ext