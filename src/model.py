import numpy as np
from scipy.sparse import csr_matrix
#from pypardiso import spsolve
from scipy.sparse.linalg import spsolve

from src.utils import get_flat_index


def get_laplacian(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    A=sparse(idx_Ai, idx_Aj, a_ij, ???, ???); %??? and ???? is the size of matrix A
    x=mldivide(A,b); 
    u_ext=reshape(x, ni+2, nj+2);

    %Inner points
    for j=2:nj+1
        for i=2:ni+1
            %from image matrix (i,j) coordinates to vectorial (p) coordinate
            p = (j-1)*(ni+2)+i;

            if (dom2Inp_ext(i,j)==1) %If we have to inpaint this pixel
                %Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and vector b
                %TO COMPLETE 
    """
    ni, nj = image.shape[:2]
    mask_ext = np.zeros((mask.shape[0]+2, mask.shape[1]+2))
    mask_ext[1:-1, 1:-1] = mask
    image_ext = np.zeros((image.shape[0]+2, image.shape[1]+2))
    image_ext[1:-1, 1:-1] = image

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

    for i in range(1, ni+1):
        for j in range(1, nj+1):
            idx = get_flat_index(i, j, nj+2)
            
            if mask_ext[i, j]:
                # 4𝑉 𝑥 𝑦
                idx_Ai.append(idx)
                idx_Aj.append(idx)
                a_ij.append(4)
                #  − (𝑉 𝑥 − 1, 𝑦 + 𝑉 𝑥 − 1, 𝑦 + 𝑉 𝑥, 𝑦 − 1 + 𝑉 𝑥, 𝑦 + 1 )
                col = idx-1
                idx_Ai.append(idx)
                idx_Aj.append(col)
                a_ij.append(-1)

                col = idx+1
                idx_Ai.append(idx)
                idx_Aj.append(col)
                a_ij.append(-1)

                col = get_flat_index(i+1, j, nj+2)
                idx_Ai.append(idx)
                idx_Aj.append(col)
                a_ij.append(-1)

                col = get_flat_index(i-1, j, nj+2)
                idx_Ai.append(idx)
                idx_Aj.append(col)
                a_ij.append(-1)
                b[idx] = 0
            else:
                idx_Ai.append(idx)
                idx_Aj.append(idx)
                a_ij.append(1)
                b[idx] = image_ext[i, j]

    A = csr_matrix((a_ij, (idx_Ai, idx_Aj)), shape=(n_pixels, n_pixels), dtype=np.float64)

    # Solve equation
    x = spsolve(A, b)
    u_ext = np.reshape(x, (image.shape[0]+2, image.shape[1]+2))[1:-1, 1:-1]
    return u_ext