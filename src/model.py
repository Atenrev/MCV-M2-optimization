import numpy as np
import cv2
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from src.utils import get_flat_index


def inpaint_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
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
    # TODO: We are not padding!
    image = np.mean(image, axis=-1).squeeze()
    V = np.zeros_like(image)
    m = image.shape[0]; n = image.shape[1]
    Am = An = m * n # (m+2) * (n+2)

    # Compute equation for A
    mask_flattened = mask.flatten()
    idx_Ai = idx_Aj = (mask_flattened == 0).nonzero()[0]
    idx_Ai = list(idx_Ai)
    idx_Aj = list(idx_Aj)
    a_ij = np.ones_like(idx_Ai).tolist()

    # Compute equation for B
    mask_i, mask_j = (mask).nonzero()

    for i, j in zip(mask_i, mask_j):
        row = get_flat_index(i, j, n)

        # 4𝑉 𝑥 𝑦
        idx_Ai.append(row)
        idx_Aj.append(row)
        a_ij.append(4)
        #  − (𝑉 𝑥 − 1, 𝑦 + 𝑉 𝑥 − 1, 𝑦 + 𝑉 𝑥, 𝑦 − 1 + 𝑉 𝑥, 𝑦 + 1 )
        col = row-1
        idx_Ai.append(row)
        idx_Aj.append(col)
        a_ij.append(-1)

        col = row+1
        idx_Ai.append(row)
        idx_Aj.append(col)
        a_ij.append(-1)

        col = get_flat_index(i, j-1, n)
        idx_Ai.append(row)
        idx_Aj.append(col)
        a_ij.append(-1)

        col = get_flat_index(i, j+1, n)
        idx_Ai.append(row)
        idx_Aj.append(col)
        a_ij.append(-1)

    A = csr_matrix((a_ij, (idx_Ai, idx_Aj)), shape=(Am, An))
    b = (image * ~mask).flatten()

    # Solve equation
    x = spsolve(A, b)
    u_ext = np.reshape(x, image.shape)
    return u_ext