import numpy as np
import cv2
from scipy.sparse import csr_matrix


def inpaint_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    A=sparse(idx_Ai, idx_Aj, a_ij, ???, ???); %??? and ???? is the size of matrix A
    x=mldivide(A,b); u_ext=reshape(x, ni+2, nj+2);

    %Inner points
    for j=2:nj+1
        for i=2:ni+1
            %from image matrix (i,j) coordinates to vectorial (p) coordinate
            p = (j-1)*(ni+2)+i;

            if (dom2Inp_ext(i,j)==1) %If we have to inpaint this pixel
                %Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and
                %vector b
                %TO COMPLETE 
    """
    idx_Ai, idx_Aj = (mask == 0).nonzero()
    a_ij = image[idx_Ai, idx_Aj]
    ab_ij = a_ij[:, 0].ravel()
    ag_ij = a_ij[:, 1].ravel()
    ar_ij = a_ij[:, 2].ravel()
    # A = csr_matrix(a_ij, (idx_Ai, idx_Aj))
    return np.zeros(0)