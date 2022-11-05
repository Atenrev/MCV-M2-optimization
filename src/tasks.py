import argparse
import os
import cv2
import numpy as np
from typing import Callable
from tqdm import tqdm
from scipy.ndimage.morphology import distance_transform_edt

from src.model import (solve_equation, chan_vese_calc_color, chan_vese_dirac)
from src.dataset import (DatasetDefault, DatasetInpainting,
                         DatasetPoissonEdit, SamplePoissonEdit,)
from src.utils import (derivative, fast_laplacian, PHI_INIT_FUNC,
                       di_fwd, di_bwd, di_cent, dj_fwd, dj_bwd, dj_cent)


def do_inpainting(args: argparse.Namespace):
    dataset = DatasetInpainting(args.images_dir, args.masks_dir)

    for n, sample in tqdm(enumerate(dataset), total=dataset.size()):
        image = sample.image
        mask = sample.mask

        image -= np.min(image)
        image = image / np.max(image)

        if len(image.shape) > 2:
            image_channels = [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
        else:
            image_channels = image

        V = np.zeros_like(image)

        for c, img in enumerate(image_channels):
            u = np.array(img)
            img[mask] = 0

            u = solve_equation(u, mask)

            if len(image_channels) > 2:
                V[:, :, c] = u * 255
            else:
                V = u * 255

        cv2.imwrite(os.path.join(args.output_dir,
                    f"{sample.name}.jpg"), V.astype(np.uint8))


def get_importing_gradients(sample: SamplePoissonEdit) -> np.ndarray:
    dst_image = sample.dst_image
    dst_mask = sample.dst_mask
    src_image = sample.src_image
    src_mask = sample.src_mask

    src_gradient = fast_laplacian(src_image)
    gradient = np.zeros_like(dst_image)
    gradient[dst_mask] = src_gradient[src_mask]
    return gradient


def get_mixed_gradients(sample: SamplePoissonEdit) -> np.ndarray:
    dst_image = sample.dst_image
    dst_mask = sample.dst_mask
    src_image = sample.src_image
    src_mask = sample.src_mask

    # gradient for source image
    G1_DiBwd_src = np.zeros_like(src_image)
    G1_DiBwd_src[1:-1, :] = src_image[1:-1, :] - src_image[:-2, :]
    aux = np.zeros_like(dst_image)
    aux[dst_mask] = G1_DiBwd_src[src_mask]
    G1_DiBwd_src = aux

    G1_DiFwd_src = np.zeros_like(src_image)
    G1_DiFwd_src[1:-1, :] = src_image[1:-1, :] - src_image[2:, :]
    aux = np.zeros_like(dst_image)
    aux[dst_mask] = G1_DiFwd_src[src_mask]
    G1_DiFwd_src = aux

    G1_DjBwd_src = np.zeros_like(src_image)
    G1_DjBwd_src[:, 1:-1] = src_image[:, 1:-1] - src_image[:, :-2]
    aux = np.zeros_like(dst_image)
    aux[dst_mask] = G1_DjBwd_src[src_mask]
    G1_DjBwd_src = aux

    G1_DjFwd_src = np.zeros_like(src_image)
    G1_DjFwd_src[:, 1:-1] = src_image[:, 1:-1] - src_image[:, 2:]
    aux = np.zeros_like(dst_image)
    aux[dst_mask] = G1_DjFwd_src[src_mask]
    G1_DjFwd_src = aux

    # gradient for target image
    G1_DiBwd_dst = np.zeros_like(dst_image)
    G1_DiBwd_dst[1:-1, :] = dst_image[1:-1, :] - dst_image[:-2, :]
    G1_DiFwd_dst = np.zeros_like(dst_image)
    G1_DiFwd_dst[1:-1, :] = dst_image[1:-1, :] - dst_image[2:, :]
    G1_DjBwd_dst = np.zeros_like(dst_image)
    G1_DjBwd_dst[:, 1:-1] = dst_image[:, 1:-1] - dst_image[:, :-2]
    G1_DjFwd_dst = np.zeros_like(dst_image)
    G1_DjFwd_dst[:, 1:-1] = dst_image[:, 1:-1] - dst_image[:, 2:]

    mask = (np.abs(G1_DiBwd_src) - np.abs(G1_DiBwd_dst)) < 0
    G1_DiBwd_src[mask] = G1_DiBwd_dst[mask]
    mask = (np.abs(G1_DiFwd_src) - np.abs(G1_DiFwd_dst)) < 0
    G1_DiFwd_src[mask] = G1_DiFwd_dst[mask]
    mask = (np.abs(G1_DjBwd_src) - np.abs(G1_DjBwd_dst)) < 0
    G1_DjBwd_src[mask] = G1_DjBwd_dst[mask]
    mask = (np.abs(G1_DjFwd_src) - np.abs(G1_DjFwd_dst)) < 0
    G1_DjFwd_src[mask] = G1_DjFwd_dst[mask]

    return (G1_DiBwd_src + G1_DiFwd_src + G1_DjBwd_src + G1_DjFwd_src)


def get_weighted_gradients(sample: SamplePoissonEdit, a: float = 0.5) -> np.ndarray:
    dst_image = sample.dst_image
    dst_mask = sample.dst_mask
    dst_gradient = derivative(dst_image)

    src_image = sample.src_image
    src_mask = sample.src_mask
    src_gradient_in_src = derivative(src_image)
    src_gradient = np.zeros_like(dst_image)
    src_gradient[dst_mask] = src_gradient_in_src[src_mask]

    gradient = a * src_gradient + (1-a) * dst_gradient
    return gradient


def do_poisson_edit(args: argparse.Namespace, get_gradient: Callable, **kwargs):
    dataset = DatasetPoissonEdit(args.images_dir, args.masks_dir)

    for sample in tqdm(dataset):
        dst_image = sample.dst_image
        dst_mask = sample.dst_mask

        gradient = get_gradient(sample, **kwargs)

        if len(dst_image.shape) > 2:
            image_channels = [dst_image[:, :, 0],
                              dst_image[:, :, 1], dst_image[:, :, 2]]
            gradient_channels = [gradient[:, :, 0],
                                 gradient[:, :, 1], gradient[:, :, 2]]
        else:
            image_channels = [dst_image]
            gradient_channels = [gradient]

        V = np.zeros_like(dst_image)

        for c, (img, grad) in enumerate(zip(image_channels, gradient_channels)):
            u = np.array(img)
            img[dst_mask] = 0

            u = solve_equation(u, dst_mask, grad)

            if len(image_channels) > 2:
                V[:, :, c] = u
            else:
                V = u

        V = np.clip(V, 0, 255)
        cv2.imwrite(os.path.join(args.output_dir,
                    f"{sample.name}.jpg"), V.astype(np.uint8))


def do_chan_vese(args: argparse.Namespace, **kwargs):
    dataset = DatasetDefault(args.images_dir)
    dt = (10 ^ -1) / args.mu

    for sample in tqdm(dataset):
        image = cv2.cvtColor(sample.image, cv2.COLOR_BGR2GRAY)

        phi_0 = PHI_INIT_FUNC[args.phi_init](image)
        c1 = 255
        c2 = 0

        phi = phi_0
        dif = np.inf
        it = 0

        while dif > args.tol and it < args.max_iter:
            phi_old = phi.copy()

            c1 = chan_vese_calc_color(image, phi_old)
            c2 = chan_vese_calc_color(image, phi_old, negate=True)

            phi[0, :] = phi_old[1, :]
            phi[-1, :] = phi_old[-2, :]
            phi[:, 0] = phi_old[:, 1]
            phi[:, -1] = phi_old[:, -2]

            delta_phi = chan_vese_dirac(phi_old, args.ep_heaviside)

            phi_iFwd = di_fwd(phi_old)
            # phi_iBwd = di_bwd(phi_old)
            phi_jFwd = dj_fwd(phi_old)
            # phi_jBwd = dj_bwd(phi_old)
            phi_icent = di_cent(phi_old)
            phi_jcent = dj_cent(phi_old)
            # phi_icent = (phi_iFwd + phi_iBwd) / 2
            # phi_jcent = (phi_jFwd + phi_jBwd) / 2


            A = args.mu / np.sqrt(args.eta**2 + phi_iFwd**2 + phi_jcent**2)
            B = args.mu / np.sqrt(args.eta**2 + phi_jFwd**2 + phi_icent**2)

            phi[1:-1, 1:-1] = phi_old[1:-1, 1:-1] + dt * delta_phi[1:-1, 1:-1] \
                * (A[1:-1, 1:-1]*phi_old[2:, 1:-1] + A[:-2, 1:-1]*phi[:-2, 1:-1]
                   + B[1:-1, 1:-1]*phi_old[1:-1, 2:] + B[1:-1, :-2]*phi[1:-1, :-2]
                   - args.nu - args.lambda1 * (image[1:-1, 1:-1] - c1)**2
                   + args.lambda2 * (image[1:-1, 1:-1] - c2)**2
                   ) / (1.0 + dt*delta_phi[1:-1, 1:-1]
                        + A[1:-1, 1:-1] + A[:-2, 1:-1] + B[1:-1, 1:-1] + B[1:-1, :-2]
                        )

            if args.re_init > 0 and it > 0 and it % args.re_init == 0:
                indGT = phi >= 0;
                indLT = phi < 0;
                
                phi = distance_transform_edt(1-indLT) - distance_transform_edt(1-indGT);
                
                nor = min(np.abs(np.min(phi)), np.max(phi))
                phi = phi / nor;

            dif = np.mean(np.sum((phi - phi_old)**2))
            it += 1

        segmented = image.copy()
        segmented[phi>=0] = c1 
        segmented[phi<0] = c2
        cv2.imwrite(os.path.join(args.output_dir,
                    f"{sample.name}.jpg"), segmented.astype(np.uint8))
