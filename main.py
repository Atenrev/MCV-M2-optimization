import argparse
import os

from src.tasks import (do_inpainting, do_poisson_edit, get_importing_gradients,
                       get_mixed_gradients, get_weighted_gradients, do_chan_vese)


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Steganography trainer parser')
    parser.add_argument('--images_dir', type=str, default='./dataset_w1/images',
                        help='location of the dataset')
    parser.add_argument('--masks_dir', type=str, default='./dataset_w2/masks',
                        help='location of the dataset')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='location of the dataset')
    parser.add_argument('--task', type=str, default='chan_vese',
                        help='task')
    # Poison edit
    parser.add_argument('--mg_apha', type=float, default=0.5,
                        help='alfa value used in weighted_gradients')
    # Chan vese segmentation
    parser.add_argument('--phi_init', type=str, default='xavier',
                        help='phi init function')
    parser.add_argument('--tol', type=float, default=0.1,
                        help='tolerance for the stopping criterium')
    parser.add_argument('--ep_heaviside', type=float, default=1.0,
                        help='epsilon for the regularized heaviside')
    parser.add_argument('--mu', type=float, default=1.0,
                        help='mu lenght parameter (regularizer term)')
    parser.add_argument('--nu', type=float, default=0.0,
                        help='nu area parameter (regularizer term)')
    parser.add_argument('--lambda1', type=float, default=1.0,
                        help='mu lenght parameter (regularizer term)')
    parser.add_argument('--lambda2', type=float, default=1.0,
                        help='nu area parameter (regularizer term)')
    parser.add_argument('--eta', type=float, default=1.0,
                        help='epsilon for the total variation regularization')
    parser.add_argument('--max_iter', type=float, default=1000,
                        help='maximum number of iterations')
    parser.add_argument('--re_init', type=float, default=100,
                        help='Iterations for reinitialization. 0 means no reinitializacion')
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    if args.task == "inpaint":
        do_inpainting(args)
    elif args.task == "importing_gradients":
        do_poisson_edit(args, get_importing_gradients)
    elif args.task == "mixed_gradients":
        do_poisson_edit(args, get_mixed_gradients)
    elif args.task == "weighted_gradients":
        do_poisson_edit(args, get_weighted_gradients, a=args.mg_apha)
    elif args.task == "chan_vese":
        do_chan_vese(args)


if __name__ == "__main__":
    args = __parse_args()
    main(args)
