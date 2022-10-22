import argparse
import os

from src.tasks import do_inpainting, do_poisson_edit, get_importing_gradients, get_mixed_gradients, get_weighted_gradients


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Steganography trainer parser')
    parser.add_argument('--images_dir', type=str, default='./dataset_w2/images',
                        help='location of the dataset')
    parser.add_argument('--masks_dir', type=str, default='./dataset_w2/masks',
                        help='location of the dataset')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='location of the dataset')
    parser.add_argument('--task', type=str, default='mixed_gradients',
                        help='task')
    parser.add_argument('--mg_apha', type=float, default=0.5,
                        help='alfa value used in weighted_gradients')
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


if __name__ == "__main__":
    args = __parse_args()
    main(args)