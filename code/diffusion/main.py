"""
Main script for training the diffusion model.

Example of how to run the script:
python main.py --dataset_path /home/thomas/graph-diffusion-project/graphs_fixed_num_135/ --run_name 123 --max_epochs 1000 --check_val_every_n_epoch 10 --batch_size 4 --tensors_to_diffuse edge_sequence --pad_length 135 --diffusion_timesteps 3 --num_workers 8 --log_every_n_steps 10
"""
import argparse
import logging
from pprint import pformat

import torch
torch.autograd.set_detect_anomaly(True)

import pytorch_lightning as pl

from train import train

logging.basicConfig(format='%(asctime)s - %(levelname)s %(message)s',
                    level=logging.INFO,
                    datefmt="%I:%M:%S")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # ## Run parameters ##
    parser.add_argument("--run_name", type=str,
                        help="Name of the run, used for logging and saving models")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train for")
    parser.add_argument("--device", type=str, default="cpu",
                        help=("Device to train on, options: cpu, cuda, if set to cuda, "
                              "will check first if cuda is available"))
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers to use for data loading")

    # ## Data parameters ##
    parser.add_argument("--batch_size", type=int, default=14,
                        help="Batch size to use for training")
    parser.add_argument("--dataset_path", type=str,
                        default="/Users/rasmus/Projects/Diffusion Models Framework/data/cifar10-64/train",
                        help="Path to the dataset folder")
    parser.add_argument("--num_train_samples", type=int, default=None,
                        help="Number of training samples to use for training, None means all")
    parser.add_argument("--num_val_samples", type=int, default=None,
                        help="Number of validation samples to use for training, None means all")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of training data to use for validation")
    parser.add_argument("--pad_length", type=int, default=512,  # Longest over all datasets is 511
                        help="Length to pad the graphs to, None means no the script will pad to the max length")
    parser.add_argument("--single_sample", action="store_true", default=False,
                        help="Whether to use a single (specific) sample for training")

    # ## Model parameters ##
    parser.add_argument("--model", type=str, default="self_attention",
                        help="Model to use for training, options: self_attention, conditional_unet")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate to use for training")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of classes to use for the UNET.")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="checkpoint path. Will randomly initialise if None.")

    # ## Noise/Diffusion parameters ##
    parser.add_argument("--diffusion", type=str, default="gaussian",
                        help="Diffusion model to use")
    parser.add_argument("--noise_shape", type=int, nargs="+", default=[3, 64, 64])
    parser.add_argument("--model_target", type=str, default="noise",
                        help="Target of the model, options: noise, image")
    parser.add_argument("--n_categorical_vars", type=int, default=3,
                        help="Number of categorical variables. Only for uniform categorical noise")
    parser.add_argument("--n_values", type=int, default=7,
                        help="Number of values each categorical variable can take. \
                            Only for uniform categorical noise")
    parser.add_argument("--diffusion_timesteps", type=int, default=100,
                        help="Number of diffusion timesteps to use")
    parser.add_argument("--noise_schedule", type=str, default="cosine",
                        help="Noise schedule to use, options: linear, cosine")
    parser.add_argument("--time_dim", type=int, default=2,
                        help="Dimension of the time axis")
    parser.add_argument("--tensors_to_diffuse", type=str, nargs="+", default=["adj_matrix"],
                        help="Names of the tensors that should be diffused")

    # ## Callbacks ##
    parser.add_argument("--disable_carbon_tracker", action="store_true",
                        help="Whether to disable the carbon tracker callback")
    parser.add_argument("--sample_interval", type=int, default=10,
                        help="Interval of epochs between sampling, set to negative number to disable sampling")
    parser.add_argument("--checkpoint", action="store_true",
                        help="Whether to enable the checkpoint callback",
                        default=True)

    # ## Metrics ##
    parser.add_argument('--metrics', type=str, nargs='+', default=[],
                        help='Metrics during training, options are [mse, psnr, snr]')

    # ## Equivariant diffusion parameters ##
    parser.add_argument("--equiv_n_layers", type=int, default=5,
                        help="Number of layers in the equivariant diffusion model")
    parser.add_argument("--equiv_hidden_dim", type=int, default=256,
                        help="Hidden dimension of the equivariant diffusion model")
    parser.add_argument("--equiv_pdf_hidden_dim", type=int, default=1024,
                        help="Hidden dimension of the equivariant diffusion model pdf \
                            intermediate layer")

    # ## Conditional diffusion parameters ##
    parser.add_argument("--conditional", action="store_true",
                        help="Whether to train a conditional model")

    # ## Plotting arguments ##
    t_skips_help = "How many t steps to skip when generating images and gifs. " + \
        "Can be used to speed up generation. Only used for gif generation."
    parser.add_argument("--t_skips", type=int, default=1, help=t_skips_help)
    parser.add_argument("--method", type=str, default="adj_matrix",
                        help="Method for simple plotting, disregard to set to adj_matrix or xyz")
    parser.add_argument("--fix_noise", action="store_true",
                        help="Whether to fix the noise for plotting")

    # ## Pytorch Lightning parameters ##
    parser = pl.Trainer.add_argparse_args(parser, use_argument_group=False)

    return parser.parse_args()


def check_args(args):
    """Check if the arguments are valid"""
    if args.val_split < 0 or args.val_split > 1:
        raise ValueError("val_split must be between 0 and 1")
    if args.num_train_samples is not None and args.num_train_samples <= 0:
        raise ValueError("num_train_samples must be positive")
    if args.num_val_samples is not None and args.num_val_samples <= 0:
        raise ValueError("num_val_samples must be positive")
    return args


if __name__ == "__main__":
    args = parse_args()
    logger.log(logging.INFO, f"Arguments:\n{pformat(vars(args))}")
    train(args)
