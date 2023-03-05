import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s %(message)s',
                    level=logging.INFO,
                    datefmt="%I:%M:%S")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # ## Run parameters ##
    parser.add_argument("--run_name", type=str, default="DDPM_Unconditional",
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

    # ## Model parameters ##
    parser.add_argument("--model", type=str, default="self_attention",
                        help="Model to use for training, options: self_attention, conditional_unet")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate to use for training")

    # ## Noise/Diffusion parameters ##
    parser.add_argument("--noise_shape", type=int, nargs="+", default=[3, 64, 64])
    parser.add_argument("--noise_function", type=str, default="gaussian",
                        help="Noise function to use, options: gaussian, uniform, symmetricgaussian")
    parser.add_argument("--diffusion_timesteps", type=int, default=100,
                        help="Number of diffusion timesteps to use")

    return parser.parse_args()


def check_args(args):
    """Check if the arguments are valid"""
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("Cuda is not available, falling back to cpu")
        args.device = "cpu"
    if args.val_split < 0 or args.val_split > 1:
        raise ValueError("val_split must be between 0 and 1")
    return args


if __name__ == "__main__":
    args = parse_args()
    # By importing train here we can run the help command from the command line
    # faster, since we don't have to wait for the imports to finish.
    from train import train
    import torch
    args = check_args(args)
    train(args)
