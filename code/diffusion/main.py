import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s %(message)s',
                    level=logging.INFO,
                    datefmt="%I:%M:%S")


def parse_args():
    parser = argparse.ArgumentParser()

    # ## Run parameters ##
    parser.add_argument("--run_name", type=str, default="DDPM_Unconditional",
                        help="Name of the run, used for logging and saving models")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train for")
    parser.add_argument("--device", type=str, default="cpu",
                        help=("Device to train on, options: cpu, cuda, if set to cuda, "
                              "will check first if cuda is available"))

    # ## Data parameters ##
    parser.add_argument("--batch_size", type=int, default=14,
                        help="Batch size to use for training")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--dataset_path", type=str,
                        default="/Users/rasmus/Projects/Diffusion Models Framework/data/cifar10-64/train",
                        help="Path to the dataset folder")
    parser.add_argument("--graph_data", type=int, default=0, help="Use graph data")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to use for training")

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


if __name__ == "__main__":
    args = parse_args()
    # By importing train here we can run the script from the command line
    # faster, since we don't have to wait for the imports to finish.
    from train import train
    train(args)
