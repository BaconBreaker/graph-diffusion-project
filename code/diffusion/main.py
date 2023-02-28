import argparse
from train import train
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s %(message)s',
                    level=logging.INFO,
                    datefmt="%I:%M:%S")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="DDPM_Unconditional")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=14)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--dataset_path", type=str,
                        default="/Users/rasmus/Projects/Diffusion Models Framework/data/cifar10-64/train")
    parser.add_argument("--graph_data", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--noise_shape", type=int, nargs="+", default=[3, 64, 64])

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
