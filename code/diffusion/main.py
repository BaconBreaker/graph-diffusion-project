import argparse
from train import train
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s %(message)s',
                    level=logging.INFO,
                    datefmt="%I:%M:%S")


def launch():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Unconditional"
    args.epochs = 1
    args.batch_size = 14
    args.image_size = 64
    args.num_classes = 10
    args.dataset_path = "/Users/rasmus/Projects/Diffusion Models/data/cifar10-64/train"
    args.device = "cpu"
    args.lr = 3e-4
    train(args)


if __name__ == "__main__":
    launch()
