import logging

from main import parse_args, check_args
from utils.get_config import get_model, get_diffusion
from DiffusionWrapper import DiffusionWrapper
from dataloader import MoleculeDataModule
from callbacks import get_callbacks
from utils.plots import save_graph
from utils.metrics import get_metrics

def generate_gif(args):
    logging.info("Loading model and transforms.")
    model, pretransform, posttransform = get_model(args)
    logging.info("Loading diffusion noise model")
    diffusion_model = get_diffusion(args)
    logging.info("Loading dataloader")
    dataloader = MoleculeDataModule(args=args, transform=pretransform)
    logging.info("Loading metrics")
    metrics = get_metrics(args)
    logging.info("Loading diffusion wrapper model.")
    if args.checkpoint_path is not None:
        diffusion = DiffusionWrapper.load_from_checkpoint(
            args.checkpoint_path,denoising_fn=model, diffusion_model=diffusion_model,
            lr=args.lr, posttransform=posttransform, metrics=metrics,
            sample_interval=args.sample_interval
        )
    else:
        diffusion = DiffusionWrapper(denoising_fn=model,
                                     diffusion_model=diffusion_model, lr=args.lr,
                                     posttransform=posttransform, metrics=metrics,
                                     sample_interval=args.sample_interval)

    logging.info("Preparing data.")
    dataloader.prepare_data()
    dataloader.setup()

    logging.info("preparing batch.")
    train_dl = dataloader.train_dataloader()
    ex_batch = next(iter(train_dl))
    ex_batch = dataloader.transfer_batch_to_device(ex_batch, args.device, 0)

    logging.info("Sampling graph.")
    diffusion.sample_graphs(ex_batch,
                            post_process=posttransform,
                            skips=10)

    logging.info("Gifs have been generated")

def main():
    args = parse_args()
    check_args(args)
    generate_gif(args)


if __name__ == "__main__":
    main()