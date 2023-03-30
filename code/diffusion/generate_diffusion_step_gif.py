import logging

from main import parse_args, check_args
from utils.get_config import get_model, get_diffusion
from DiffusionWrapper import DiffusionWrapper
from dataloader import MoleculeDataModule
from utils.plots import save_graph
from utils.metrics import get_metrics
from tqdm.auto import tqdm
from utils.plots import save_gif

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
    tensors_to_diffuse = diffusion.diffusion_model.tensors_to_diffuse
    fixed_noises = [diffusion.diffusion_model.sample_from_noise_fn(
        ex_batch[tensor].shape) for tensor in tensors_to_diffuse]

    noise_steps = diffusion.diffusion_model.noise_steps
    pbar = tqdm(range(0, noise_steps))


    for t in pbar:
        diffused_batch, _ = diffusion.diffusion_model.diffuse(ex_batch, t, fixed_noises)
        save_graph(diffused_batch, t + 1, args.run_name, posttransform)

    logging.info("tensors have been generated")

    save_gif(args.run_name, noise_steps, args.batch_size, fps=10,
             skips=args.t_skips)

    logging.info("Gifs have been saved")

def main():
    args = parse_args()
    check_args(args)
    generate_gif(args)


if __name__ == "__main__":
    main()