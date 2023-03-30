from main import parse_args, check_args
import logging
from utils.get_config import get_model, get_diffusion
from dataloader import MoleculeDataModule
from utils.metrics import get_metrics
from DiffusionWrapper import DiffusionWrapper
from tqdm.auto import tqdm
from utils.plots import save_graph_str_batch


def diffusion_process_log(batch_dict, posttransform, T, t_skips, diffusion_model, fixed_noises):
    pbar = tqdm(range(0, T, t_skips))
    log_strs = save_graph_str_batch(batch_dict, posttransform, [])
    logging.info("TEST", batch_dict['adj_matrix'].shape, type(batch_dict['adj_matrix'].shape))
    
    diffused_batch = batch_dict.copy()
    for t in pbar:
        diffused_batch, _ = diffusion_model.diffuse(diffused_batch, t, fixed_noises)
        log_strs = save_graph_str_batch(diffused_batch, posttransform, log_strs)

    return log_strs

def generate_samples(args):
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

    if args.fix_noise:
        fixed_noises = [diffusion_model.sample_from_noise_fn(
            ex_batch[tensor].shape) for tensor in diffusion_model.tensors_to_diffuse]
    else:
        fixed_noises = None

    logging.info("Starting diffusion process")
    log_strs = diffusion_process_log(ex_batch, posttransform, args.diffusion_timesteps, args.t_skips, diffusion_model, fixed_noises)
    logging.info("Diffusion process finished")

    logging.info("Starting reverse diffusion process")
    diffusion.sample_graphs(ex_batch,
                            post_process=posttransform,
                            save_output=True,
                            noise=fixed_noises,
                            t_skips=args.t_skips,
                            log_strs=log_strs)
    logging.info("Reverse diffusion process finished")

def main():
    args = parse_args()
    check_args(args)
    generate_samples(args)

if __name__ == "__main__":
    main()