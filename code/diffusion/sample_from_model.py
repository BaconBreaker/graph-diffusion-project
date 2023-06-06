from main import parse_args, check_args
import logging
from utils.get_config import get_model, get_diffusion
from dataloader import MoleculeDataModule
from utils.metrics import get_metrics
from DiffusionWrapper import DiffusionWrapper
from tqdm.auto import tqdm
from utils.pdf import calculate_pdf_batch
from utils.metrics import rwp_metric, mse_metric, pearson_metric
from utils.plots import save_graph_str_batch, make_histograms
import os
import torch


def diffusion_process_log(batch_dict, posttransform, T, t_skips, diffusion_model, fixed_noises):
    pbar = tqdm(range(0, T, t_skips))
    batch_size = batch_dict[list(batch_dict.keys())[0]].shape[0]
    log_strs = save_graph_str_batch(batch_dict, posttransform, [""] * batch_size)

    diffused_batch = batch_dict.copy()
    for t in pbar:
        diffused_batch, _ = diffusion_model.diffuse(diffused_batch, t, fixed_noises)
        log_strs = save_graph_str_batch(diffused_batch, posttransform, log_strs)

    return log_strs


def generate_samples(args):
    logging.info("Loading model and transforms.")
    model, pretransform, posttransform = get_model(args)
    model.to(args.device)

    logging.info("Loading diffusion noise model")
    diffusion_model = get_diffusion(args)

    logging.info("Loading dataloader")
    dataloader = MoleculeDataModule(args=args, transform=pretransform)

    logging.info("Loading metrics")
    metrics = get_metrics(args)

    logging.info("Loading diffusion wrapper model.")
    if args.checkpoint_path is not None:
        diffusion = DiffusionWrapper.load_from_checkpoint(
            args.checkpoint_path, denoising_fn=model, diffusion_model=diffusion_model,
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
    val_dl = dataloader.val_dataloader()
    ex_batch = next(iter(val_dl))
    ex_batch = dataloader.transfer_batch_to_device(ex_batch, args.device, 0)

    print("lenght of dataloader: ", len(val_dl))

    if args.fix_noise:
        fixed_noises = [diffusion_model.sample_from_noise_fn(
            ex_batch[tensor].shape) for tensor in diffusion_model.tensors_to_diffuse]
    else:
        fixed_noises = None

    # logging.info("Starting diffusion process")
    # log_strs = diffusion_process_log(ex_batch, posttransform, args.diffusion_timesteps,
    #                                  args.t_skips, diffusion_model, fixed_noises)
    # logging.info(f"Diffusion process finished with {len(log_strs[0])} length logs.")

    logging.info("Starting sampling process of 64 samples")
    rwps = []
    n_samples_per_structure = 5
    logging.info(f"Number of samples per structure {n_samples_per_structure}")
    n_samples_pbar = tqdm(range(n_samples_per_structure),
                          total=n_samples_per_structure,
                          position=0,
                          leave=True)
    for i in n_samples_pbar:
        # logging.info("Starting sample")
        samples, _ = diffusion.sample_graphs(ex_batch,
                                             post_process=posttransform,
                                             save_output=False,
                                             noise=fixed_noises,
                                             t_skips=args.t_skips,
                                             pbar_position=1)

        # print(samples.keys())
        # print(samples)
        # logging.info("Computed sample")
        matrix_in, atom_species, r, pdf, pad_mask = posttransform(samples)
        # logging.info("Performed post transform on sample")
        predicted_pdf = calculate_pdf_batch(matrix_in, atom_species, pad_mask)
        # logging.info("Calculated pdf on sample")
        rwps.append(rwp_metric(predicted_pdf, pdf))
        # logging.info("computed rwp on sample")

    print(rwps)

    # logging.info(f"Reverse diffusion process finished with {len(log_strs[0])} length logs.")


    # logging.info("Making sample dirs")
    # for i in range(len(log_strs)):
    #     os.makedirs(f"{args.run_name}_sample_{i}", exist_ok=True)

    # logging.info("Making histograms")
    # for i in range(len(log_strs)):
    #     log = log_strs[i]
    #     size = int(log[0].strip()) + 2  # +2 for the header lines
    #     n_samples = len(log) // size
    #     # Make position tensor
    #     mols = []
    #     for i in range(n_samples):
    #         atoms = log[i * size:(i + 1) * size]  # Fetch the lines corresponding to the atoms
    #         atoms = atoms[2:]  # Remove header
    #         mol = torch.tensor([atom.split()[1:] for atom in atoms], dtype=torch.float32)  # Convert positions to tensor
    #         mols.append(mol)
    #     mols = torch.stack(mols)

        # Make histogram
        # make_histograms(mols, f"{args.run_name}_sample_{i}/")

    # logging.info("Completed saving")


def main():
    args = parse_args()
    check_args(args)
    generate_samples(args)


if __name__ == "__main__":
    main()