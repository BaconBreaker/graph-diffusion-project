"""
Main training functionality for the diffusion model.

@Author Thomas Christensen and Rasmus Pallisgaard
"""
import logging

import pytorch_lightning as pl
from dataloader import MoleculeDataModule
from utils.get_config import get_model, get_diffusion
from callbacks import get_callbacks
from utils.metrics import get_metrics
from DiffusionWrapper import DiffusionWrapper

import torch
torch.autograd.set_detect_anomaly(True)


def train(args):
    logging.info("Training started.")

    logging.info("Loading model and transforms.")
    model, pretransform, posttransform = get_model(args)
    logging.info("Loading diffusion noise model")
    diffusion_model = get_diffusion(args)
    logging.info("Loading metrics")
    metrics = get_metrics(args)

    logging.info("Loading diffusion wrapper model.")
    if args.checkpoint_path is not None:
        logging.info("Loading checkpoint.")
        diffusion = DiffusionWrapper.load_from_checkpoint(
            args.checkpoint_path, denoising_fn=model, diffusion_model=diffusion_model,
            lr=args.lr, posttransform=posttransform, metrics=metrics,
            sample_interval=args.sample_interval
        )
        logging.info("Checkpoint loaded.")
    else:
        diffusion = DiffusionWrapper(denoising_fn=model,
                                diffusion_model=diffusion_model, lr=args.lr,
                                posttransform=posttransform, metrics=metrics,
                                sample_interval=args.sample_interval)

    logging.info("Loading dataloader")
    dataloader = MoleculeDataModule(args=args, transform=pretransform)

    logging.info("Loading callbacks.")
    callbacks = get_callbacks(args)
    logging.info("callbacks:\n{}".format(callbacks))

    logging.info("Creating trainer.")
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)

    logging.info("Beginning training now.")
    trainer.fit(diffusion, dataloader)
    logging.info("Finished training.")
