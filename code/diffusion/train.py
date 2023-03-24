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

def train(args):
    logging.info("Started training.")
    model, pretransform, posttransform = get_model(args)
    logging.info("Model loaded.")
    diffusion_model = get_diffusion(args)
    logging.info("Diffusion model loaded.")
    dataloader = MoleculeDataModule(args=args, transform=pretransform)
    logging.info("Dataloader loaded.")
    metrics = get_metrics(args)
    logging.info("Metrics loaded.")
    diffusion = DiffusionWrapper(denoising_fn=model, diffusion_model=diffusion_model,
                                 lr=args.lr, posttransform=posttransform, metrics=metrics,
                                 sample_interval=args.sample_interval)
    logging.info("Diffusion wrapper created.")
    callbacks = get_callbacks(args)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    logging.info("Trainer created.")
    logging.info("Beginning training now.")
    trainer.fit(diffusion, dataloader)
