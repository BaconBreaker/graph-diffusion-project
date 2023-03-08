"""
Main training functionality for the diffusion model.

@Author Thomas Christensen and Rasmus Pallisgaard
"""
import pytorch_lightning as pl

from dataloader import MoleculeDataModule

from utils.get_model import get_model
from utils.logging import setup_logging

from carbontracker.tracker import CarbonTracker


def train(args):
    tracker = CarbonTracker(epochs=args.epochs)

    setup_logging(args.run_name)
    model, pretransform, posttransform = get_model(args)
    dataloader = MoleculeDataModule(args=args, preransform=pretransform)
    diffusion = Diffusion(denoising_fn=model, posttransform=posttransform,
                          tracker=tracker, args=args)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(diffusion, dataloader)
