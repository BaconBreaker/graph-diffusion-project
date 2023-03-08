"""
Main training functionality for the diffusion model.

@Author Thomas Christensen and Rasmus Pallisgaard
"""
import pytorch_lightning as pl

from dataloader import MoleculeDataModule

from utils.get_config import get_model, get_diffusion
from utils.logging import setup_logging

from DiffusionWrapper import DiffusionWrapper

def train(args):
    setup_logging(args.run_name)
    model, pretransform, posttransform = get_model(args)
    diffusion_model = get_diffusion(args)
    dataloader = MoleculeDataModule(args=args, transform=pretransform)
    diffusion = DiffusionWrapper(denoising_fn=model, diffusion_model=diffusion_model, lr=args.lr, posttransform=posttransform)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(diffusion, dataloader)
