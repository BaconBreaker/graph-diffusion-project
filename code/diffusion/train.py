"""
Main training functionality for the diffusion model.

@Author Thomas Christensen and Rasmus Pallisgaard
"""
import pytorch_lightning as pl
from dataloader import MoleculeDataModule
from utils.get_config import get_model, get_diffusion
from callbacks import get_callbacks
from utils.metrics import get_metrics
from DiffusionWrapper import DiffusionWrapper

def train(args):
    model, pretransform, posttransform = get_model(args)
    diffusion_model = get_diffusion(args)
    dataloader = MoleculeDataModule(args=args, transform=pretransform)
    metrics = get_metrics(args)
    diffusion = DiffusionWrapper(denoising_fn=model, diffusion_model=diffusion_model,
                                 lr=args.lr, posttransform=posttransform, metrics=metrics)
    callbacks = get_callbacks(args)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.fit(diffusion, dataloader)
