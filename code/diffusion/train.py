"""
Main training functionality for the diffusion model.

@Author Thomas Christensen and Rasmus Pallisgaard
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm.auto import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter

from diffusion import Diffusion

from dataloader import MoleculeDataModule

from utils.get_model import get_model
from utils.logging import setup_logging
from utils.metrics import pearson_metric, rwp_metric, mse_metric
from utils.pdf import calculate_pdf_batch
# from utils.ema import EMA


def train(args):
    setup_logging(args.run_name)
    device = args.device
    model, pretransform, posttransform = get_model(args)
    dataloader = MoleculeDataModule(args, pretransform)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    noise_shape = args.noise_shape  # (3, args.image_size, args.image_size)
    diffusion = Diffusion(noise_function=args.noise_function,
                          noise_schedule="cosine",
                          noise_steps=args.diffusion_timesteps,
                          device=device,
                          noise_shape=noise_shape)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    # ema = EMA(beta=0.995)
    # ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        dataloader.prepare_data()
        dataloader.setup()
        dataloader = dataloader.train_dataloader()
        pbar = tqdm(dataloader)
        for i, inputs in enumerate(pbar):
            print(inputs)
            atom_species = inputs[0][:, :, 0]
            labels = labels.to(device)
            t = diffusion.sample_time_steps(images.shape[0]).to(device)
            x_t, noise = diffusion.diffuse(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * dl_len + i)

        if epoch % 10 == 0 and epoch > 0:
            labels = torch.arange(1).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            predicted_pdfs = calculate_pdf_batch(sampled_images, atom_species, pad_mask)

            # ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)

            mse_m = mse_metric(pdfs, predicted_pdfs)
            rwp = rwp_metric(pdfs, predicted_pdfs)
            pearson = pearson_metric(pdfs, predicted_pdfs)
            print(f"MSE: {mse_m}, RWP: {rwp}, Pearson: {pearson}")

            torch.save(model.state_dict(),
                       os.path.join("models", args.run_name, "ckpt.pth"))
            # torch.save(ema_model.state_dict(),
            #            os.path.join("models", args.run_name, "ema_ckpt.pth"))
            torch.save(optimizer.state_dict(),
                       os.path.join("models", args.run_name, "optim.pth"))
