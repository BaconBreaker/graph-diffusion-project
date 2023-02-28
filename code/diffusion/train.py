import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm.auto import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter

from models import ConditionalUNet
from utils import get_data, setup_logging, save_images
from diffusion import Diffusion, image_sample_post_process
# from ema import EMA


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args, subset=args.num_samples)
    model = ConditionalUNet(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    noise_shape = args.noise_shape  # (3, args.image_size, args.image_size)
    diffusion = Diffusion(noise_function="gaussian",
                          noise_schedule="cosine",
                          noise_steps=10,
                          device=device,
                          noise_shape=noise_shape)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    dl_len = len(dataloader)
    # ema = EMA(beta=0.995)
    # ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader, total=dl_len)
        for i, (images, labels) in enumerate(pbar):
            labels = labels.to(device)
            t = diffusion.sample_time_steps(images.shape[0]).to(device)
            x_t, noise = diffusion.diffuse(images, t)
            if np.random.random() < 0.1:
                labels = None
            # predict x_0
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * dl_len + i)

        if epoch % 10 == 0:
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels,
                                              sample_post_process=image_sample_post_process)
            # ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            # plot_images(sampled_images)
            save_images(sampled_images,
                        os.path.join("results", args.run_name, f"{epoch}.png"))
            # save_images(ema_sampled_images,
            #             os.path.join("results", args.run_name, f"{epoch}_ema.png"))
            torch.save(model.state_dict(),
                       os.path.join("models", args.run_name, "ckpt.pth"))
            # torch.save(ema_model.state_dict(),
            #            os.path.join("models", args.run_name, "ema_ckpt.pth"))
            torch.save(optimizer.state_dict(),
                       os.path.join("models", args.run_name, "optim.pth"))
