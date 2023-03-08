import pytorch_lightning as pl
import torch

from utils.pdf import calculate_pdf_batch
from utils.metrics import rwp_metric, mse_metric, pearson_metric


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, denoising_fn, diffusion_model, args, posttransform=None):
        """
        Wrapper for the diffusion model for training with pytorch lightning
        """
        super().__init__()
        self.denoising_fn = denoising_fn
        self.diffusion_model = diffusion_model
        self.posttransform = posttransform
        self.lr = args.lr

    def forward(self, x):
        return self.denoising_fn(x)

    def training_step(self, batch, batch_idx):
        noisy_batch, noise = self.diffusion_model.diffuse(batch)
        prediction = self.denoising_fn(noisy_batch)
        loss = self.diffusion_model.loss(prediction, noise, batch)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        samples = self.diffusion.sample(batch)
        samples = self.posttransform(samples)
        real_pdf = batch['pdf']
        predicted_pdf = calculate_pdf_batch(samples)
        
        self.log("rwp", rwp_metric(predicted_pdf, real_pdf))
        self.log("mse", mse_metric(predicted_pdf, real_pdf))
        self.log("pearson", pearson_metric(predicted_pdf, real_pdf))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
