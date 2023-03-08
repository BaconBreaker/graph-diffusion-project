import pytorch_lightning as pl
import torch

from utils.pdf import calculate_pdf_batch
from utils.metrics import rwp_metric, mse_metric, pearson_metric


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, denoising_fn, diffusion_model, lr, posttransform):
        """
        Wrapper for the diffusion model for training with pytorch lightning
        """
        super().__init__()
        self.denoising_fn = denoising_fn
        self.diffusion_model = diffusion_model
        self.posttransform = posttransform
        self.lr = lr

    def forward(self, x):
        return self.denoising_fn(x)

    def training_step(self, batch, batch_idx):
        noisy_batch, noise = self.diffusion_model.diffuse(batch)
        prediction = self.denoising_fn(noisy_batch)
        loss = self.diffusion_model.loss(prediction, noise, batch)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        samples = self.diffusion_model.sample(batch)
        matrix_in, atom_species, r, pdf, pad_mask = self.posttransform(samples)
        predicted_pdf = calculate_pdf_batch(matrix_in, atom_species, pad_mask)

        self.log("rwp", rwp_metric(predicted_pdf, pdf))
        self.log("mse", mse_metric(predicted_pdf, pdf))
        self.log("pearson", pearson_metric(predicted_pdf, pdf))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
