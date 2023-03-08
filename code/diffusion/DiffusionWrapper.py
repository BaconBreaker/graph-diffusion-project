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
        self.tensors_to_diffuse = diffusion_model.tensors_to_diffuse

    def forward(self, x):
        return self.denoising_fn(x)

    def training_step(self, batch, batch_idx):
        batch_size = batch['pdf'].shape[0]
        t = self.diffusion_model.sample_time_steps(batch_size)
        noisy_batch, noise = self.diffusion_model.diffuse(batch, t)
        prediction = self.denoising_fn(noisy_batch, t)
        loss = torch.tensor(0.0)
        for i, name in enumerate(self.tensors_to_diffuse):
            loss += self.diffusion_model.loss(prediction[name], noise[i], batch)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        samples = self.diffusion_model.sample(self.denoising_fn, batch)
        matrix_in, atom_species, r, pdf, pad_mask = self.posttransform(samples)
        predicted_pdf = calculate_pdf_batch(matrix_in, atom_species, pad_mask)

        self.log("rwp", rwp_metric(predicted_pdf, pdf), prog_bar=True)
        self.log("mse", mse_metric(predicted_pdf, pdf), prog_bar=True)
        self.log("pearson", pearson_metric(predicted_pdf, pdf), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
