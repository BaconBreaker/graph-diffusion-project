import pytorch_lightning as pl
import torch

from utils.pdf import calculate_pdf_batch
from utils.metrics import rwp_metric, mse_metric, pearson_metric


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, denoising_fn, diffusion_model, lr, posttransform, metrics, sample_interval):
        """
        Wrapper for the diffusion model for training with pytorch lightning
        """
        super().__init__()
        self.denoising_fn = denoising_fn
        self.diffusion_model = diffusion_model
        self.posttransform = posttransform
        self.lr = lr
        self.tensors_to_diffuse = diffusion_model.tensors_to_diffuse
        self.metrics = metrics
        self.sample_interval = sample_interval

    def forward(self, x):
        return self.denoising_fn(x)

    def training_step(self, batch, batch_idx):
        batch_size = batch['pdf'].shape[0]
        t = self.diffusion_model.sample_time_steps(batch_size)
        noisy_batch, noise = self.diffusion_model.diffuse(batch, t)
        prediction = self.denoising_fn(noisy_batch, t)
        loss = torch.tensor(0.0)
        for i, name in enumerate(self.tensors_to_diffuse):
            pred = prediction[name]
            noise_i = noise[i]

            # Check if tthe noise should be padded
            if hasattr(self.denoising_fn, "pad_noise"):
                noise_i = self.denoising_fn.pad_noise(noise_i, batch)
            
            loss += self.diffusion_model.loss(pred, noise_i, batch)
            for metric_name, metric_fn in self.metrics:
                self.log(f"{i}: {metric_name}", metric_fn(pred, noise_i), prog_bar=True)

        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch['pdf'].shape[0]
        t = self.diffusion_model.sample_time_steps(batch_size)
        noisy_batch, noise = self.diffusion_model.diffuse(batch, t)
        prediction = self.denoising_fn(noisy_batch, t)
        loss = torch.tensor(0.0)
        for i, name in enumerate(self.tensors_to_diffuse):
            pred = prediction[name]
            noise_i = noise[i]

            # Check if tthe noise should be padded
            if hasattr(self.denoising_fn, "pad_noise"):
                noise_i = self.denoising_fn.pad_noise(noise_i, batch)

            loss += self.diffusion_model.loss(pred, noise_i, batch)
            for metric_name, metric_fn in self.metrics:
                self.log(f"{i}: val_{metric_name}", metric_fn(pred, noise_i), prog_bar=True)

        self.log("val_loss", loss)
        
        if (self.current_epoch + 1) % self.sample_interval == 0 and self.sample_interval > 0:
            self.sample_graphs(batch)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def sample_graphs(self, batch):
        """
        Function to sample graphs from the diffusion model and evaluate them
        """
        samples = self.diffusion_model.sample(self.denoising_fn, batch)
        matrix_in, atom_species, r, pdf, pad_mask = self.posttransform(samples)
        predicted_pdf = calculate_pdf_batch(matrix_in, atom_species, pad_mask)

        self.log("RWP", rwp_metric(predicted_pdf, pdf), prog_bar=True)
        self.log("MSE of pdf", mse_metric(predicted_pdf, pdf), prog_bar=True)
        self.log("Pearson", pearson_metric(predicted_pdf, pdf), prog_bar=True)
