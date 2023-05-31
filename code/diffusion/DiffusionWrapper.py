import time

import pytorch_lightning as pl
import torch

from utils.pdf import calculate_pdf_batch
from utils.metrics import rwp_metric, mse_metric, pearson_metric


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, denoising_fn, diffusion_model, lr,
                 posttransform, metrics, sample_interval):
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

        # For quick debugging
        self.count_dict = None

    def forward(self, x):
        return self.denoising_fn(x)

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        batch_size = batch['pdf'].shape[0]
        t = self.diffusion_model.sample_time_steps(batch_size)
        noisy_batch, noise = self.diffusion_model.diffuse(batch, t)
        prediction = self.denoising_fn(noisy_batch, t)
        prediction = prediction
        loss = torch.tensor(0.0).to(self.device)
        for i, name in enumerate(self.tensors_to_diffuse):
            pred = prediction[name]
            noise_i = noise[i]

            # Check if tthe noise should be padded
            if hasattr(self.denoising_fn, "pad_noise"):
                noise_i = self.denoising_fn.pad_noise(noise_i, batch)

            loss += self.diffusion_model.loss(pred, noise_i, batch)
            for metric_name, metric_fn in self.metrics:
                self.log(f"{i}: {metric_name}", metric_fn(pred, noise_i),
                         prog_bar=True, sync_dist=True)

        self.log("loss", loss, sync_dist=True)

        if self.count_dict is None:
            self.count_dict = {}
            for name, param in self.denoising_fn.named_parameters():
                if 'weight' in name:
                    self.count_dict[name] = torch.zeros(param.grad.shape)

        # For quick debuggung
        for name, param in self.denoising_fn.named_parameters():
            if 'weight' in name and param.grad is not None:
                temp = torch.zeros(param.grad.shape)
                temp[param.grad != 0] += 1
                self.count_dict[name] += temp

        print(sum([1 for name, ten in self.count_dict.items() if torch.sum(ten) == 0]))

        return loss

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        batch_size = batch['pdf'].shape[0]
        t = self.diffusion_model.sample_time_steps(batch_size)
        noisy_batch, noise = self.diffusion_model.diffuse(batch, t)
        prediction = self.denoising_fn(noisy_batch, t)
        loss = torch.tensor(0.0).to(self.device)
        for i, name in enumerate(self.tensors_to_diffuse):
            pred = prediction[name]
            noise_i = noise[i]

            # Check if tthe noise should be padded
            if hasattr(self.denoising_fn, "pad_noise"):
                noise_i = self.denoising_fn.pad_noise(noise_i, batch)

            loss += self.diffusion_model.loss(pred, noise_i, batch)
            for metric_name, metric_fn in self.metrics:
                self.log(f"{i}: val_{metric_name}", metric_fn(pred, noise_i),
                         prog_bar=True, sync_dist=True)

        self.log("val_loss", loss, sync_dist=True)

        do_sample = self.sample_interval > 0
        do_sample = do_sample and (self.current_epoch + 1) % self.sample_interval == 0
        do_sample = do_sample and batch_idx == 0
        if do_sample:
            self.sample_graphs(batch)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def sample_graphs(self, batch, post_process=None, save_output=False, noise=None, t_skips=1, log_strs=None):
        """
        Function to sample graphs from the diffusion model and evaluate them
        args:
            batch: batch of data to sample from
            post_process: function to post process the samples
            save_output: whether to save the output
            noise: noise to use for sampling
            t_skips: number of time steps to skip
            log_strs: list of string logs to append to
        """
        samples, log_strs = self.diffusion_model.sample(self.denoising_fn,
                                                        batch,
                                                        save_output=save_output,
                                                        post_process=post_process,
                                                        noise=noise,
                                                        t_skips=t_skips,
                                                        log_strs=log_strs)

        matrix_in, atom_species, r, pdf, pad_mask = self.posttransform(samples)
        predicted_pdf = calculate_pdf_batch(matrix_in, atom_species, pad_mask)

        self.log("RWP", rwp_metric(predicted_pdf, pdf), prog_bar=True, sync_dist=True)
        self.log("MSE of pdf", mse_metric(predicted_pdf, pdf), prog_bar=True,
                 sync_dist=True)
        self.log("Pearson", pearson_metric(predicted_pdf, pdf), prog_bar=True,
                 sync_dist=True)

        return log_strs

    def on_fit_start(self):
        self.diffusion_model.set_device(self.device)

    def on_validation_start(self):
        self.diffusion_model.set_device(self.device)

    def on_test_start(self):
        self.diffusion_model.set_device(self.device)
