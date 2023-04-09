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

    def forward(self, x):
        return self.denoising_fn(x)

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())
        # print("training")
        # print("batch")
        # print(batch)
        # print("batch size shape")
        # print(batch["edge_sequence"].shape)
        batch_size = batch['pdf'].shape[0]
        t = self.diffusion_model.sample_time_steps(batch_size)
        noisy_batch, noise = self.diffusion_model.diffuse(batch, t)
        # print(1)
        prediction = self.denoising_fn(noisy_batch, t)
        # print(2)
        prediction = prediction
        loss = torch.tensor(0.0).to(self.device)
        # print("loss at initialization")
        # print(loss.device)
        # print(f"len of tensors to diffuse: {len(self.tensors_to_diffuse)}")
        for i, name in enumerate(self.tensors_to_diffuse):
            # print(f"name of tensor: {name}")
            pred = prediction[name]
            noise_i = noise[i]
            # print("calc_loss")
            # print(pred.device)
            # print(noise_i.device)
            # print(3)

            # time.sleep(5)

            # Check if tthe noise should be padded
            if hasattr(self.denoising_fn, "pad_noise"):
                noise_i = self.denoising_fn.pad_noise(noise_i, batch)

            loss += self.diffusion_model.loss(pred, noise_i, batch)
            for metric_name, metric_fn in self.metrics:
                self.log(f"{i}: {metric_name}", metric_fn(pred, noise_i),
                         prog_bar=True, sync_dist=True)

        # print("end of loop")
        # print("logged loss")
        # print(loss)

        self.log("loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        # print("validation")
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
        pdf = pdf.cpu().detach().numpy()

        if save_output:
            predicted_pdf.save("predicted_pdf.pt")

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
