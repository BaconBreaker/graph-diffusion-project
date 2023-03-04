"""
Functions for calculating metrics between predicted and target pdfs

@Author Thomas Chirstensen and Rasmus Pallisgaard
"""
import torch
from scipy.stats import pearsonr


def pearson_metric(predicted_pdf, target_pdf):
    """
    Calculate pearson correlation coefficient between predicted and target pdf
    args:
        predicted_pdf (torch.Tensor): predicted pdf of shape (batch_size, 3000)
        target_pdf (torch.Tensor): target pdf of shape (batch_size, 3000)
    returns:
        pearson (torch.Tensor): pearson correlation coefficient of shape (batch_size,)
    """
    pearson = []
    for i in range(predicted_pdf.shape[0]):
        pearson.append(
            pearsonr(predicted_pdf[i].detach().cpu().numpy(), target_pdf[i].detach().cpu().numpy())[0]
        )
    return torch.tensor(pearson, dtype=torch.float32)


def mse_metric(predicted_pdf, target_pdf):
    """
    Calculate mean squared error between predicted and target pdf
    args:
        predicted_pdf (torch.Tensor): predicted pdf of shape (batch_size, 3000)
        target_pdf (torch.Tensor): target pdf of shape (batch_size, 3000)
    returns:
        mse (torch.Tensor): mean squared error of shape (batch_size,)
    """
    return torch.mean((predicted_pdf - target_pdf) ** 2, dim=1)


def rwp_metric(predicted_pdf, true_pdf, sigmas=None):
    """
    Calculate the Reitveld Weighted Profile (RWP) metric
    args:
        predicted_pdf (torch.Tensor): predicted pdf of shape (batch_size, 3000)
        target_pdf (torch.Tensor): target pdf of shape (batch_size, 3000)
        sigmas (torch.Tensor): condfidence of predicted pdf, defaults to 1 (batch_size, 3000)
    returns:
        rwp (torch.Tensor): RWP metric of shape (batch_size,)
    """
    if sigmas is None:
        sigmas = torch.ones_like(predicted_pdf)
    sigmas_inv = 1 / sigmas
    diff_squarred = (predicted_pdf - true_pdf) ** 2
    true_squarred = true_pdf ** 2
    rwp = torch.sqrt((sigmas_inv * diff_squarred).sum(dim=1) / (sigmas_inv * true_squarred).sum(dim=1))

    return rwp
