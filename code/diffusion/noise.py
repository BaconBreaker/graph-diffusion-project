import torch
import logging

logger = logging.getLogger(__name__)


def categorical_noise_function(beta_t, n_categorical):
    """
    Defined by Austin et al. 2023, appendix 2.1.
    We use the linear algebra notation.
    """
    k = n_categorical
    q = (1 - beta_t) * torch.eye(k) + beta_t * torch.ones([k, k]) / k
    return q
