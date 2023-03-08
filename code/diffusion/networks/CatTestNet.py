import torch.nn as nn


class CatTestNet(nn.Module):
    def __init__(self, n_vals):
        super().__init__()
        self.n_vals = n_vals

        self.layer = nn.Linear(n_vals, n_vals)

    def forward(self, x, t, labels=None):
        out = self.layer(x + t)
        return out