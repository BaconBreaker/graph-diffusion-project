import torch
import torch.nn as nn
import pytorch_lightning as pl

from networks.blocks import SelfAttention


class SelfAttentionNetwork(pl.LightningModule):
    def __init__(self, args):
        """
        Self attention network.
        """
        super().__init__()
        self.device = args.device
        self.time_dim = args.time_dim
        self.num_classes = args.num_classes
        input_size = args.image_size

        self.sa1 = SelfAttention(3, input_size)
        self.sa2 = SelfAttention(3, input_size)
        self.sa3 = SelfAttention(3, input_size)
        self.sa4 = SelfAttention(3, input_size)

        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(3)
        self.bn3 = nn.BatchNorm2d(3)

        self.outc = nn.Conv2d(3, 1, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, labels=None):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        t = t[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        x = torch.concat([x, t], dim=1)

        x = self.sa1(x)
        x = self.bn1(x)
        x = self.sa2(x)
        x = self.bn2(x)
        x = self.sa3(x)
        x = self.bn3(x)
        x = self.sa4(x)

        x = self.outc(x)

        upper = torch.triu(x)
        x = upper + upper.transpose(-1, -2)

        return x

    def training_step(self, batch, batch_idx):
        """
        Training step for the network.
        """
        pass

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the network.
        """
        pass

    def test_step(self, batch, batch_idx):
        """
        Test step for the network.
        """
        raise NotImplementedError

    def configure_optimizers(self):
        """
        Configures the optimizer for the network.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
