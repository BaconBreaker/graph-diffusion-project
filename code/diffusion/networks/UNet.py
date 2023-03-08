import torch
import torch.nn as nn

from networks.blocks import DoubleConv, Down, Up, SelfAttention


class UNet(nn.Module):
    def __init__(self, c_in, c_out, time_dim, device="cpu"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x = self.inc(x)
        x = self.down1(x, t)
        x = self.sa1(x)
        x = self.down2(x, t)
        x = self.sa2(x)
        x = self.down3(x, t)
        x = self.sa3(x)

        x = self.bot1(x)
        x = self.bot2(x)
        x = self.bot3(x)

        x = self.up1(x, t)
        x = self.sa4(x)
        x = self.up2(x, t)
        x = self.sa5(x)
        x = self.up3(x, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
