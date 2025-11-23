# scratch_gan/gan_models.py

import torch
import torch.nn as nn


class ScratchGenerator(nn.Module):
    """
    Mask-only generator.
    Input:  z (B, z_dim)
    Output: mask_logits (B, 1, H, W)
    """
    def __init__(self, z_dim=128, base_channels=64, out_size=64):
        super().__init__()
        self.z_dim = z_dim

        self.net = nn.Sequential(
            # input: (z_dim, 1, 1)
            nn.ConvTranspose2d(z_dim, base_channels * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),

            # final: single-channel mask logits
            nn.ConvTranspose2d(base_channels, 1, 4, 2, 1, bias=False),
            # no activation here; we'll apply sigmoid outside
        )

    def forward(self, z):
        # z: (B, z_dim)
        z = z.view(z.size(0), self.z_dim, 1, 1)
        mask_logits = self.net(z)  # (B, 1, H, W), real-valued
        return mask_logits


class ScratchDiscriminator(nn.Module):
    """
    Discriminator that looks only at the mask (1 channel).
    Input:  (B, 1, H, W)
    Output: (B,) logits
    """
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 4, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        # x: (B, 1, H, W)
        out = self.net(x)  # (B, 1, 1, 1)
        return out.view(-1)
