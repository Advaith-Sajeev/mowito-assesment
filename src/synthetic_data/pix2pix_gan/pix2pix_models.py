# pix2pix_models.py

import torch
import torch.nn as nn


# ------------------------
# U-Net building blocks
# ------------------------

class UNetDown(nn.Module):
    def __init__(self, in_ch, out_ch, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetUp(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.block(x)
        # concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        return x


# ------------------------
# U-Net Generator (adapted for 128x128)
# ------------------------

class Pix2PixGenerator(nn.Module):
    """
    U-Net-like generator for pix2pix.
    Input:  (B, in_channels, H, W)  e.g. in_channels=2 (clean, mask)
    Output: (B, out_channels, H, W) e.g. out_channels=1 (bad patch)

    Depth is chosen so that 128x128 -> 1x1 at bottleneck:
        128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1  (7 downs)
    """

    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()

        # ---- Encoder (7 downs) ----
        self.down1 = UNetDown(in_channels, 64,  normalize=False)  # 128 -> 64
        self.down2 = UNetDown(64,  128)                           # 64  -> 32
        self.down3 = UNetDown(128, 256)                           # 32  -> 16
        self.down4 = UNetDown(256, 512, dropout=0.5)              # 16  -> 8
        self.down5 = UNetDown(512, 512, dropout=0.5)              # 8   -> 4
        self.down6 = UNetDown(512, 512, dropout=0.5)              # 4   -> 2
        self.down7 = UNetDown(512, 512, normalize=False, dropout=0.5)  # 2 -> 1 (bottleneck)

        # ---- Decoder (6 ups + final) ----
        self.up1 = UNetUp(512, 512, dropout=0.5)   # 1 -> 2, concat d6 (512) => 1024
        self.up2 = UNetUp(1024, 512, dropout=0.5)  # 2 -> 4, concat d5 => 1024
        self.up3 = UNetUp(1024, 512, dropout=0.5)  # 4 -> 8, concat d4 => 1024
        self.up4 = UNetUp(1024, 256)               # 8 -> 16, concat d3 => 512
        self.up5 = UNetUp(512, 128)                # 16 -> 32, concat d2 => 256
        self.up6 = UNetUp(256, 64)                 # 32 -> 64, concat d1 => 128

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),  # 64 -> 128
            nn.Tanh(),  # output in [-1, 1]
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)   # (B,64,64,64)
        d2 = self.down2(d1)  # (B,128,32,32)
        d3 = self.down3(d2)  # (B,256,16,16)
        d4 = self.down4(d3)  # (B,512,8,8)
        d5 = self.down5(d4)  # (B,512,4,4)
        d6 = self.down6(d5)  # (B,512,2,2)
        d7 = self.down7(d6)  # (B,512,1,1) bottleneck

        # Decoder with skips
        u1 = self.up1(d7, d6)  # (B,1024,2,2)
        u2 = self.up2(u1, d5)  # (B,1024,4,4)
        u3 = self.up3(u2, d4)  # (B,1024,8,8)
        u4 = self.up4(u3, d3)  # (B,512,16,16)
        u5 = self.up5(u4, d2)  # (B,256,32,32)
        u6 = self.up6(u5, d1)  # (B,128,64,64)

        out = self.final(u6)   # (B,1,128,128)
        return out


# ------------------------
# PatchGAN Discriminator
# ------------------------

class PatchDiscriminator(nn.Module):
    """
    Like pix2pix discriminator.
    Input: concatenation of condition and generated/real patch.
           e.g. (B, in_channels, H, W) where in_channels = 2+1=3.
    Output: (B, 1, H', W') logits.
    """

    def __init__(self, in_channels=3):
        super().__init__()

        def block(in_ch, out_ch, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 1),  # (B,1,H',W')
        )

    def forward(self, x):
        return self.model(x)
