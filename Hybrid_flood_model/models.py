# Updated Local UNet Architectures
# 1. Lightweight UNet
# 2. Attention UNet

import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic conv block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# ----------------------
# 1. Lightweight UNet
# ----------------------
class LightweightUNet(nn.Module):
    def __init__(self, in_channels=5, global_channels=64):
        super().__init__()
        self.in_channels = in_channels + global_channels

        self.enc1 = ConvBlock(self.in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x, global_context):
        gctx = F.interpolate(global_context, size=x.shape[2:], mode='bilinear')
        x = torch.cat([x, gctx], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        b = self.bottleneck(self.pool(e2))

        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)

# ----------------------
# 2. Attention UNet
# ----------------------
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(F.relu(g1 + x1))
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=5, global_channels=64):
        super().__init__()
        self.in_channels = in_channels + global_channels

        self.enc1 = ConvBlock(self.in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(64, 128)

        self.att2 = AttentionBlock(128, 64, 32)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.att1 = AttentionBlock(64, 32, 16)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x, global_context):
        gctx = F.interpolate(global_context, size=x.shape[2:], mode='bilinear')
        x = torch.cat([x, gctx], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        b = self.bottleneck(self.pool(e2))

        e2_att = self.att2(b, e2)
        d2 = self.dec2(torch.cat([self.up2(b), e2_att], dim=1))

        e1_att = self.att1(d2, e1)
        d1 = self.dec1(torch.cat([self.up1(d2), e1_att], dim=1))

        return self.out_conv(d1)
