"""Small CNN flood-extent classifier on terrain+forcing stacks (64x64).

NOT a hydraulic surrogate U-Net: predicts extent probability from statics +
event forcings, benchmarked against a HAND-threshold baseline (and Sentinel-1
when available).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                               nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                               nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                               nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.b(x)

class FloodCNN(nn.Module):
    """Encoder-decoder with skip connections (light U-shape ~1.5M params)."""
    def __init__(self, in_channels=6, base=32):
        super().__init__()
        self.e1 = ConvBlock(in_channels, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        d2 = self.d2(torch.cat([self.up2(s3), s2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), s1], dim=1))
        return self.out(d1)  # logits (B,1,H,W)
