from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, gating_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gating = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gating: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.gating(gating)
        x = self.skip(skip)
        if g.shape[-2:] != x.shape[-2:]:
            g = nn.functional.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        alpha = self.attention(self.relu(g + x))
        return skip * alpha


class CustomAttentionUNet(nn.Module):
    """Five-level U-Net with additive attention gates on skip connections."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.enc5 = ConvBlock(512, 1024)

        self.up5 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.att5 = AttentionGate(512, 512, 256)
        self.dec5 = ConvBlock(1024, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.att4 = AttentionGate(256, 256, 128)
        self.dec4 = ConvBlock(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.att3 = AttentionGate(128, 128, 64)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.att2 = AttentionGate(64, 64, 32)
        self.dec2 = ConvBlock(128, 64)

        self.output = nn.Conv2d(64, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        u5 = self.up5(e5)
        d5 = self.dec5(torch.cat([self.att5(u5, e4), u5], dim=1))

        u4 = self.up4(d5)
        d4 = self.dec4(torch.cat([self.att4(u4, e3), u4], dim=1))

        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([self.att3(u3, e2), u3], dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([self.att2(u2, e1), u2], dim=1))

        return self.output(d2)


def create_model(name: str, pretrained: bool = True) -> nn.Module:
    if name == "ResNet34_UNet_Baseline":
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=1,
            activation=None,
        )

    if name == "ResNet34_UNet_scSE":
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            decoder_attention_type="scse",
            in_channels=3,
            classes=1,
            activation=None,
        )

    if name == "Custom_Attention_UNet":
        return CustomAttentionUNet()

    raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
