"""
PyTorch ResUNet++-style (2D) demo: image-to-image regression/segmentation
- 2D adaptation of your TF3D model:
  stem_block + resnet_blocks + ASPP bridge + attention gates + decoder
- Input:  (B, C_in, H, W)
- Output: (B, C_out, H, W) with tanh (as in your TF code)

Dependencies:
  pip install torch

Notes:
- This is a faithful *structural* translation to 2D (Conv2d/Pool2d/Upsample).
- Attention block here follows your TF logic: pool(g) to match x scale, fuse, then gate x.
- ASPP is implemented as sum of dilated convs + 1x1 conv (matching your Add + Conv).
"""


from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Building blocks (2D)
# -----------------------------
class SEBlock2D(nn.Module):
    """Squeeze-and-Excitation for 2D feature maps."""
    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        # Your TF code used two Dense(filters) layers (no reduction).
        # A typical SE uses reduction; to stay close but stable, we keep a mild reduction by default.
        hidden = max(channels // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.avg_pool(x)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ConvBNReLU2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, d: int = 1):
        super().__init__()
        p = (k // 2) * d
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class StemBlock2D(nn.Module):
    """
    TF stem_block:
      Conv -> BN -> ReLU -> Conv
      shortcut 1x1 conv
      Add -> SE
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, se_ratio: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.short = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False)
        self.short_bn = nn.BatchNorm2d(out_ch)

        self.se = SEBlock2D(out_ch, ratio=se_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_init = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.conv2(x)

        s = self.short_bn(self.short(x_init))
        x = x + s
        x = self.se(x)
        return x


class ResNetBlock2D(nn.Module):
    """
    TF resnet_block:
      BN -> ReLU -> Conv(stride)
      BN -> ReLU -> Conv
      shortcut 1x1 conv(stride)
      Add -> SE
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, se_ratio: int = 8):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.short = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False)
        self.short_bn = nn.BatchNorm2d(out_ch)

        self.se = SEBlock2D(out_ch, ratio=se_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_init = x

        x = F.relu(self.bn1(x), inplace=True)
        x = self.conv1(x)

        x = F.relu(self.bn2(x), inplace=True)
        x = self.conv2(x)

        s = self.short_bn(self.short(x_init))
        x = x + s
        x = self.se(x)
        return x


class ASPPBlock2D(nn.Module):
    """
    TF aspp_block:
      x1(dil=2) + x2(dil=4) + x3(dil=6) + x4(dil=1) -> Add
      -> 1x1 conv
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.c4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.proj = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.c1(x) + self.c2(x) + self.c3(x) + self.c4(x)
        y = self.proj(y)
        return y


class AttentionBlock2D(nn.Module):
    """
    2D adaptation of your TF attention_block(g, x):

    TF steps:
      g -> BN -> ReLU -> Conv
      g_pool = MaxPool(stride=2)
      x -> BN -> ReLU -> Conv
      gc_sum = g_pool + x_conv
      gc_conv = BN -> ReLU -> Conv
      gc_mul = gc_conv * x

    In decoder call:
      d = attention_block(skip, decoder_feature)
      d = upsample(d) then concat with skip
    So here:
      - g: skip feature at higher resolution
      - x: decoder feature at lower resolution (same channels as filters)
      We pool g to match x, fuse, produce gate, multiply x.
    """
    def __init__(self, g_ch: int, x_ch: int):
        super().__init__()
        # map g -> x_ch
        self.g_bn = nn.BatchNorm2d(g_ch)
        self.g_conv = nn.Conv2d(g_ch, x_ch, kernel_size=3, padding=1, bias=False)
        self.g_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.x_bn = nn.BatchNorm2d(x_ch)
        self.x_conv = nn.Conv2d(x_ch, x_ch, kernel_size=3, padding=1, bias=False)

        self.gc_bn = nn.BatchNorm2d(x_ch)
        self.gc_conv = nn.Conv2d(x_ch, x_ch, kernel_size=3, padding=1, bias=False)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = F.relu(self.g_bn(g), inplace=True)
        g1 = self.g_conv(g1)
        g1 = self.g_pool(g1)  # downsample to match x

        x1 = F.relu(self.x_bn(x), inplace=True)
        x1 = self.x_conv(x1)

        s = g1 + x1
        s = F.relu(self.gc_bn(s), inplace=True)
        s = self.gc_conv(s)

        return s * x


# -----------------------------
# ResUNet++ (2D)
# -----------------------------
class ResUNetPlusPlus2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        n_filters: List[int] = [8, 16, 32, 64, 128],
        se_ratio: int = 8,
        out_activation: str = "tanh",  # "tanh" or "sigmoid" or "none"
    ):
        super().__init__()
        f0, f1, f2, f3, f4 = n_filters

        # Stem
        self.c1 = StemBlock2D(in_channels, f0, stride=1, se_ratio=se_ratio)

        # Encoder
        self.c2 = ResNetBlock2D(f0, f1, stride=2, se_ratio=se_ratio)
        self.c3 = ResNetBlock2D(f1, f2, stride=2, se_ratio=se_ratio)
        self.c4 = ResNetBlock2D(f2, f3, stride=2, se_ratio=se_ratio)

        # Bridge
        self.b1 = ASPPBlock2D(f3, f4)

        # Decoder
        self.att1 = AttentionBlock2D(g_ch=f2, x_ch=f4)  # g=c3, x=b1
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d1 = ResNetBlock2D(f4 + f2, f3, stride=1, se_ratio=se_ratio)

        self.att2 = AttentionBlock2D(g_ch=f1, x_ch=f3)  # g=c2, x=d1
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d2 = ResNetBlock2D(f3 + f1, f2, stride=1, se_ratio=se_ratio)

        self.att3 = AttentionBlock2D(g_ch=f0, x_ch=f2)  # g=c1, x=d2
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d3 = ResNetBlock2D(f2 + f0, f1, stride=1, se_ratio=se_ratio)

        # Output head
        self.out_aspp = ASPPBlock2D(f1, f0)
        self.out_conv = nn.Conv2d(f0, out_channels, kernel_size=1)

        self.out_activation = out_activation.lower().strip()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        c1 = self.c1(x)     # (B,f0,H,W)
        c2 = self.c2(c1)    # (B,f1,H/2,W/2)
        c3 = self.c3(c2)    # (B,f2,H/4,W/4)
        c4 = self.c4(c3)    # (B,f3,H/8,W/8)

        # Bridge
        b1 = self.b1(c4)    # (B,f4,H/8,W/8)

        # Decoder 1
        a1 = self.att1(c3, b1)   # gated b1
        u1 = self.up1(a1)        # -> H/4,W/4
        u1 = torch.cat([u1, c3], dim=1)
        d1 = self.d1(u1)         # (B,f3,H/4,W/4)

        # Decoder 2
        a2 = self.att2(c2, d1)   # pool(c2)->H/4,W/4 matches d1
        u2 = self.up2(a2)        # -> H/2,W/2
        u2 = torch.cat([u2, c2], dim=1)
        d2 = self.d2(u2)         # (B,f2,H/2,W/2)

        # Decoder 3
        a3 = self.att3(c1, d2)   # pool(c1)->H/2,W/2 matches d2
        u3 = self.up3(a3)        # -> H,W
        u3 = torch.cat([u3, c1], dim=1)
        d3 = self.d3(u3)         # (B,f1,H,W)

        # Output
        y = self.out_aspp(d3)
        y = self.out_conv(y)

        if self.out_activation == "tanh":
            y = torch.tanh(y)
        elif self.out_activation == "sigmoid":
            y = torch.sigmoid(y)
        elif self.out_activation in ("none", "linear", ""):
            pass
        else:
            raise ValueError("out_activation must be: 'tanh', 'sigmoid', or 'none'")

        return y


# -----------------------------
# Quick demo usage (2D)
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Example: 2D image-to-image
    # Input channels: 1 (filtered grayscale) or 3 if you stack features
    model = ResUNetPlusPlus2D(in_channels=1, out_channels=1, out_activation="tanh").to(device)

    # Dummy batch: (B,C,H,W)
    x = torch.randn(4, 1, 128, 128, device=device)
    y = model(x)
    print("Output shape:", tuple(y.shape))  # (4,1,128,128)

    # Simple training step demo (regression with MSE)
    target = torch.randn_like(y)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    loss_fn = nn.MSELoss()

    model.train()
    opt.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, target)
    loss.backward()
    opt.step()
    print("Loss:", float(loss))

# -----------------------------
# Building blocks (3D)
# -----------------------------
"""
PyTorch ResUNet++-style (3D) demo: volume-to-volume (e.g., 64x64x64)
- 3D adaptation of your TF3D model:
  stem_block + resnet_blocks + ASPP bridge + attention gates + decoder
- Input:  (B, C_in, D, H, W)
- Output: (B, C_out, D, H, W) with tanh/sigmoid/linear

Dependencies:
  pip install torch

Notes:
- This is a faithful structural translation to 3D (Conv3d/Pool3d/Upsample).
- Attention block follows your TF logic: pool(g) to match x scale, fuse, then gate x.
- ASPP is sum of dilated convs + 1x1 conv (matching Add + Conv).
"""



# -----------------------------
# Building blocks (3D)
# -----------------------------
class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation for 3D feature maps."""
    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        hidden = max(channels // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, hidden, kernel_size=1, bias=False)
        self.fc2 = nn.Conv3d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.avg_pool(x)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class StemBlock3D(nn.Module):
    """
    TF stem_block:
      Conv -> BN -> ReLU -> Conv
      shortcut 1x1 conv
      Add -> SE
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, se_ratio: int = 8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.short = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False)
        self.short_bn = nn.BatchNorm3d(out_ch)

        self.se = SEBlock3D(out_ch, ratio=se_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_init = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.conv2(x)

        s = self.short_bn(self.short(x_init))
        x = x + s
        x = self.se(x)
        return x


class ResNetBlock3D(nn.Module):
    """
    TF resnet_block:
      BN -> ReLU -> Conv(stride)
      BN -> ReLU -> Conv
      shortcut 1x1 conv(stride)
      Add -> SE
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, se_ratio: int = 8):
        super().__init__()
        self.bn1 = nn.BatchNorm3d(in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.short = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False)
        self.short_bn = nn.BatchNorm3d(out_ch)

        self.se = SEBlock3D(out_ch, ratio=se_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_init = x

        x = F.relu(self.bn1(x), inplace=True)
        x = self.conv1(x)

        x = F.relu(self.bn2(x), inplace=True)
        x = self.conv2(x)

        s = self.short_bn(self.short(x_init))
        x = x + s
        x = self.se(x)
        return x


class ASPPBlock3D(nn.Module):
    """
    TF aspp_block:
      x1(dil=2) + x2(dil=4) + x3(dil=6) + x4(dil=1) -> Add
      -> 1x1 conv
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.c1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm3d(out_ch),
        )
        self.c2 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm3d(out_ch),
        )
        self.c3 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm3d(out_ch),
        )
        self.c4 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm3d(out_ch),
        )
        self.proj = nn.Conv3d(out_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.c1(x) + self.c2(x) + self.c3(x) + self.c4(x)
        y = self.proj(y)
        return y


class AttentionBlock3D(nn.Module):
    """
    3D adaptation of your TF attention_block(g, x):

    TF steps:
      g -> BN -> ReLU -> Conv
      g_pool = MaxPool(stride=2)
      x -> BN -> ReLU -> Conv
      sum -> BN -> ReLU -> Conv
      multiply with x

    In decoder call:
      d = attention_block(skip, decoder_feature)
      d = upsample(d) then concat with skip
    """
    def __init__(self, g_ch: int, x_ch: int):
        super().__init__()
        self.g_bn = nn.BatchNorm3d(g_ch)
        self.g_conv = nn.Conv3d(g_ch, x_ch, kernel_size=3, padding=1, bias=False)
        self.g_pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.x_bn = nn.BatchNorm3d(x_ch)
        self.x_conv = nn.Conv3d(x_ch, x_ch, kernel_size=3, padding=1, bias=False)

        self.gc_bn = nn.BatchNorm3d(x_ch)
        self.gc_conv = nn.Conv3d(x_ch, x_ch, kernel_size=3, padding=1, bias=False)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = F.relu(self.g_bn(g), inplace=True)
        g1 = self.g_conv(g1)
        g1 = self.g_pool(g1)  # downsample to match x

        x1 = F.relu(self.x_bn(x), inplace=True)
        x1 = self.x_conv(x1)

        s = g1 + x1
        s = F.relu(self.gc_bn(s), inplace=True)
        s = self.gc_conv(s)

        return s * x


# -----------------------------
# ResUNet++ (3D)
# -----------------------------
class ResUNetPlusPlus3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        n_filters: List[int] = [8, 16, 32, 64, 128],
        se_ratio: int = 8,
        out_activation: str = "tanh",  # "tanh" | "sigmoid" | "none"
    ):
        super().__init__()
        f0, f1, f2, f3, f4 = n_filters

        # Stem
        self.c1 = StemBlock3D(in_channels, f0, stride=1, se_ratio=se_ratio)

        # Encoder
        self.c2 = ResNetBlock3D(f0, f1, stride=2, se_ratio=se_ratio)
        self.c3 = ResNetBlock3D(f1, f2, stride=2, se_ratio=se_ratio)
        self.c4 = ResNetBlock3D(f2, f3, stride=2, se_ratio=se_ratio)

        # Bridge
        self.b1 = ASPPBlock3D(f3, f4)

        # Decoder
        self.att1 = AttentionBlock3D(g_ch=f2, x_ch=f4)  # g=c3, x=b1
        self.up1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.d1 = ResNetBlock3D(f4 + f2, f3, stride=1, se_ratio=se_ratio)

        self.att2 = AttentionBlock3D(g_ch=f1, x_ch=f3)  # g=c2, x=d1
        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.d2 = ResNetBlock3D(f3 + f1, f2, stride=1, se_ratio=se_ratio)

        self.att3 = AttentionBlock3D(g_ch=f0, x_ch=f2)  # g=c1, x=d2
        self.up3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.d3 = ResNetBlock3D(f2 + f0, f1, stride=1, se_ratio=se_ratio)

        # Output head
        self.out_aspp = ASPPBlock3D(f1, f0)
        self.out_conv = nn.Conv3d(f0, out_channels, kernel_size=1)

        self.out_activation = out_activation.lower().strip()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        c1 = self.c1(x)     # (B,f0,D,H,W)
        c2 = self.c2(c1)    # (B,f1,D/2,H/2,W/2)
        c3 = self.c3(c2)    # (B,f2,D/4,H/4,W/4)
        c4 = self.c4(c3)    # (B,f3,D/8,H/8,W/8)

        # Bridge
        b1 = self.b1(c4)    # (B,f4,D/8,H/8,W/8)

        # Decoder 1
        a1 = self.att1(c3, b1)   # gated b1
        u1 = self.up1(a1)        # -> D/4,H/4,W/4
        u1 = torch.cat([u1, c3], dim=1)
        d1 = self.d1(u1)         # (B,f3,D/4,H/4,W/4)

        # Decoder 2
        a2 = self.att2(c2, d1)   # pool(c2)->D/4,H/4,W/4 matches d1
        u2 = self.up2(a2)        # -> D/2,H/2,W/2
        u2 = torch.cat([u2, c2], dim=1)
        d2 = self.d2(u2)         # (B,f2,D/2,H/2,W/2)

        # Decoder 3
        a3 = self.att3(c1, d2)   # pool(c1)->D/2,H/2,W/2 matches d2
        u3 = self.up3(a3)        # -> D,H,W
        u3 = torch.cat([u3, c1], dim=1)
        d3 = self.d3(u3)         # (B,f1,D,H,W)

        # Output
        y = self.out_aspp(d3)
        y = self.out_conv(y)

        if self.out_activation == "tanh":
            y = torch.tanh(y)
        elif self.out_activation == "sigmoid":
            y = torch.sigmoid(y)
        elif self.out_activation in ("none", "linear", ""):
            pass
        else:
            raise ValueError("out_activation must be: 'tanh', 'sigmoid', or 'none'")

        return y


# -----------------------------
# Quick demo usage (3D)
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # For 3D volumes like 64x64x64
    model = ResUNetPlusPlus3D(in_channels=1, out_channels=1, out_activation="tanh").to(device)

    x = torch.randn(2, 1, 64, 64, 64, device=device)
    y = model(x)
    print("Output shape:", tuple(y.shape))  # (2,1,64,64,64)

    # Simple training step demo (regression with MSE)
    target = torch.randn_like(y)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    loss_fn = nn.MSELoss()

    model.train()
    opt.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, target)
    loss.backward()
    opt.step()
    print("Loss:", float(loss))