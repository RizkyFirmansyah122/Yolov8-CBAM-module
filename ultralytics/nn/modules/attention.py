from __future__ import annotations

import torch
import torch.nn as nn


# =====================================================
# Channel Attention (sudah benar, tidak diubah)
# =====================================================
class ChannelAttention(nn.Module):
    def __init__(self, c1: int | None = None, reduction: int = 16):
        super().__init__()
        self.reduction = reduction
        self._built = False
        if c1 is not None:
            self._build(c1)

    def _build(self, c1: int):
        hidden = max(1, c1 // self.reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(hidden, c1, 1, bias=False)
        )
        self._built = True

    def forward(self, x):
        if not self._built:
            self._build(x.shape[1])
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale = torch.sigmoid(avg_out + max_out)
        return x * scale


# =====================================================
# Spatial Attention (DIPERBAIKI)
# =====================================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Menggabungkan hasil avg dan max pool untuk membuat peta spasial 2 channel
        y = torch.cat([avg_out, max_out], dim=1)
        # Mengonvolusi peta spasial menjadi 1 channel
        y = self.conv(y)
        # Mengembalikan HANYA attention map (bukan dikalikan dengan input)
        return self.sigmoid(y)


# =====================================================
# CBAM (DIPERBAIKI)
# =====================================================
class CBAM(nn.Module):
    """CBAM (Channel + Spatial) dengan auto-build untuk YOLOv8 scaling."""

    def __init__(self, c1: int | None = None, reduction: int = 16, k: int = 7):
        super().__init__()
        self.reduction = reduction
        self.k = k
        self._built = False
        if c1 is not None:
            self._build(c1)

    def _build(self, c1: int):
        self.ca = ChannelAttention(c1, reduction=self.reduction)
        self.sa = SpatialAttention(kernel_size=self.k)
        self._built = True

    def forward(self, x):
        if not self._built:
            self._build(x.shape[1])

        # 1. Terapkan Channel Attention
        x_refined_by_channel = self.ca(x)

        # 2. Hasilkan Spatial Attention Map dari feature map yang sudah disempurnakan
        spatial_attention_map = self.sa(x_refined_by_channel)

        # 3. Kalikan feature map dari langkah 1 dengan spatial map dari langkah 2
        x_refined_finally = x_refined_by_channel * spatial_attention_map

        # 4. Kembalikan feature map akhir yang channel-nya tidak berubah
        return x_refined_finally
