from __future__ import annotations

import torch
from torch import nn


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padding = (self.kernel_size - 1) // 2
        transposed = values.transpose(1, 2)
        left = transposed[:, :, :1].expand(-1, -1, padding)
        right = transposed[:, :, -1:].expand(-1, -1, padding)
        trend = self.pool(torch.cat([left, transposed, right], dim=-1)).transpose(1, 2)
        return values - trend, trend


class DLinear(nn.Module):
    """DLinear decomposition with an explicit track-level channel fusion head."""

    def __init__(self, context_length: int, horizon: int, channels: int, kernel_size: int = 25) -> None:
        super().__init__()
        self.decomposition = SeriesDecomposition(kernel_size)
        self.seasonal = nn.Linear(context_length, horizon)
        self.trend = nn.Linear(context_length, horizon)
        self.channel_fusion = nn.Linear(channels, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decomposition(values)
        projected = self.seasonal(seasonal.transpose(1, 2)) + self.trend(trend.transpose(1, 2))
        return self.channel_fusion(projected.transpose(1, 2)).squeeze(-1)


class PatchTST(nn.Module):
    """Channel-independent patch encoder with a declared channel-fusion forecast head."""

    def __init__(
        self,
        context_length: int,
        horizon: int,
        channels: int,
        patch_length: int = 16,
        stride: int = 8,
        d_model: int = 64,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if context_length < patch_length:
            raise ValueError("context_length must be at least patch_length")
        self.patch_length = patch_length
        self.stride = stride
        self.patch_count = 1 + (context_length - patch_length) // stride
        self.patch_projection = nn.Linear(patch_length, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.patch_count, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.head = nn.Linear(channels * self.patch_count * d_model, horizon)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=1, keepdim=True)
        scale = values.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        normalized = (values - mean) / scale
        patches = normalized.transpose(1, 2).unfold(-1, self.patch_length, self.stride)
        batch, channels, patch_count, patch_length = patches.shape
        encoded = self.patch_projection(patches.reshape(batch * channels, patch_count, patch_length))
        encoded = self.encoder(encoded + self.position)
        forecast = self.head(encoded.reshape(batch, channels * patch_count * encoded.shape[-1]))
        return forecast * scale[:, 0, 0:1] + mean[:, 0, 0:1]
