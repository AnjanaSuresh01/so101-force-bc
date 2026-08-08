"""Shared observation encoder for both policies.

Deliberately small. These are 96x96 images of a bench with one workpiece on it,
recorded from fixed viewpoints, and there are a few thousand frames per task. A
ResNet-18 here would be almost entirely untrained parameters, and CPU-only
training of the ablation grid would take a day instead of an hour.

The vision trunk ends in a spatial soft-argmax rather than global average
pooling. For manipulation that is not a stylistic choice: average pooling
discards *where* things are, which is the only thing either camera is being
asked. Soft-argmax returns the expected (x, y) of each channel's activation, so
the representation is a set of keypoint coordinates -- the fixture's edge, the
peg's shadow, the pad's contact patch.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from griff.policies.config import PolicyConfig


class SpatialSoftmax(nn.Module):
    """Expected pixel coordinate of each channel's activation."""

    # Declared so the registered buffers type as Tensors rather than
    # `Tensor | Module`, which is what nn.Module.__getattr__ returns.
    grid_x: Tensor
    grid_y: Tensor

    def __init__(self, height: int, width: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        xs = torch.linspace(-1.0, 1.0, width)
        ys = torch.linspace(-1.0, 1.0, height)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("grid_x", grid_x.reshape(1, 1, -1), persistent=False)
        self.register_buffer("grid_y", grid_y.reshape(1, 1, -1), persistent=False)

    def forward(self, features: Tensor) -> Tensor:
        batch, channels, height, width = features.shape
        flat = features.reshape(batch, channels, height * width) / self.temperature
        weights = torch.softmax(flat, dim=-1)
        x = (weights * self.grid_x).sum(-1)
        y = (weights * self.grid_y).sum(-1)
        return torch.cat([x, y], dim=-1)  # (batch, 2 * channels)


class VisionTrunk(nn.Module):
    """Strided conv stack shared in architecture (not weights) by both cameras."""

    def __init__(self, channels: tuple[int, ...], image_size: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 3
        size = image_size
        for index, out_channels in enumerate(channels):
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=5 if index == 0 else 3,
                          stride=2, padding=2 if index == 0 else 1),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(),
            ]
            in_channels = out_channels
            size = (size + 1) // 2
        self.body = nn.Sequential(*layers)
        self.out_size = size
        self.out_channels = in_channels

    def forward(self, images: Tensor) -> Tensor:
        return self.body(images)


class ObservationEncoder(nn.Module):
    """Images (+ state, + optionally force) -> a sequence of conditioning tokens.

    Each camera contributes a pooled grid of tokens plus its keypoint vector;
    state and force contribute one token each. Returning tokens rather than one
    flat vector is what lets ACT attend over them; the diffusion policy just
    flattens the sequence back down, which costs nothing and keeps one encoder
    serving both.
    """

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.trunks = nn.ModuleDict(
            {
                camera: VisionTrunk(config.vision_channels, config.image_size)
                for camera in config.cameras
            }
        )
        trunk = next(iter(self.trunks.values()))
        self.grid = int(round(config.tokens_per_camera**0.5))
        self.pool = nn.AdaptiveAvgPool2d(self.grid)
        self.token_proj = nn.ModuleDict(
            {
                camera: nn.Linear(trunk.out_channels, config.hidden_dim)
                for camera in config.cameras
            }
        )
        self.keypoints = SpatialSoftmax(trunk.out_size, trunk.out_size)
        self.keypoint_proj = nn.ModuleDict(
            {
                camera: nn.Linear(2 * trunk.out_channels, config.hidden_dim)
                for camera in config.cameras
            }
        )
        self.state_proj = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.force_proj = (
            nn.Sequential(
                nn.Linear(config.force_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            if config.uses_force
            else None
        )

    @property
    def num_tokens(self) -> int:
        per_camera = self.grid * self.grid + 1
        return len(self.config.cameras) * per_camera + 1 + (1 if self.config.uses_force else 0)

    def forward(self, images: dict[str, Tensor], state: Tensor, force: Tensor) -> Tensor:
        tokens: list[Tensor] = []
        for camera in self.config.cameras:
            features = self.trunks[camera](images[camera])
            pooled = self.pool(features).flatten(2).transpose(1, 2)  # (B, grid^2, C)
            tokens.append(self.token_proj[camera](pooled))
            tokens.append(self.keypoint_proj[camera](self.keypoints(features)).unsqueeze(1))
        tokens.append(self.state_proj(state).unsqueeze(1))
        if self.force_proj is not None:
            # The zero-force control arm keeps the branch and its parameters and
            # replaces its input with zeros, so capacity is held fixed and only
            # the information differs. Zeroing here rather than at the call site
            # means no evaluation path can accidentally feed it a real force.
            if self.config.force_is_blinded:
                force = torch.zeros_like(force)
            tokens.append(self.force_proj(force).unsqueeze(1))
        return torch.cat(tokens, dim=1)


class Normaliser(nn.Module):
    """Per-channel standardisation with statistics baked into the checkpoint.

    Kept as buffers rather than recomputed at load time so a checkpoint is
    self-contained: evaluating a policy must not depend on still having the
    dataset it was trained on.
    """

    mean: Tensor
    std: Tensor

    def __init__(self, mean: Tensor, std: Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean.float())
        # A joint that never moves in the demonstrations has zero variance, and
        # dividing by it produces NaNs several thousand steps into training,
        # long after the run looked healthy.
        self.register_buffer("std", std.float().clamp_min(1e-4))

    def normalise(self, values: Tensor) -> Tensor:
        return (values - self.mean) / self.std

    def denormalise(self, values: Tensor) -> Tensor:
        return values * self.std + self.mean
