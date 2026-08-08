"""Diffusion Policy -- a conditional 1D UNet denoising an action chunk.

Follows Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action
Diffusion" (2023): the policy models the *distribution* over action chunks by
learning to denoise them, conditioned on the observation through FiLM.

The reason it is worth having alongside ACT in this repo is that the two fail
differently on contact tasks, and the evaluation is meant to show that rather
than assert it. ACT's CVAE handles multimodality through a single latent draw
per chunk; diffusion handles it by construction, which tends to matter most
exactly where the demonstrations disagree -- which way to nudge a peg that has
caught on the chamfer.

Training uses DDPM over `diffusion_steps`; inference uses DDIM over
`inference_steps` (10 by default). That is a 10x saving in the rollout loop and
it is not optional here: the evaluation harness runs hundreds of episodes on a
CPU, and 100 network evaluations per 30 Hz control tick would put a single
rollout in the tens of minutes.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from griff.policies.config import PolicyConfig
from griff.policies.encoders import Normaliser, ObservationEncoder


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
    """Nichol & Dhariwal's squared-cosine schedule.

    Preferred over linear betas at small step counts: a linear schedule spends
    most of its budget where the signal is already destroyed, which shows up as
    a policy that reproduces the mean trajectory and nothing else.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-8, 0.999)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1)
        )
        angles = t.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class ConditionalResidualBlock1D(nn.Module):
    """Two conv blocks with FiLM conditioning between them."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, groups: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.film = nn.Linear(cond_dim, out_channels * 2)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = torch.nn.functional.silu(self.norm1(self.conv1(x)))
        scale, shift = self.film(cond).unsqueeze(-1).chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = torch.nn.functional.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class ConditionalUNet1D(nn.Module):
    """Small UNet over the action horizon."""

    def __init__(self, action_dim: int, horizon: int, cond_dim: int, channels: tuple[int, ...]) -> None:
        super().__init__()
        self.horizon = horizon
        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        in_channels = action_dim
        for out_channels in channels:
            self.down.append(ConditionalResidualBlock1D(in_channels, out_channels, cond_dim))
            self.downsample.append(nn.Conv1d(out_channels, out_channels, 3, stride=2, padding=1))
            in_channels = out_channels

        self.mid = ConditionalResidualBlock1D(in_channels, in_channels, cond_dim)

        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for out_channels in reversed(channels):
            self.upsample.append(
                nn.ConvTranspose1d(in_channels, out_channels, 4, stride=2, padding=1)
            )
            # Input is the upsampled path concatenated with the matching skip.
            self.up.append(
                ConditionalResidualBlock1D(out_channels * 2, out_channels, cond_dim)
            )
            in_channels = out_channels
        self.final = nn.Conv1d(in_channels, action_dim, 1)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        skips: list[Tensor] = []
        h = x
        for block, down in zip(self.down, self.downsample, strict=True):
            h = block(h, cond)
            skips.append(h)
            h = down(h)
        h = self.mid(h, cond)
        for block, up, skip in zip(self.up, self.upsample, reversed(skips), strict=True):
            h = up(h)
            if h.shape[-1] != skip.shape[-1]:
                h = torch.nn.functional.interpolate(h, size=skip.shape[-1], mode="nearest")
            h = block(torch.cat([h, skip], dim=1), cond)
        return self.final(h)


class DiffusionPolicy(nn.Module):
    betas: Tensor
    alphas_cumprod: Tensor
    sqrt_alphas_cumprod: Tensor
    sqrt_one_minus_alphas_cumprod: Tensor

    def __init__(
        self,
        config: PolicyConfig,
        state_norm: Normaliser,
        force_norm: Normaliser,
        action_norm: Normaliser,
    ) -> None:
        super().__init__()
        self.config = config
        self.state_norm = state_norm
        self.force_norm = force_norm
        self.action_norm = action_norm

        self.observation_encoder = ObservationEncoder(config)
        observation_dim = self.observation_encoder.num_tokens * config.hidden_dim
        self.observation_proj = nn.Sequential(
            nn.Linear(observation_dim, config.hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2),
        )
        self.time_embedding = nn.Sequential(
            SinusoidalEmbedding(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2),
        )
        self.unet = ConditionalUNet1D(
            config.action_dim, config.chunk, config.hidden_dim * 2, config.unet_channels
        )

        betas = cosine_beta_schedule(config.diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt(), persistent=False)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt(), persistent=False
        )

    def _condition(self, images: dict[str, Tensor], state: Tensor, force: Tensor) -> Tensor:
        tokens = self.observation_encoder(images, state, force)
        return self.observation_proj(tokens.flatten(1))

    def loss(
        self,
        images: dict[str, Tensor],
        state: Tensor,
        force: Tensor,
        actions: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        state = self.state_norm.normalise(state)
        force = self.force_norm.normalise(force)
        target = self.action_norm.normalise(actions).transpose(1, 2)  # (B, action, horizon)

        batch = target.shape[0]
        timesteps = torch.randint(0, self.config.diffusion_steps, (batch,), device=target.device)
        noise = torch.randn_like(target)
        noisy = (
            self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1) * target
            + self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1) * noise
        )

        cond = self._condition(images, state, force) + self.time_embedding(timesteps)
        predicted = self.unet(noisy, cond)
        weights = mask.unsqueeze(1)
        mse = (((predicted - noise) ** 2) * weights).sum() / (
            weights.sum() * target.shape[1]
        ).clamp_min(1.0)
        return mse, {"mse": float(mse.detach())}

    @torch.no_grad()
    def predict_chunk(self, images: dict[str, Tensor], state: Tensor, force: Tensor) -> Tensor:
        """DDIM sampling, deterministic (eta = 0)."""
        state = self.state_norm.normalise(state)
        force = self.force_norm.normalise(force)
        observation = self._condition(images, state, force)

        batch = state.shape[0]
        sample = torch.randn(
            batch, self.config.action_dim, self.config.chunk, device=state.device
        )
        schedule = torch.linspace(
            self.config.diffusion_steps - 1, 0, self.config.inference_steps
        ).long()
        for index, step in enumerate(schedule):
            timesteps = torch.full((batch,), int(step), device=state.device, dtype=torch.long)
            cond = observation + self.time_embedding(timesteps)
            noise = self.unet(sample, cond)

            alpha_bar = self.alphas_cumprod[step]
            x0 = (sample - (1 - alpha_bar).sqrt() * noise) / alpha_bar.sqrt()
            x0 = x0.clamp(-4.0, 4.0)
            if index + 1 < len(schedule):
                alpha_bar_prev = self.alphas_cumprod[schedule[index + 1]]
                sample = alpha_bar_prev.sqrt() * x0 + (1 - alpha_bar_prev).sqrt() * noise
            else:
                sample = x0
        return self.action_norm.denormalise(sample.transpose(1, 2))
