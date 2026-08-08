"""ACT -- Action Chunking with Transformers, as a CVAE over action chunks.

Follows Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost
Hardware" (2023), at a size that suits a few thousand demonstration frames and a
CPU: a small conv trunk instead of ResNet-18, two encoder and two decoder layers
instead of four, and a 16-tick chunk.

The two pieces of ACT that actually matter for contact-rich work are both here:

* **Action chunking.** Predicting 16 ticks at once rather than one removes the
  per-step compounding that makes single-step behaviour cloning drift, and it
  lets the policy commit to a search motion that only pays off half a second
  later -- which is exactly the shape of "press, feel it catch, slide across,
  drop in".

* **Temporal ensembling.** At inference every tick produces a fresh chunk, and
  the action executed is an exponentially weighted average over all chunks that
  predicted this timestep. Without it the arm twitches every `execute` ticks as
  one chunk hands over to the next, and on a force-limited controller those
  discontinuities show up directly as contact-force spikes.

The CVAE latent is what lets one policy represent a multimodal demonstration
set -- two operators who search a stuck peg in opposite directions are not
averaged into a policy that presses straight down.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from griff.policies.config import PolicyConfig
from griff.policies.encoders import Normaliser, ObservationEncoder


class ACTPolicy(nn.Module):
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

        hidden = config.hidden_dim
        self.observation_encoder = ObservationEncoder(config)
        self.observation_pos = nn.Parameter(
            torch.randn(1, self.observation_encoder.num_tokens + 1, hidden) * 0.02
        )
        self.latent_proj = nn.Linear(config.latent_dim, hidden)

        # --- CVAE encoder: (state, action chunk) -> latent ---
        self.cvae_cls = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.cvae_state = nn.Linear(config.state_dim, hidden)
        self.cvae_action = nn.Linear(config.action_dim, hidden)
        self.cvae_pos = nn.Parameter(torch.randn(1, config.chunk + 2, hidden) * 0.02)
        self.cvae_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden, config.heads, config.feedforward_dim, batch_first=True, norm_first=True
            ),
            num_layers=config.encoder_layers,
        )
        self.latent_head = nn.Linear(hidden, 2 * config.latent_dim)

        # --- main encoder/decoder ---
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden, config.heads, config.feedforward_dim, batch_first=True, norm_first=True
            ),
            num_layers=config.encoder_layers,
        )
        self.query = nn.Parameter(torch.randn(1, config.chunk, hidden) * 0.02)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                hidden, config.heads, config.feedforward_dim, batch_first=True, norm_first=True
            ),
            num_layers=config.decoder_layers,
        )
        self.action_head = nn.Linear(hidden, config.action_dim)

    # ------------------------------------------------------------------ pieces

    def _memory(self, images: dict[str, Tensor], state: Tensor, force: Tensor, z: Tensor) -> Tensor:
        tokens = self.observation_encoder(images, state, force)
        tokens = torch.cat([self.latent_proj(z).unsqueeze(1), tokens], dim=1)
        return self.encoder(tokens + self.observation_pos)

    def encode_latent(
        self, state: Tensor, actions: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Actions + state -> the CVAE posterior.

        `mask` marks which action steps are real (1) and which are padding at
        the end of an episode (0). Padded steps are excluded from attention.
        Leaving them in is a subtle leak rather than an obvious bug: the L1 term
        is masked, so the loss looks correctly padded, but the latent -- and
        therefore every predicted action in the chunk -- is still computed from
        actions that never happened.
        """
        batch = state.shape[0]
        tokens = torch.cat(
            [
                self.cvae_cls.expand(batch, -1, -1),
                self.cvae_state(state).unsqueeze(1),
                self.cvae_action(actions),
            ],
            dim=1,
        )
        padding = None
        if mask is not None:
            prefix = torch.zeros(batch, 2, dtype=torch.bool, device=mask.device)
            padding = torch.cat([prefix, mask <= 0], dim=1)
        encoded = self.cvae_encoder(tokens + self.cvae_pos, src_key_padding_mask=padding)
        mu, log_var = self.latent_head(encoded[:, 0]).chunk(2, dim=-1)
        return mu, log_var.clamp(-8.0, 8.0)

    def forward(
        self,
        images: dict[str, Tensor],
        state: Tensor,
        force: Tensor,
        actions: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (predicted chunk, mu, log_var). mu/log_var are zeros at test time."""
        state = self.state_norm.normalise(state)
        force = self.force_norm.normalise(force)

        if actions is None:
            zeros = torch.zeros(
                state.shape[0], self.config.latent_dim, device=state.device, dtype=state.dtype
            )
            mu = log_var = zeros
            z = zeros
        else:
            mu, log_var = self.encode_latent(
                state, self.action_norm.normalise(actions), mask
            )
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

        memory = self._memory(images, state, force, z)
        decoded = self.decoder(self.query.expand(state.shape[0], -1, -1), memory)
        return self.action_head(decoded), mu, log_var

    def loss(
        self,
        images: dict[str, Tensor],
        state: Tensor,
        force: Tensor,
        actions: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        predicted, mu, log_var = self.forward(images, state, force, actions, mask)
        target = self.action_norm.normalise(actions)
        weights = mask.unsqueeze(-1)
        l1 = (torch.abs(predicted - target) * weights).sum() / weights.sum().clamp_min(1.0)
        kl = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(-1)).mean()
        total = l1 + self.config.kl_weight * kl
        return total, {"l1": float(l1.detach()), "kl": float(kl.detach())}

    # ------------------------------------------------------------------ acting

    @torch.no_grad()
    def predict_chunk(
        self, images: dict[str, Tensor], state: Tensor, force: Tensor
    ) -> Tensor:
        predicted, _, _ = self.forward(images, state, force, None)
        return self.action_norm.denormalise(predicted)


class TemporalEnsemble:
    """Exponentially weighted average over every chunk that covered this tick.

    `weight_decay` is ACT's `m`: larger means older predictions count for less.
    At m = 0.01 a chunk's vote decays by ~1% per tick of age, so the executed
    action is dominated by the most recent plan while still being smoothed by
    the ones before it.
    """

    def __init__(self, chunk: int, action_dim: int, weight_decay: float = 0.01) -> None:
        self.chunk = chunk
        self.action_dim = action_dim
        self.weight_decay = weight_decay
        self.reset()

    def reset(self) -> None:
        self._predictions: list[tuple[int, np.ndarray]] = []
        self._tick = 0

    def add(self, chunk: np.ndarray) -> np.ndarray:
        self._predictions.append((self._tick, chunk))
        self._predictions = [
            (start, actions)
            for start, actions in self._predictions
            if self._tick - start < self.chunk
        ]
        numerator = np.zeros(self.action_dim)
        denominator = 0.0
        for start, actions in self._predictions:
            age = self._tick - start
            weight = float(np.exp(-self.weight_decay * age))
            numerator += weight * actions[age]
            denominator += weight
        self._tick += 1
        return numerator / max(denominator, 1e-9)
