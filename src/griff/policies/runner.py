"""Running a trained policy at 30 Hz, and loading one from a checkpoint.

Both policies predict a chunk of `config.chunk` actions. How that chunk becomes
one command per tick differs, and the difference is not cosmetic:

* ACT uses temporal ensembling -- re-plan every tick, execute a weighted average
  over all chunks that covered this tick.
* Diffusion Policy uses receding horizon -- re-plan every `config.execute`
  ticks, execute the chunk in order. Denoising is far too expensive to run every
  tick, which is the reason the original does this too.

Both are what their papers specify. It does mean the two policies re-plan at
different rates, which is a real confound for any comparison *between* them --
stated here and in the README rather than left for a reader to notice. The
force ablation, which is the comparison this repo is actually built around, is
unaffected: it holds the policy and its inference scheme fixed and changes only
whether force is in the observation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from griff.policies.act import ACTPolicy, TemporalEnsemble
from griff.policies.config import PolicyConfig
from griff.policies.diffusion import DiffusionPolicy
from griff.policies.encoders import Normaliser


def build_policy(
    config: PolicyConfig,
    state: Normaliser,
    force: Normaliser,
    action: Normaliser,
) -> ACTPolicy | DiffusionPolicy:
    if config.kind == "act":
        return ACTPolicy(config, state, force, action)
    if config.kind == "diffusion":
        return DiffusionPolicy(config, state, force, action)
    raise ValueError(f"unknown policy kind {config.kind!r}")


class PolicyRunner:
    """Wraps a trained model into something the evaluation loop can call."""

    def __init__(self, model: ACTPolicy | DiffusionPolicy, config: PolicyConfig) -> None:
        self.model = model.eval()
        self.config = config
        self._ensemble = (
            TemporalEnsemble(config.chunk, config.action_dim) if config.kind == "act" else None
        )
        self._pending: np.ndarray | None = None
        self._cursor = 0

    def reset(self) -> None:
        if self._ensemble is not None:
            self._ensemble.reset()
        self._pending = None
        self._cursor = 0

    def _tensors(self, observation) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        images = {
            camera: torch.from_numpy(observation.images[camera])
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            for camera in self.config.cameras
        }
        state = torch.from_numpy(observation.state.astype(np.float32)).unsqueeze(0)
        force = torch.from_numpy(observation.force.astype(np.float32)).unsqueeze(0)
        return images, state, force

    @torch.no_grad()
    def act(self, observation) -> np.ndarray:
        if self._ensemble is not None:
            images, state, force = self._tensors(observation)
            chunk = self.model.predict_chunk(images, state, force)[0].numpy()
            return self._ensemble.add(chunk)

        if self._pending is None or self._cursor >= self.config.execute:
            images, state, force = self._tensors(observation)
            self._pending = self.model.predict_chunk(images, state, force)[0].numpy()
            self._cursor = 0
        action = self._pending[self._cursor]
        self._cursor += 1
        return action.copy()


def save_checkpoint(
    path: str | Path,
    model: ACTPolicy | DiffusionPolicy,
    config: PolicyConfig,
    metrics: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config.to_dict(),
            "state_dict": model.state_dict(),
            "normalisers": {
                "state": (model.state_norm.mean, model.state_norm.std),
                "force": (model.force_norm.mean, model.force_norm.std),
                "action": (model.action_norm.mean, model.action_norm.std),
            },
            "metrics": metrics or {},
        },
        path,
    )


def load_policy(path: str | Path) -> PolicyRunner:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = PolicyConfig.from_dict(payload["config"])
    normalisers = {
        key: Normaliser(mean, std) for key, (mean, std) in payload["normalisers"].items()
    }
    model = build_policy(config, normalisers["state"], normalisers["force"], normalisers["action"])
    model.load_state_dict(payload["state_dict"])
    return PolicyRunner(model, config)
