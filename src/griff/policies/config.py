"""Policy configuration, and the one axis the whole repo is built to compare.

`conditioning` is the experiment, and it has three arms rather than two:

    vision            images + joint state. No force branch at all.
    vision_zeroforce  images + joint state + a force branch that is fed a
                      constant zero. Byte-for-byte the same architecture and
                      parameter count as vision_force.
    vision_force      images + joint state + the contact-force estimate.

The middle arm is the one that makes this an experiment rather than a
demonstration. `vision_force` carries ~17k more parameters than `vision`
(1.6% of a 1.06M model), so a gain over `vision` alone could be capacity
rather than information. `vision_zeroforce` has exactly those parameters and
none of the information, so vision_force - vision_zeroforce isolates what the
force *signal* contributes, with capacity held fixed.

Everything else -- backbone, width, depth, chunk length, optimiser, seed, the
data itself -- is identical across the three.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Conditioning = Literal["vision", "vision_zeroforce", "vision_force"]
PolicyKind = Literal["act", "diffusion"]

CONDITIONINGS: tuple[Conditioning, ...] = ("vision", "vision_zeroforce", "vision_force")
POLICY_KINDS: tuple[PolicyKind, ...] = ("act", "diffusion")


@dataclass
class PolicyConfig:
    kind: PolicyKind = "act"
    conditioning: Conditioning = "vision_force"
    cameras: tuple[str, ...] = ("top", "wrist")
    image_size: int = 96
    state_dim: int = 6
    force_dim: int = 3
    action_dim: int = 6

    # --- shared trunk ---
    hidden_dim: int = 128
    vision_channels: tuple[int, ...] = (16, 32, 64)
    tokens_per_camera: int = 9  # 3x3 pooled grid of the final feature map

    # --- action chunking ---
    #: Predicted horizon. 16 ticks is 0.53 s at 30 Hz -- long enough to cover the
    #: search-and-slide motion in peg insertion, short enough that the policy is
    #: not asked to plan through a contact event it cannot see the end of.
    chunk: int = 16
    #: Executed before re-planning. ACT overrides this with temporal ensembling.
    execute: int = 8

    # --- ACT ---
    latent_dim: int = 16
    encoder_layers: int = 2
    decoder_layers: int = 2
    heads: int = 4
    feedforward_dim: int = 256
    kl_weight: float = 10.0

    # --- diffusion ---
    diffusion_steps: int = 100
    inference_steps: int = 10
    unet_channels: tuple[int, ...] = (64, 128)

    # --- optimisation ---
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    steps: int = 4000
    warmup_steps: int = 200
    seed: int = 0

    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def uses_force(self) -> bool:
        """Whether the model *has* a force branch (not whether it is informative)."""
        return self.conditioning in ("vision_force", "vision_zeroforce")

    @property
    def force_is_blinded(self) -> bool:
        """Whether that branch is fed a constant zero instead of the estimate."""
        return self.conditioning == "vision_zeroforce"

    @property
    def name(self) -> str:
        return f"{self.kind}-{self.conditioning}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cameras"] = list(self.cameras)
        payload["vision_channels"] = list(self.vision_channels)
        payload["unet_channels"] = list(self.unet_channels)
        return payload

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PolicyConfig:
        payload = dict(payload)
        for key in ("cameras", "vision_channels", "unet_channels"):
            if key in payload:
                payload[key] = tuple(payload[key])
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})

    @classmethod
    def load(cls, path: str | Path) -> PolicyConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))
