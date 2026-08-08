"""Behaviour-cloning training loop.

One loop for both policies -- they differ in their loss, not in how they are
optimised, and holding the optimiser, schedule, batch size, step count and seed
identical across the grid is what makes the vision / vision+force comparison a
comparison rather than two separate experiments.

Determinism: the seed fixes torch, numpy and the sampler. Two runs of the same
config on the same machine produce the same checkpoint. Across machines they
will not, exactly -- CPU BLAS reduction order is not fixed -- so the results
this repo publishes are reported with per-seed spread rather than as a single
number.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from griff.paths import DATASETS, RUNS
from griff.policies.config import PolicyConfig
from griff.policies.dataset import ChunkDataset, collate
from griff.policies.runner import build_policy, save_checkpoint


@dataclass
class TrainingReport:
    task: str
    policy: str
    conditioning: str
    steps: int
    frames: int
    episodes: int
    parameters: int
    final_loss: float
    best_loss: float
    seconds: float
    seed: int
    loss_curve: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _lr_at(step: int, config: PolicyConfig) -> float:
    """Linear warmup, then cosine decay to a tenth of the peak."""
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    return config.learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def train(
    task: str,
    config: PolicyConfig,
    *,
    dataset_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    log_every: int = 200,
    progress: bool = True,
) -> TrainingReport:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.use_deterministic_algorithms(False)

    dataset_root = Path(dataset_root) if dataset_root else DATASETS / task
    output_dir = Path(output_dir) if output_dir else RUNS / task / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    data = ChunkDataset(dataset_root, config)
    normalisers = data.normalisers()
    model = build_policy(config, normalisers.state, normalisers.force, normalisers.action)
    parameters = sum(p.numel() for p in model.parameters())

    loader = DataLoader(
        data,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,  # the images are already in RAM; workers would only copy them
        drop_last=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    started = time.time()
    curve: list[float] = []
    running: list[float] = []
    best = float("inf")
    step = 0
    model.train()

    while step < config.steps:
        for batch in loader:
            if step >= config.steps:
                break
            for group in optimiser.param_groups:
                group["lr"] = _lr_at(step, config)

            loss, parts = model.loss(
                batch["images"], batch["state"], batch["force"], batch["actions"], batch["mask"]
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            value = float(loss.detach())
            if not math.isfinite(value):
                raise RuntimeError(
                    f"loss became {value} at step {step}; refusing to keep training a "
                    "diverged model into a checkpoint that would look fine on disk"
                )
            running.append(value)
            step += 1

            if step % log_every == 0 or step == config.steps:
                mean = float(np.mean(running[-log_every:]))
                curve.append(mean)
                best = min(best, mean)
                if progress:
                    detail = " ".join(f"{k}={v:.4f}" for k, v in parts.items())
                    print(
                        f"  [{task}/{config.name}] step {step:>5}/{config.steps} "
                        f"loss={mean:.4f} {detail} lr={_lr_at(step, config):.2e}",
                        flush=True,
                    )

    report = TrainingReport(
        task=task,
        policy=config.kind,
        conditioning=config.conditioning,
        steps=config.steps,
        frames=len(data),
        episodes=len(data.episode_bounds),
        parameters=parameters,
        final_loss=curve[-1] if curve else float("nan"),
        best_loss=best,
        seconds=time.time() - started,
        seed=config.seed,
        loss_curve=curve,
    )
    save_checkpoint(output_dir / "policy.pt", model, config, report.to_dict())
    config.save(output_dir / "config.json")
    (output_dir / "training.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    return report
