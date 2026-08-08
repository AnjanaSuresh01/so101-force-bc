"""Recording teleoperated episodes into a LeRobot dataset.

One frame per 30 Hz control tick, holding the observation the operator acted on
and the joint command they produced from it. That pairing is the whole content
of a behaviour-cloning dataset and it is easy to get subtly wrong: recording the
observation *after* stepping pairs each command with the state it caused rather
than the state that caused it, and trains a policy to predict the past.

Failed episodes are dropped by default. Demonstration datasets are meant to
demonstrate; a policy cloned from a mixture of successes and timeouts learns
the average of the two. The counts are reported either way, so the yield of the
teleoperation session is visible rather than implied.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from griff.calibrate import load_or_fit
from griff.data.lerobot import DatasetWriter, so101_features
from griff.kinematics import JOINT_NAMES
from griff.sim import make_env
from griff.teleop import make_operator


@dataclass
class RecordingSummary:
    task: str
    root: str
    episodes_recorded: int
    episodes_attempted: int
    episodes_discarded: int
    frames: int
    mean_episode_length: float
    mean_peak_force_n: float
    max_peak_force_n: float
    overload_episodes: int
    contact_fraction: float
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def record_dataset(
    task: str,
    *,
    episodes: int,
    root: str | Path,
    seed: int = 0,
    image_size: int = 96,
    cameras: tuple[str, ...] = ("top", "wrist"),
    keep_failures: bool = False,
    max_attempts_factor: int = 3,
    overwrite: bool = False,
) -> RecordingSummary:
    started = time.time()
    calibration = load_or_fit(task)
    env = make_env(
        task, image_size=image_size, cameras=cameras, calibration=calibration, seed=seed
    )
    operator = make_operator(task, np.random.default_rng(seed))
    writer = DatasetWriter(
        root,
        fps=30,
        features=so101_features(cameras, image_size, joint_names=list(JOINT_NAMES)),
        overwrite=overwrite,
    )

    recorded = attempted = discarded = frames = 0
    lengths: list[int] = []
    peaks: list[float] = []
    overloads = 0
    contact_ticks = total_ticks = 0

    while recorded < episodes and attempted < episodes * max_attempts_factor:
        episode_seed = seed * 100_000 + attempted
        observation = env.reset(seed=episode_seed)
        operator.reset(env, np.random.default_rng(episode_seed))
        attempted += 1

        rows: list[dict[str, Any]] = []
        success = False
        while True:
            action = operator.act(env, observation.force)
            row = {
                "observation.state": observation.state,
                "observation.force": observation.force,
                "action": action,
            }
            for camera, image in observation.images.items():
                row[f"observation.images.{camera}"] = image
            rows.append(row)

            result = env.step(action)
            observation = result.observation
            if result.done:
                success = result.success
                break

        if not success and not keep_failures:
            discarded += 1
            continue

        writer.start_episode(env.spec.description)
        for row in rows:
            writer.add_frame(row)
        writer.end_episode(
            {
                "seed": episode_seed,
                "success": bool(success),
                "steps": len(rows),
                "peak_force_estimated_n": float(env.peak_force),
                "peak_force_true_n": float(env.peak_force_true),
                "overloaded": bool(env.peak_force_true > env.spec.overload_force),
                "operator_speed": float(operator.style.speed),
                "operator_aim_error_mm": [float(v * 1000) for v in operator.style.aim_error],
                "home_retreat_mm": float(env._home_retreat_mm),
                **env.episode_parameters,
            }
        )
        recorded += 1
        frames += len(rows)
        lengths.append(len(rows))
        peaks.append(float(env.peak_force_true))
        overloads += int(env.peak_force_true > env.spec.overload_force)
        contact_ticks += int(sum(1 for r in rows if np.linalg.norm(r["observation.force"]) > 1.0))
        total_ticks += len(rows)

    env.close()
    if recorded < episodes:
        raise RuntimeError(
            f"{task}: only {recorded}/{episodes} episodes succeeded in "
            f"{attempted} attempts. The operator is failing more often than the "
            "attempt budget allows -- fix the operator rather than raising the budget."
        )
    writer.finalise()

    return RecordingSummary(
        task=task,
        root=str(root),
        episodes_recorded=recorded,
        episodes_attempted=attempted,
        episodes_discarded=discarded,
        frames=frames,
        mean_episode_length=float(np.mean(lengths)),
        mean_peak_force_n=float(np.mean(peaks)),
        max_peak_force_n=float(np.max(peaks)),
        overload_episodes=overloads,
        contact_fraction=contact_ticks / max(total_ticks, 1),
        seconds=time.time() - started,
    )
