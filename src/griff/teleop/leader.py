"""The leader half of the leader-follower rig.

An SO-101 teleoperation rig is two identical arms. The leader has its servo
torque disabled so a human can back-drive it; the follower mirrors the leader's
joint angles. There is no force feedback in either direction -- the operator
works from what they can see, which for contact-rich tasks means from a force
readout on screen alongside the camera views.

`JointCalibration` is the piece that makes two arms one system. The servos are
mounted at whatever angle the horn happened to seat at, so each joint needs a
zero offset and a direction sign before the leader's raw counts mean anything
on the follower. Getting a sign wrong is the classic first-day failure: the
follower mirrors one joint backwards and drives itself into the bench.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from griff.kinematics import JOINT_NAMES


class LeaderArm(Protocol):
    """Anything that can report six joint angles at the control rate."""

    def read_joints(self) -> np.ndarray: ...


@dataclass(frozen=True)
class JointCalibration:
    """Maps raw leader joint angles onto follower command angles."""

    offsets: np.ndarray  # (6,) rad, subtracted from the raw leader reading
    signs: np.ndarray  # (6,) +1 or -1
    lower: np.ndarray  # (6,) rad, follower safe limits
    upper: np.ndarray  # (6,) rad

    def __post_init__(self) -> None:
        for name in ("offsets", "signs", "lower", "upper"):
            if getattr(self, name).shape != (6,):
                raise ValueError(f"{name} must have shape (6,)")
        if not np.all(np.isin(self.signs, (-1.0, 1.0))):
            raise ValueError("signs must be exactly +1 or -1 per joint")
        if not np.all(self.lower < self.upper):
            raise ValueError("lower limits must be below upper limits")

    def apply(self, raw: np.ndarray) -> np.ndarray:
        """Leader reading -> follower command, clipped to the safe range."""
        mapped = self.signs * (np.asarray(raw, dtype=float) - self.offsets)
        return np.clip(mapped, self.lower, self.upper)

    @classmethod
    def identity(cls, lower: np.ndarray, upper: np.ndarray) -> JointCalibration:
        return cls(np.zeros(6), np.ones(6), np.asarray(lower, float), np.asarray(upper, float))

    def to_json(self) -> str:
        return json.dumps(
            {
                "joint_names": list(JOINT_NAMES),
                "offsets_rad": self.offsets.tolist(),
                "signs": self.signs.tolist(),
                "lower_rad": self.lower.tolist(),
                "upper_rad": self.upper.tolist(),
            },
            indent=2,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> JointCalibration:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls(
            offsets=np.asarray(payload["offsets_rad"], dtype=float),
            signs=np.asarray(payload["signs"], dtype=float),
            lower=np.asarray(payload["lower_rad"], dtype=float),
            upper=np.asarray(payload["upper_rad"], dtype=float),
        )


class FeetechLeader:
    """A physical SO-101 leader arm on a Feetech bus.

    NOT EXECUTED -- see griff.teleop.feetech. Construction disables servo torque
    so the arm is back-drivable, which is the one thing that must happen before
    a human touches it.
    """

    def __init__(self, bus: object, calibration: JointCalibration) -> None:
        from griff.teleop.feetech import FeetechBus

        if not isinstance(bus, FeetechBus):
            raise TypeError("FeetechLeader needs a FeetechBus")
        self.bus = bus
        self.calibration = calibration
        self.bus.set_torque(False)

    def read_joints(self) -> np.ndarray:
        return self.calibration.apply(self.bus.read_positions())


class ReplayLeader:
    """Replays a recorded leader trace. Used to re-run an episode exactly."""

    def __init__(self, trace: np.ndarray) -> None:
        self.trace = np.asarray(trace, dtype=float)
        if self.trace.ndim != 2 or self.trace.shape[1] != 6:
            raise ValueError(f"expected an (N, 6) trace, got {self.trace.shape}")
        self.index = 0

    def read_joints(self) -> np.ndarray:
        joints = self.trace[min(self.index, len(self.trace) - 1)]
        self.index += 1
        return joints.copy()
