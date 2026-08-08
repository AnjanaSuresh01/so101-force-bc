"""The deployment path: policy -> admittance -> follower arm.

This is the piece the CV line calls "deployed behind a force-limited admittance
controller, bounding commanded contact regardless of what the policy predicts",
and it is written so that the "regardless" is literal. The policy's output never
reaches the arm. What reaches the arm is:

    q_policy  --FK-->  Cartesian tool target
                       |
                       admittance + reference governor, on the force estimate
                       |
                       admissible tool target  --IK-->  q_command

Because the guard re-derives the joint command from a Cartesian reference it has
authority over, a policy that predicts a joint configuration 5 cm inside the
fixture cannot express it. The one thing that passes through untouched is the
gripper, which has no contact authority in these tasks.

The cost is real and is measured rather than waved at: an IK solve per control
tick, and the loss of any tool orientation the policy asked for beyond pitch --
which the SO-101's five joints could not have produced anyway (see
griff.kinematics).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from griff.control.admittance import AdmittanceConfig, AdmittanceController
from griff.kinematics import forward_kinematics, solve_ik
from griff.sim.env import TaskEnv


@dataclass
class GuardStats:
    ticks: int = 0
    governed_ticks: int = 0
    ik_failures: int = 0
    max_correction_mm: float = 0.0

    @property
    def governed_fraction(self) -> float:
        return self.governed_ticks / max(self.ticks, 1)


class ForceGuard:
    """Wraps a joint-space command in Cartesian force-limited admittance."""

    def __init__(self, env: TaskEnv, controller: AdmittanceController | None = None) -> None:
        self.env = env
        self.controller = controller or AdmittanceController(
            AdmittanceConfig(force_limit=env.spec.force_limit)
        )
        self.stats = GuardStats()
        self._last_q = env.joint_positions.copy()

    def reset(self) -> None:
        self.controller.reset(self.env.tool_point())
        self.stats = GuardStats()
        self._last_q = self.env.joint_positions.copy()

    def apply(self, joint_action: np.ndarray, force: np.ndarray) -> np.ndarray:
        env = self.env
        requested = np.asarray(joint_action, dtype=float)
        target, pitch = forward_kinematics(env.model, requested)
        reference = self.controller.step(target, np.asarray(force, dtype=float))

        result = solve_ik(env.model, env.data, reference, pitch, self._last_q)
        self.stats.ticks += 1
        self.stats.governed_ticks += int(self.controller.state.governed)
        self.stats.max_correction_mm = max(
            self.stats.max_correction_mm, float(np.linalg.norm(reference - target) * 1000)
        )

        if not result.converged:
            # The admissible reference is unreachable. Holding the previous
            # command is the safe failure: the alternative is issuing a partial
            # IK solve, which points the arm at a joint limit at speed.
            self.stats.ik_failures += 1
            command = self._last_q.copy()
        else:
            command = result.q.copy()
            self._last_q = command.copy()

        command[5] = requested[5]
        return command
