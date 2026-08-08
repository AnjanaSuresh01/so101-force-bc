"""The deployment path: policy -> admittance -> follower arm.

This is the piece the CV line calls "deployed behind a force-limited admittance
controller, bounding commanded contact regardless of what the policy predicts",
and it is written so that the "regardless" is literal. The policy's output never
reaches the arm. What reaches the arm is:

    q_policy  --FK-->  Cartesian tool target
                       |
                       admittance + reference governor, on the force estimate
                       |
                       lead cap, on where the arm measurably is
                       |
                       admissible tool target  --IK-->  q_command

Because the guard re-derives the joint command from a Cartesian reference it has
authority over, a policy that predicts a joint configuration 5 cm inside the
fixture cannot express it. The one thing that passes through untouched is the
gripper, which has no contact authority in these tasks.

Two of those three stages act on the *reference* and one acts on the arm, and
the difference is the whole lesson of `_limit_lead`. Against an arm that is
where it was told to be, the admittance and the governor bound force exactly.
Against a real one they do not, because a wedged arm lags its command by
centimetres and keeps pressing from a command issued ticks ago -- a state in
which the governor sees a reference retreating and correctly declines to act
while the true force climbs past three times the limit. The lead cap is the
stage that catches that, and it is a constraint between two positions rather
than a reaction to a force.

The cost is real and is measured rather than waved at: an IK solve per control
tick, and the loss of any tool orientation the policy asked for beyond pitch --
which the SO-101's five joints could not have produced anyway (see
griff.kinematics).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from griff.control.admittance import AdmittanceConfig, AdmittanceController
from griff.kinematics import forward_kinematics, solve_ik, tool_axis
from griff.sim.env import TaskEnv

#: Cartesian stiffness between where the tool is commanded and where it is, N/m.
#:
#: Measured, not derived: pressing each fixture quasi-statically and regressing
#: true contact force on the command-to-measurement distance gives 600 N/m
#: (press fit), 633 (wipe) and 674 (peg insertion) over the sub-overload band.
#: The constant is set above all three so the resulting lead cap is conservative
#: on every task -- a stiffness guess that is too high permits too little lead,
#: which is the direction to be wrong in.
SERVO_CARTESIAN_STIFFNESS = 850.0


@dataclass
class GuardStats:
    ticks: int = 0
    governed_ticks: int = 0
    lead_clamped_ticks: int = 0
    ik_failures: int = 0
    max_correction_mm: float = 0.0
    max_lead_mm: float = 0.0

    @property
    def governed_fraction(self) -> float:
        return self.governed_ticks / max(self.ticks, 1)


class ForceGuard:
    """Wraps a joint-space command in Cartesian force-limited admittance."""

    def __init__(self, env: TaskEnv, controller: AdmittanceController | None = None) -> None:
        self.env = env
        self.controller = controller or AdmittanceController(
            AdmittanceConfig(
                force_limit=env.spec.force_limit,
                stiffness=env.spec.admittance_stiffness,
            )
        )
        self.servo_stiffness = SERVO_CARTESIAN_STIFFNESS
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
        # Comply along the tool axis only. That is the direction the peg, the pad
        # and the part can all be crushed in, and it is the only direction where
        # yielding is the right response. Yielding sideways means yielding to
        # friction, which drags the arm backwards along whatever it is doing.
        reference = self.controller.step(
            target, np.asarray(force, dtype=float), compliance_axis=tool_axis(env.model, requested)
        )
        reference = self._limit_lead(reference)

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

    def _limit_lead(self, reference: np.ndarray) -> np.ndarray:
        """Cap how far the commanded pose may lead the measured one.

        This is the mechanism that actually bounds force on a position-controlled
        arm, and it took a rammer to find out. The admittance and the governor
        both act on the *reference*, and both work: against an arm that is where
        it was told to be, the settled force lands on the limit. A real arm is
        not where it was told to be. It lags, and when it is wedged against a
        fixture it lags a long way -- the reference can retreat while the servos,
        still catching up from a command issued ticks ago, keep driving in. In
        that state the governor sees a reference moving *away* from the contact
        and correctly declines to intervene, while the true force climbs past
        three times the limit.

        Servo torque is proportional to command-minus-measurement, so bounding
        that distance bounds the force directly, with no dynamics and no
        estimate in the loop:

            F ~ K_servo * |q_commanded - q_measured|  =>  lead <= F_max / K_servo

        At 850 N/m and a 6 N limit that is 7 mm of permitted lead. Unlike the
        governor, this holds whatever the arm is doing, because it is a
        constraint between two positions rather than a reaction to a force.
        """
        actual = self.env.tool_point()
        lead = reference - actual
        distance = float(np.linalg.norm(lead))
        self.stats.max_lead_mm = max(self.stats.max_lead_mm, distance * 1000)
        allowed = self.controller.config.force_limit / self.servo_stiffness
        if distance > allowed:
            self.stats.lead_clamped_ticks += 1
            return actual + lead * (allowed / distance)
        return reference
