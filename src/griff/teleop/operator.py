"""Scripted operators: the thing on the other end of the leader arm.

These stand in for a human teleoperating the rig. They are not policies and they
are not baselines -- they exist only to generate demonstrations, and they are
allowed to use information a human would have and a policy would not: where the
fixture is, that the plate is tilted, that the part has seated.

What they are *not* allowed is to be better than a human at the part that
matters. Each one carries a per-episode aiming error of a few millimetres that
it cannot see and has to resolve through contact -- which is the reason the
demonstrations contain force-guided corrections at all. Take that away and
every episode becomes a clean feedforward trajectory, the force channel becomes
constant, and the vision-versus-force ablation has nothing to measure.

They regulate on the *estimated* force, not the simulator's ground truth, for
the same reason: an operator watching a force readout on screen sees the
estimate, and demonstrations conditioned on a signal the policy will never see
would teach behaviour the policy cannot reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from griff.kinematics import solve_ik
from griff.sim.env import TaskEnv
from griff.sim.tasks import PegInsertEnv, PressFitEnv, WipeEnv


@dataclass
class OperatorStyle:
    """Per-episode operator variation. Redrawn every episode.

    Demonstrations that all share one speed and one approach make a dataset a
    policy can overfit to the point of memorising the clock.
    """

    speed: float = 1.0
    aim_error: np.ndarray = field(default_factory=lambda: np.zeros(2))
    jitter: float = 0.00012  # m, per-tick reference noise
    pause_probability: float = 0.02
    force_target_bias: float = 0.0  # N; some operators press harder than others

    @classmethod
    def draw(cls, rng: np.random.Generator, *, aim_error_mm: float) -> OperatorStyle:
        return cls(
            speed=float(rng.uniform(0.82, 1.18)),
            aim_error=rng.uniform(-aim_error_mm, aim_error_mm, size=2) / 1000.0,
            jitter=float(rng.uniform(0.00007, 0.00018)),
            pause_probability=float(rng.uniform(0.0, 0.05)),
            force_target_bias=float(rng.uniform(-0.6, 0.8)),
        )


class Operator:
    """Base class: Cartesian reference integration, IK, and operator noise.

    The reference is rate-limited centrally, in `act`, rather than left to each
    phase to be careful. That is not tidiness -- it is what makes the recorded
    force channel mean anything. An SO-101's position servos have kp = 34 N.m/rad
    against roughly 0.6 N.m.s/rad of joint damping, so commanding the tool at
    10 cm/s leaves an 80 mrad tracking error standing at the shoulder, which
    `griff.sensing` reads as 3-6 N of contact force that nothing is touching.
    Demonstrations recorded that way teach a policy to react to an artefact.

    A human on a leader arm is subject to the same limit and solves it the same
    way: for contact-rich work they move at a few centimetres per second.
    """

    #: How far off the operator's aim is at the start of an episode, in mm.
    aim_error_mm: float = 3.0
    #: Hard ceiling on Cartesian reference motion, m per 30 Hz tick (~3.6 cm/s).
    MAX_REFERENCE_RATE: float = 0.0012

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng(0)
        self.style = OperatorStyle()
        self.reference: np.ndarray = np.zeros(3)
        self.pitch = float(np.pi)
        self.phase = "start"
        self.phase_ticks = 0
        self._last_q = np.zeros(6)

    # ------------------------------------------------------------------ shared

    def reset(self, env: TaskEnv, rng: np.random.Generator | None = None) -> None:
        if rng is not None:
            self.rng = rng
        self.style = OperatorStyle.draw(self.rng, aim_error_mm=self.aim_error_mm)
        self.reference = env.tool_point().copy()
        self.pitch = float(np.pi)
        self.phase = "start"
        self.phase_ticks = 0
        self._last_q = env.joint_positions.copy()

    def _enter(self, phase: str) -> None:
        if phase != self.phase:
            self.phase = phase
            self.phase_ticks = 0

    def _approach_xy(self, target_xy: np.ndarray, rate: float) -> None:
        """Move the reference laterally toward a target, at a limited rate."""
        delta = target_xy - self.reference[:2]
        distance = float(np.linalg.norm(delta))
        limit = rate * self.style.speed
        if distance > limit:
            delta = delta * (limit / distance)
        self.reference[:2] += delta

    def _regulate_force(
        self, measured: float, target: float, *, gain: float = 0.00055, limit: float = 0.0016
    ) -> None:
        """Move the reference along z to drive the measured force to a target.

        Pressing harder means lowering the reference, so the sign is negative.
        The per-tick limit is what keeps a force estimate glitch from becoming a
        3 cm lunge.
        """
        correction = gain * (target - measured)
        self.reference[2] -= float(np.clip(correction, -limit, limit))

    def _noise(self) -> None:
        if self.rng.random() < self.style.pause_probability:
            return
        self.reference[:2] += self.rng.normal(0.0, self.style.jitter, size=2)
        self.reference[2] += self.rng.normal(0.0, self.style.jitter * 0.5)

    def _to_joints(self, env: TaskEnv) -> np.ndarray:
        result = solve_ik(env.model, env.data, self.reference, self.pitch, self._last_q)
        joints = result.q.copy()
        joints[5] = env.spec.gripper_closed
        if result.converged:
            self._last_q = joints
        else:
            # Out of reach: hold the last good pose rather than command the
            # partial solve, which would be a lunge toward a joint limit.
            self.reference = env.tool_point().copy()
            joints = self._last_q.copy()
        return joints

    def act(self, env: TaskEnv, force: np.ndarray) -> np.ndarray:
        self.phase_ticks += 1
        previous = self.reference.copy()
        self._plan(env, float(np.linalg.norm(force)))
        self._noise()
        delta = self.reference - previous
        distance = float(np.linalg.norm(delta))
        if distance > self.MAX_REFERENCE_RATE:
            self.reference = np.asarray(previous + delta * (self.MAX_REFERENCE_RATE / distance))
        return self._to_joints(env)

    def _plan(self, env: TaskEnv, force_magnitude: float) -> None:
        raise NotImplementedError


class PegInsertOperator(Operator):
    """Aim, land on the lip, wiggle until it drops, then press it home."""

    aim_error_mm = 3.2

    APPROACH_Z = 0.104
    LIP_TOP = 0.056
    BORE_TOP = 0.048
    SEARCH_FORCE = 2.2  # N, light enough to slide rather than jam
    INSERT_FORCE = 3.2  # N

    def _plan(self, env: PegInsertEnv, force_magnitude: float) -> None:  # type: ignore[override]
        hole = env._hole_xy
        aim = hole + self.style.aim_error
        tip_z = env.site_position("peg_tip")[2]
        inserted = tip_z < self.BORE_TOP - 0.004

        if self.phase == "start":
            self._enter("aim")
            self._found_xy = aim.copy()

        if self.phase == "aim":
            self._approach_xy(aim, 0.0011)
            self.reference[2] += (self.APPROACH_Z - self.reference[2]) * 0.08
            if np.linalg.norm(self.reference[:2] - aim) < 0.0015 and self.phase_ticks > 6:
                self._enter("descend")

        elif self.phase == "descend":
            self._approach_xy(aim, 0.0006)
            self.reference[2] -= 0.0009 * self.style.speed
            if force_magnitude > 1.0 or tip_z < self.LIP_TOP - 0.002:
                self._enter("search")

        elif self.phase == "search":
            # A widening spiral, pressed on lightly. This is the whole point of
            # the task: the residual misalignment is below what the operator can
            # see, so it is resolved by feeling the peg catch and slide.
            radius = min(0.0042, 0.0006 + 0.00006 * self.phase_ticks)
            angle = 0.10 * self.phase_ticks * self.style.speed
            # Centred on where the operator *believes* the hole is, not where it
            # is. Spiralling around the true centre would silently cancel the
            # aim error and turn this into a feedforward drop with no contact
            # in it at all -- which is exactly what it did before this line was
            # fixed: 12/12 successes at 0.3 N peak, i.e. never touching.
            probe = aim + radius * np.array([np.cos(angle), np.sin(angle)])
            self._approach_xy(probe, 0.0009)
            self._regulate_force(force_magnitude, self.SEARCH_FORCE + self.style.force_target_bias)
            if inserted:
                # The peg found the bore here; hold this position rather than
                # the original aim, which is what was wrong by construction.
                self._found_xy = self.reference[:2].copy()
                self._enter("insert")
            elif self.phase_ticks > 90:
                # Give up searching and just press; a stuck operator eventually
                # forces it, and the dataset should contain that too.
                self._found_xy = self.reference[:2].copy()
                self._enter("insert")

        elif self.phase == "insert":
            self._approach_xy(self._found_xy, 0.0006)
            self._regulate_force(force_magnitude, self.INSERT_FORCE + self.style.force_target_bias)
            self.reference[2] -= 0.0004 * self.style.speed

        self.reference[2] = float(np.clip(self.reference[2], 0.062, 0.130))


class WipeOperator(Operator):
    """Engage the plate, then sweep while holding the force in the band."""

    aim_error_mm = 2.0

    APPROACH_Z = 0.092
    BAND_TARGET = 3.4  # N, mid-band
    SWEEP_RATE = 0.0009  # m per tick

    def _plan(self, env: WipeEnv, force_magnitude: float) -> None:  # type: ignore[override]
        plate = env._plate_xy
        start = np.array([plate[0], plate[1] + env.CHECKPOINTS[0]]) + self.style.aim_error
        end = np.array([plate[0], plate[1] + env.CHECKPOINTS[-1]]) + self.style.aim_error

        if self.phase == "start":
            self._enter("approach")

        if self.phase == "approach":
            self._approach_xy(start, 0.0011)
            self.reference[2] += (self.APPROACH_Z - self.reference[2]) * 0.08
            if np.linalg.norm(self.reference[:2] - start) < 0.002 and self.phase_ticks > 5:
                self._enter("engage")

        elif self.phase == "engage":
            self.reference[2] -= 0.0008 * self.style.speed
            if force_magnitude >= 2.0 or self.phase_ticks > 70:
                self._enter("sweep")

        elif self.phase == "sweep":
            self._approach_xy(end, self.SWEEP_RATE * self.style.speed)
            # The tilt is unknown to the operator, so the plate surface rises or
            # falls under the pad as it travels. Holding the band is a
            # closed-loop act on the force reading, tick by tick.
            self._regulate_force(
                force_magnitude,
                self.BAND_TARGET + self.style.force_target_bias,
                gain=0.00075,
            )
            if np.linalg.norm(self.reference[:2] - end) < 0.0025:
                self._enter("return")

        elif self.phase == "return":
            self._approach_xy(start, self.SWEEP_RATE * 0.8 * self.style.speed)
            self._regulate_force(
                force_magnitude,
                self.BAND_TARGET + self.style.force_target_bias,
                gain=0.00075,
            )

        self.reference[2] = float(np.clip(self.reference[2], 0.058, 0.115))


class PressFitOperator(Operator):
    """Align, meet the retainer, then ramp the press until it seats."""

    aim_error_mm = 1.6

    APPROACH_Z = 0.108
    CONTACT_FORCE = 1.5  # N
    RAMP_START = 3.0  # N
    RAMP_RATE = 0.16  # N per tick
    #: Ceiling on the operator's press. Below the task's 14 N overload, because
    #: a demonstration that damages the workpiece is not a demonstration.
    RAMP_CEILING = 11.5  # N

    def _plan(self, env: PressFitEnv, force_magnitude: float) -> None:  # type: ignore[override]
        socket = env._socket_xy
        aim = socket + self.style.aim_error

        if self.phase == "start":
            self._enter("align")
            self._force_target = self.RAMP_START

        if self.phase == "align":
            self._approach_xy(aim, 0.0011)
            self.reference[2] += (self.APPROACH_Z - self.reference[2]) * 0.08
            if np.linalg.norm(self.reference[:2] - aim) < 0.0012 and self.phase_ticks > 6:
                self._enter("descend")

        elif self.phase == "descend":
            self._approach_xy(aim, 0.0005)
            self.reference[2] -= 0.0009 * self.style.speed
            if force_magnitude > self.CONTACT_FORCE:
                self._enter("press")
                self._force_target = self.RAMP_START

        elif self.phase == "press":
            # The retainer stiffness is not observable, so the required force is
            # not known in advance. The operator ramps until it gives.
            # Ramp until it is *actually* seated, not almost. Stopping at 85%
            # of the depth left the last millimetre unforced, which is the
            # millimetre the success criterion is decided on.
            seated = env._retainer_travel() > env.SEAT_DEPTH
            if not seated:
                self._force_target = min(
                    self.RAMP_CEILING, self._force_target + self.RAMP_RATE * self.style.speed
                )
            self._approach_xy(aim, 0.0004)
            self._regulate_force(
                force_magnitude,
                self._force_target + self.style.force_target_bias,
                gain=0.00048,
                limit=0.0014,
            )

        self.reference[2] = float(np.clip(self.reference[2], 0.068, 0.130))


OPERATORS: dict[str, type[Operator]] = {
    "peg_insert": PegInsertOperator,
    "wipe": WipeOperator,
    "press_fit": PressFitOperator,
}


def make_operator(task: str, rng: np.random.Generator | None = None) -> Operator:
    if task not in OPERATORS:
        raise KeyError(f"no operator for task {task!r}; known: {sorted(OPERATORS)}")
    return OPERATORS[task](rng)
