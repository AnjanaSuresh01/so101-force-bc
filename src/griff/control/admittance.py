"""Force-limited Cartesian admittance control.

This is the layer that makes a behaviour-cloning policy safe to run against a
workpiece. The policy proposes where the tool should be; this decides where the
tool is allowed to be, given what the tool is currently feeling.

Three mechanisms, and it matters that they are separate, because only one of
them actually bounds anything:

1. **Admittance.** A virtual mass-damper-spring turns measured contact force
   into a retreat displacement, so the arm yields on contact instead of pushing
   through. On its own this does *not* bound force. In steady state the contact
   settles at ``F = d * (K_env * K_a) / (K_env + K_a)``, where ``d`` is how deep
   the *policy* commanded -- so a policy that commands 5 cm into a fixture still
   generates a large force, just more gently. Admittance buys compliance and
   softens impact. It does not buy a limit.

2. **A reference governor.** While the measured force is at the limit, the
   reference may not advance any further along the direction the contact is
   resisting. This bounds the *steady state* whatever the policy asks for,
   because it is a projection applied to the commanded reference itself rather
   than a term added to it.

3. **Online stiffness estimation.** A governor that only fires *after* a tick
   in which the limit was exceeded still lets one unthrottled step slam into the
   fixture. So the controller estimates the local environment stiffness from its
   own recent motion (``dF/dx``) and caps each tick's advance at the distance
   predicted to consume the remaining force headroom. This is what keeps the
   *transient* small, and it is the part that degrades gracefully: when the
   stiffness estimate is wrong, the overshoot is proportional to how wrong it
   is, not unbounded. docs/admittance-bound.md measures exactly that.

Written against a real deployment constraint: the SO-101 has no wrist F/T
sensor, so ``force`` here is an estimate from servo load (see griff.sensing).
The loop is therefore built to degrade sanely under a noisy force signal -- a
continuous deadband keeps sensor noise from ratcheting the reference backwards,
and integration is backward Euler so nothing goes unstable at the 30 Hz rate the
estimate actually arrives at.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class AdmittanceConfig:
    """Tuning for the compliance layer.

    Defaults are for the SO-101 at 30 Hz against a bench fixture.
    ``mass``/``damping``/``stiffness`` are overdamped on purpose (zeta ~ 1.3):
    an underdamped admittance against a stiff environment is the classic way to
    turn a compliant controller into an oscillator that hammers the workpiece.
    """

    dt: float = 1.0 / 30.0
    mass: float = 1.2  # kg, virtual
    damping: float = 45.0  # N.s/m
    stiffness: float = 250.0  # N/m, pulls the offset back to zero in free space
    force_limit: float = 8.0  # N, the bound the governor enforces
    deadband: float = 0.35  # N, below this the force estimate reads as noise
    max_offset: float = 0.035  # m, how far compliance may deviate from the policy
    max_step: float = 0.006  # m per tick == 0.18 m/s, the free-space slew limit

    # --- environment stiffness estimator ---
    #: Starting guess, used until contact has been probed. Deliberately stiff:
    #: guessing too soft makes the first contact tick overshoot, guessing too
    #: stiff only makes the approach cautious.
    stiffness_prior: float = 4000.0  # N/m
    stiffness_bounds: tuple[float, float] = (250.0, 80000.0)
    stiffness_smoothing: float = 0.35  # EMA weight on each new observation
    #: Advances smaller than this carry no usable stiffness information -- the
    #: ratio dF/dx is dominated by force-estimate noise below it.
    min_probe_distance: float = 5e-5  # m

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.mass <= 0 or self.damping < 0 or self.stiffness < 0:
            raise ValueError("mass must be positive; damping and stiffness non-negative")
        if self.force_limit <= self.deadband:
            raise ValueError("force_limit must exceed the deadband")
        if self.max_offset <= 0 or self.max_step <= 0:
            raise ValueError("max_offset and max_step must be positive")
        low, high = self.stiffness_bounds
        if not 0 < low < high:
            raise ValueError("stiffness_bounds must be an increasing positive pair")
        if not low <= self.stiffness_prior <= high:
            raise ValueError("stiffness_prior must lie inside stiffness_bounds")
        if not 0 < self.stiffness_smoothing <= 1:
            raise ValueError("stiffness_smoothing must be in (0, 1]")


@dataclass
class AdmittanceState:
    """Integrator and estimator state. Reset between episodes."""

    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reference: np.ndarray | None = None
    stiffness: float = 0.0
    #: Commanded motion on the previous tick, kept as a vector so the stiffness
    #: estimator can project it onto whichever direction contact turns out to be
    #: in -- including the tick that first makes contact, where there was no
    #: force to project onto at the time the motion was commanded.
    last_delta: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_direction: np.ndarray | None = None
    last_magnitude: float = 0.0
    probes: int = 0
    #: True on any tick where the advance cap or the governor clipped the command.
    governed: bool = False
    #: Largest force magnitude seen since the last reset.
    peak_force: float = 0.0
    #: Number of ticks the command was clipped, since the last reset.
    governed_ticks: int = 0


class AdmittanceController:
    """Maps a policy's Cartesian tool target to a force-safe tool target.

    One call per control tick::

        reference = controller.step(policy_target, measured_force)

    ``measured_force`` is the force the environment exerts on the tool, in the
    same frame as ``policy_target`` (the arm's base frame here). Pressing down
    into the bench therefore reads as a force with a positive z component.
    """

    def __init__(self, config: AdmittanceConfig | None = None) -> None:
        self.config = config or AdmittanceConfig()
        self.state = AdmittanceState(stiffness=self.config.stiffness_prior)

    def reset(self, reference: np.ndarray | None = None) -> None:
        self.state = AdmittanceState(
            reference=None if reference is None else np.asarray(reference, dtype=float).copy(),
            stiffness=self.config.stiffness_prior,
        )

    def with_force_limit(self, force_limit: float) -> AdmittanceController:
        """A copy tuned to a different limit -- one per task, sharing everything else."""
        return AdmittanceController(replace(self.config, force_limit=force_limit))

    def _deadbanded(self, force: np.ndarray, magnitude: float) -> np.ndarray:
        """Shrink the force toward zero by the deadband, preserving direction.

        Subtracting a fixed magnitude rather than zeroing below a threshold keeps
        the response continuous. A hard threshold makes the controller jump every
        time a noisy estimate crosses it, which on a servo-load estimate is
        several times a second.
        """
        if magnitude <= self.config.deadband:
            return np.zeros(3)
        return force * (magnitude - self.config.deadband) / magnitude

    def _update_stiffness(self, magnitude: float, direction: np.ndarray | None) -> None:
        """Estimate local environment stiffness from the last tick's own motion.

        Symmetric in sign: pressing in and seeing force rise, and backing out and
        seeing it fall, are equally good observations of dF/dx. Learning on the
        way out matters more than it looks -- it is what stops the recovery from
        an over-limit contact turning into a limit cycle of slam-and-release.

        The first observation replaces the prior outright rather than being
        blended into it. The prior is a guess; the first real contact is data.
        """
        cfg = self.config
        if direction is None:
            return
        advance = -float(self.state.last_delta @ direction)
        if abs(advance) < cfg.min_probe_distance:
            return
        rise = magnitude - self.state.last_magnitude
        if rise * advance <= 0:
            # Force moved the wrong way for the motion commanded: this is the
            # arm sliding along a surface or the estimate being noisy, not a
            # stiffness measurement.
            return
        observed = rise / advance
        alpha = 1.0 if self.state.probes == 0 else cfg.stiffness_smoothing
        self.state.probes += 1
        blended = (1 - alpha) * self.state.stiffness + alpha * observed
        self.state.stiffness = float(np.clip(blended, *cfg.stiffness_bounds))

    def step(
        self,
        policy_target: np.ndarray,
        measured_force: np.ndarray,
        compliance_axis: np.ndarray | None = None,
    ) -> np.ndarray:
        """One control tick.

        `compliance_axis`, when given, restricts the *compliance* to a single
        direction -- normally the tool axis, the one direction in which these
        tasks can crush something. The force limit is unaffected and stays
        three-dimensional.

        That split is not a refinement, it is the difference between a
        controller that works and one that does not. Compliance in all three
        axes means the arm also yields to *friction*, and friction opposes
        motion: during a wipe the offset grows backwards along the stroke and
        the pad lags several millimetres behind its reference for the whole
        sweep. Measured, that cost the scripted operator every single episode of
        the wiping task -- 0/20 with three-axis compliance against 20/20 without
        it, with the governor never once firing. The arm was not being stopped;
        it was being dragged.
        """
        cfg = self.config
        target = np.asarray(policy_target, dtype=float)
        force = np.asarray(measured_force, dtype=float)
        if target.shape != (3,) or force.shape != (3,):
            raise ValueError("policy_target and measured_force must both be 3-vectors")

        magnitude = float(np.linalg.norm(force))
        self.state.peak_force = max(self.state.peak_force, magnitude)
        # In contact, the contact direction is the force direction. Out of it,
        # reuse the last known direction so the release is still informative.
        direction = force / magnitude if magnitude > cfg.deadband else self.state.last_direction
        self._update_stiffness(magnitude, direction)
        effective = self._deadbanded(force, magnitude)
        if compliance_axis is not None:
            axis = np.asarray(compliance_axis, dtype=float)
            norm = float(np.linalg.norm(axis))
            if axis.shape != (3,) or norm < 1e-9:
                raise ValueError("compliance_axis must be a non-zero 3-vector")
            axis = axis / norm
            effective = float(effective @ axis) * axis

        # Backward Euler on  M x'' + D x' + K x = -F.  Solving for the new
        # velocity in closed form keeps this unconditionally stable, which
        # matters because dt is 33 ms -- explicit integration of a stiff
        # admittance at that rate diverges against a rigid fixture.
        denom = cfg.mass + cfg.dt * cfg.damping + cfg.dt * cfg.dt * cfg.stiffness
        self.state.velocity = (
            cfg.mass * self.state.velocity
            - cfg.dt * (effective + cfg.stiffness * self.state.offset)
        ) / denom
        offset = self.state.offset + cfg.dt * self.state.velocity

        norm = float(np.linalg.norm(offset))
        if norm > cfg.max_offset:
            offset = offset * (cfg.max_offset / norm)
            # Bleed off the outward velocity too, or the integrator winds up
            # against the clamp and the arm lurches when contact is released.
            radial = offset / max(norm, _EPS)
            outward = float(self.state.velocity @ radial)
            if outward > 0:
                self.state.velocity = self.state.velocity - outward * radial
        self.state.offset = offset

        reference = target + offset
        previous = self.state.reference if self.state.reference is not None else reference

        # Free-space slew limit. The policy can jump; the arm should not.
        delta = reference - previous
        step_norm = float(np.linalg.norm(delta))
        if step_norm > cfg.max_step:
            delta = delta * (cfg.max_step / step_norm)

        # The bound. `allowance` is how far this tick is permitted to move into
        # the contact, from the force headroom and the stiffness estimate. When
        # the force is already over the limit the headroom is negative, so the
        # allowance is a *negative* advance -- a commanded retreat of exactly the
        # distance predicted to shed the excess. One expression covers both the
        # approach cap and the recovery, and the steady state is the limit
        # rather than wherever the first over-limit tick happened to stop.
        self.state.governed = False
        if magnitude > cfg.deadband and direction is not None:
            advance = -float(delta @ direction)  # positive = deeper into contact
            headroom = cfg.force_limit - magnitude
            allowance = max(headroom / max(self.state.stiffness, _EPS), -cfg.max_step)
            if advance > allowance:
                delta = delta + (advance - allowance) * direction
                self.state.governed = True
                self.state.governed_ticks += 1
            self.state.last_direction = direction

        self.state.last_delta = delta.copy()
        self.state.last_magnitude = magnitude
        self.state.reference = previous + delta
        return self.state.reference.copy()
