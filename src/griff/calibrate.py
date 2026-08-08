"""Fitting and validating the contact-force estimator.

Calibration here follows the procedure you would run on the bench, in the same
order and with the same information:

1. Take the workpiece away, so nothing can touch the tool.
2. Move the arm through its working volume at a range of speeds, with the tool
   the task will use still in the jaws -- the payload is part of what is being
   fitted, which is why calibration is per task and not per robot.
3. Record joint angles, joint velocities and servo load.
4. Least-squares fit the static load model.

Step 1 is done by zeroing every geom's collision mask, which is the simulation
equivalent of clearing the bench. The weld holding the tool is an equality
constraint and is unaffected, so the payload stays where it belongs.

`validate` then does the thing that makes the estimate a measurement rather than
an assumption: it runs contact-rich motion with the workpiece back in place and
compares the estimate against the scene's ground-truth F/T sensor.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from griff.kinematics import position_jacobian, solve_ik
from griff.paths import CALIBRATION
from griff.sensing import ContactForceEstimator, ForceCalibration, ServoLoadModel, fit_calibration
from griff.sim import make_env
from griff.sim.env import PHYSICS_SUBSTEPS
from griff.sim.tasks import SPECS


def _disable_contacts(model: mujoco.MjModel) -> None:
    """Clear the bench: no geom can collide with any other."""
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0


def collect_free_space(
    task: str, *, samples: int = 2400, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sweep the arm through free space and record (q, qdot, servo load)."""
    env = make_env(task, render=False, cameras=(), seed=seed)
    _disable_contacts(env.model)
    rng = np.random.default_rng(seed)
    servo = ServoLoadModel(rng=rng)
    env.reset(seed=seed)

    low, high = env.limits
    # Stay inside the volume the tasks actually use, and away from the joint
    # stops -- a model fitted against the limits describes the stops, not gravity.
    margin = 0.15
    lo = low[:5] + margin
    hi = high[:5] - margin

    q_log: list[np.ndarray] = []
    qd_log: list[np.ndarray] = []
    tau_log: list[np.ndarray] = []

    # The sweep has to excite all three effects the model claims to explain, or
    # the least-squares problem cannot separate them:
    #
    #   dwells        gravity, cleanly, with nothing else happening
    #   creeps        Coulomb and viscous friction, across several speeds --
    #                 friction fitted at one speed is a constant offset wearing
    #                 a different name
    #   darts         acceleration, for the inertia term. Fitting only
    #                 quasi-static data leaves that coefficient unidentifiable,
    #                 and the estimator then reports every acceleration during
    #                 an episode as contact force that is not there.
    command = env.data.ctrl.copy()

    def record() -> None:
        for _ in range(PHYSICS_SUBSTEPS):
            mujoco.mj_step(env.model, env.data)
        q_log.append(env.joint_positions)
        qd_log.append(env.joint_velocities)
        tau_log.append(servo.read(env.data.actuator_force[:5]))

    while len(q_log) < samples:
        target = rng.uniform(lo, hi)
        darting = rng.random() < 0.35
        rate = rng.uniform(0.10, 0.35) if darting else rng.uniform(0.010, 0.045)
        ceiling = 0.045 if darting else 0.010
        for _ in range(int(rng.integers(25, 70))):
            command[:5] += np.clip((target - command[:5]) * rate, -ceiling, ceiling)
            env.data.ctrl[:] = command
            record()
            if len(q_log) >= samples:
                break
        # Dwell. Static poses are the cleanest gravity samples there are, and
        # they are also where the estimator spends most of a real episode.
        for _ in range(int(rng.integers(6, 18))):
            record()
            if len(q_log) >= samples:
                break
    env.close()
    qd = np.array(qd_log)
    # Acceleration by backward difference at the control rate, exactly as the
    # estimator will compute it online from successive position reads.
    qdd = np.vstack([np.zeros((1, qd.shape[1])), np.diff(qd, axis=0) * 30.0])
    return np.array(q_log), qd, qdd, np.array(tau_log)


def calibrate(task: str, *, samples: int = 2400, seed: int = 0) -> ForceCalibration:
    q, qd, qdd, tau = collect_free_space(task, samples=samples, seed=seed)
    return fit_calibration(
        q,
        qd,
        qdd,
        tau,
        note=(
            f"task={task} free-space sweep, {samples} samples at 30 Hz, seed={seed}; "
            "collision masks cleared so nothing contacts the tool"
        ),
    )


def calibration_path(task: str) -> Path:
    return CALIBRATION / f"{task}.json"


def load_or_fit(task: str, *, samples: int = 2400, seed: int = 0) -> ForceCalibration:
    path = calibration_path(task)
    if path.exists():
        return ForceCalibration.load(path)
    fitted = calibrate(task, samples=samples, seed=seed)
    fitted.save(path)
    return fitted


@dataclass
class ValidationReport:
    """How well the servo-load estimate tracks the ground-truth F/T sensor."""

    task: str
    samples: int
    contact_samples: int
    rmse_n: float
    contact_rmse_n: float
    #: Error restricted to forces at or below the task's overload threshold.
    #: This is the band the success criterion is decided in, and the only band
    #: where the estimate is asked to be accurate rather than merely monotone.
    in_range_rmse_n: float
    bias_n: float
    correlation: float
    free_space_rms_n: float
    peak_true_n: float
    peak_estimated_n: float
    peak_ratio: float
    median_condition_number: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate(task: str, *, episodes: int = 6, seed: int = 0) -> ValidationReport:
    """Press the tool into the workpiece and compare the estimate with truth.

    The motion here is deliberately crude -- descend, hold, retreat, repeat --
    because the point is to excite contact across a range of magnitudes, not to
    do the task well.
    """
    from griff.sensing import ForceCalibration as _Calibration

    calibration: _Calibration = load_or_fit(task)
    env = make_env(task, render=False, cameras=(), calibration=calibration, seed=seed)
    estimator = ContactForceEstimator(calibration)
    servo = ServoLoadModel(rng=np.random.default_rng(seed + 991))
    rng = np.random.default_rng(seed)

    true_mag: list[float] = []
    est_mag: list[float] = []
    conditions: list[float] = []

    for episode in range(episodes):
        env.reset(seed=seed * 1000 + episode)
        estimator.reset()
        target = env.tool_point().copy()
        descent = rng.uniform(0.0006, 0.0016)
        for tick in range(env.spec.max_steps):
            target[2] -= descent if tick < env.spec.max_steps * 0.6 else -descent
            result = solve_ik(env.model, env.data, target, np.pi, env.joint_positions)
            action = result.q.copy()
            action[5] = env.spec.gripper_closed
            env.step(action)

            tau = servo.read(env.data.actuator_force[:5])
            estimate = estimator.estimate(
                env.joint_positions,
                env.joint_velocities,
                tau,
                position_jacobian(env.model, env.data),
            )
            true_mag.append(float(np.linalg.norm(env.true_force)))
            est_mag.append(estimate.magnitude)
            conditions.append(estimate.condition_number)
    env.close()

    truth = np.array(true_mag)
    estimated = np.array(est_mag)
    error = estimated - truth
    contact = truth > 1.0
    free = truth <= 0.5
    in_range = contact & (truth <= SPECS[task].overload_force)

    return ValidationReport(
        task=task,
        samples=int(truth.size),
        contact_samples=int(contact.sum()),
        rmse_n=float(np.sqrt(np.mean(error**2))),
        contact_rmse_n=float(np.sqrt(np.mean(error[contact] ** 2))) if contact.any() else 0.0,
        in_range_rmse_n=float(np.sqrt(np.mean(error[in_range] ** 2))) if in_range.any() else 0.0,
        bias_n=float(np.mean(error)),
        correlation=float(np.corrcoef(truth, estimated)[0, 1]),
        free_space_rms_n=float(np.sqrt(np.mean(estimated[free] ** 2))) if free.any() else 0.0,
        peak_true_n=float(truth.max()),
        peak_estimated_n=float(estimated.max()),
        peak_ratio=float(estimated.max() / max(truth.max(), 1e-9)),
        median_condition_number=float(np.median(conditions)),
    )


def validate_all(*, episodes: int = 6, seed: int = 0) -> dict[str, ValidationReport]:
    return {task: validate(task, episodes=episodes, seed=seed) for task in SPECS}


def write_validation(reports: dict[str, ValidationReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {task: report.to_dict() for task, report in reports.items()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
