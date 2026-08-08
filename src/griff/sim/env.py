"""The common environment: an SO-101 with a tool welded into its jaws.

Design decisions worth stating up front, because they set the limits of every
number this repo reports:

**The policy's force channel is an estimate, not a measurement.** Each control
tick reads servo load through `ServoLoadModel` -- quantised to the STS3215's
1/1000-of-stall resolution and noised -- and puts it through
`ContactForceEstimator`. The scenes also carry a true F/T sensor at the tool
mount; it is recorded in `info` for evaluating the estimator, and is never
visible to a policy or to the controller.

**Tools are welded, not grasped.** Peg, pad and part are held by an equality
constraint rather than by friction between the jaws. Grasping is not what is
being studied, and a friction grasp would spend every episode slipping and turn
the force signal into an artefact of the grip rather than of the task.

**Control runs at 30 Hz over 600 Hz physics.** That is the SO-101's teleoperation
rate, the dataset's fps, and the rate the force estimate arrives at. Everything
downstream -- the admittance discretisation, the action chunk length, the
success dwell times -- is expressed in ticks at this rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from griff.kinematics import (
    JOINT_NAMES,
    joint_limits,
    position_jacobian,
    solve_ik,
    tool_position,
)
from griff.sensing import ContactForceEstimator, ForceCalibration, ServoLoadModel

CONTROL_HZ = 30
PHYSICS_SUBSTEPS = 20


@dataclass
class Observation:
    """What a policy sees. Nothing here is privileged."""

    state: np.ndarray  # (6,) joint positions, rad (gripper in m)
    force: np.ndarray  # (3,) estimated contact force, N, base frame
    images: dict[str, np.ndarray] = field(default_factory=dict)  # HxWx3 uint8

    def copy(self) -> Observation:
        return Observation(
            state=self.state.copy(),
            force=self.force.copy(),
            images={k: v.copy() for k, v in self.images.items()},
        )


@dataclass
class StepResult:
    observation: Observation
    success: bool
    done: bool
    info: dict[str, Any]


class TaskEnv:
    """Base class. Subclasses supply randomisation, success and the home pose."""

    #: Set by subclasses; see griff.sim.tasks.
    spec: Any

    def __init__(
        self,
        *,
        image_size: int = 96,
        cameras: tuple[str, ...] = ("top", "wrist"),
        calibration: ForceCalibration | None = None,
        render: bool = True,
        seed: int = 0,
    ) -> None:
        from griff.paths import scene

        self.model = mujoco.MjModel.from_xml_path(str(scene(self.spec.scene)))
        self.data = mujoco.MjData(self.model)
        self.image_size = image_size
        self.cameras = cameras
        self.rng = np.random.default_rng(seed)

        self._renderer = (
            mujoco.Renderer(self.model, image_size, image_size) if render and cameras else None
        )
        self._servo = ServoLoadModel(rng=self.rng)
        self._estimator = ContactForceEstimator(calibration)
        self._joint_qpos_adr = np.array(
            [
                self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                for n in JOINT_NAMES
            ]
        )
        self._joint_dof_adr = np.array(
            [
                self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                for n in JOINT_NAMES
            ]
        )
        self._held_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.spec.held_body)
        held_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.spec.held_joint)
        self._held_qpos_adr = self.model.jnt_qposadr[held_joint]
        self._held_dof_adr = self.model.jnt_dofadr[held_joint]
        self._tool_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tool")
        self.limits = joint_limits(self.model)

        self.step_index = 0
        self.peak_force = 0.0
        self.peak_force_true = 0.0
        self.home_action = np.zeros(6)
        self._home_retreat_mm = 0.0
        self._episode: dict[str, Any] = {}

    # ------------------------------------------------------------------ hooks

    def _randomise(self, rng: np.random.Generator) -> dict[str, Any]:
        """Per-episode model changes. Returns what was drawn, for the metadata."""
        return {}

    def _home_target(self, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        """(tool position, tool pitch) for the start of the episode."""
        raise NotImplementedError

    def _success(self) -> bool:
        raise NotImplementedError

    def _task_info(self) -> dict[str, Any]:
        return {}

    # ------------------------------------------------------------------ state

    @property
    def joint_positions(self) -> np.ndarray:
        return self.data.qpos[self._joint_qpos_adr].copy()

    @property
    def joint_velocities(self) -> np.ndarray:
        return self.data.qvel[self._joint_dof_adr].copy()

    @property
    def true_force(self) -> np.ndarray:
        """Ground-truth contact wrench force at the tool mount, base frame.

        The sensor reports in the site frame; rotating it into the base frame
        keeps it comparable with the estimator, which works in base coordinates.
        """
        site_force = self.data.sensordata[:3]
        rotation = self.data.site_xmat[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tool_ft")
        ].reshape(3, 3)
        return -(rotation @ site_force)

    def tool_point(self) -> np.ndarray:
        return tool_position(self.model, self.data)

    # ------------------------------------------------------------------ episode

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._servo.rng = self.rng
        rng = self.rng

        mujoco.mj_resetData(self.model, self.data)
        self._estimator.reset()
        self._episode = self._randomise(rng)
        self.step_index = 0
        self.peak_force = 0.0
        self.peak_force_true = 0.0

        target, pitch = self._home_target(rng)
        home = self._place_arm(target, pitch)
        home[5] = self.spec.gripper_closed

        self.data.qpos[self._joint_qpos_adr] = home
        self.data.ctrl[:] = home
        mujoco.mj_forward(self.model, self.data)
        self._place_held_body()
        mujoco.mj_forward(self.model, self.data)

        # Let the weld and the position servos settle before the episode starts,
        # so step 0 is not dominated by a startup transient in the force channel.
        for _ in range(PHYSICS_SUBSTEPS * 6):
            mujoco.mj_step(self.model, self.data)
        self._estimator.reset()
        self.home_action = home
        return self.observe()

    def _place_arm(self, target: np.ndarray, pitch: float) -> np.ndarray:
        """Solve for a start pose, backing off if the request is out of reach.

        The randomised workpiece position plus the randomised approach jitter can
        land a few millimetres outside the SO-101's reachable set -- the arm is
        near full stretch at these fixtures, and pointing the tool straight down
        costs most of the wrist's travel. Rather than discard the episode, do
        what an operator would: come down a little, and in a little, until the
        pose is reachable. The retreat is small and is recorded, so nothing about
        the episode is silently different from what was asked for.
        """
        seed_q = np.array([0.0, 0.55, 0.90, 1.05, 0.0, self.spec.gripper_open])
        attempt = np.asarray(target, dtype=float).copy()
        radial = attempt[:2] / max(float(np.linalg.norm(attempt[:2])), 1e-9)
        for retry in range(6):
            result = solve_ik(self.model, self.data, attempt, pitch, seed_q)
            if result.converged:
                self._home_retreat_mm = float(np.linalg.norm(attempt - target) * 1000)
                return result.q.copy()
            attempt[2] -= 0.004
            if retry >= 2:
                attempt[:2] -= radial * 0.005
        raise RuntimeError(
            f"{self.spec.name}: could not place the arm anywhere near "
            f"{np.round(target, 4)} (best position error "
            f"{result.position_error * 1000:.1f} mm). The workpiece randomisation "
            "has moved outside the SO-101's reachable set."
        )

    def _place_held_body(self) -> None:
        """Put the welded tool exactly where its weld wants it, before stepping.

        Skipping this lets the weld resolve a several-centimetre error on the
        first physics step, which launches the peg across the bench.
        """
        tool_pos = self.data.xpos[self._tool_body].copy()
        tool_mat = self.data.xmat[self._tool_body].reshape(3, 3).copy()
        offset = np.array([0.0, 0.0, self.spec.weld_offset])
        self.data.qpos[self._held_qpos_adr : self._held_qpos_adr + 3] = tool_pos + tool_mat @ offset
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, tool_mat.reshape(9))
        self.data.qpos[self._held_qpos_adr + 3 : self._held_qpos_adr + 7] = quat
        self.data.qvel[self._held_dof_adr : self._held_dof_adr + 6] = 0.0

    def step(self, action: np.ndarray) -> StepResult:
        low, high = self.limits
        command = np.clip(np.asarray(action, dtype=float), low, high)
        self.data.ctrl[:] = command
        for _ in range(PHYSICS_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        self.step_index += 1

        observation = self.observe()
        true_magnitude = float(np.linalg.norm(self.true_force))
        self.peak_force = max(self.peak_force, float(np.linalg.norm(observation.force)))
        self.peak_force_true = max(self.peak_force_true, true_magnitude)

        success = self._success()
        done = success or self.step_index >= self.spec.max_steps
        info = {
            "true_force": self.true_force,
            "true_force_magnitude": true_magnitude,
            "estimated_force_magnitude": float(np.linalg.norm(observation.force)),
            "peak_force": self.peak_force,
            "peak_force_true": self.peak_force_true,
            "overloaded": self.peak_force_true > self.spec.overload_force,
            "step": self.step_index,
            **self._task_info(),
        }
        return StepResult(observation, success, done, info)

    # ------------------------------------------------------------------ sensing

    def estimate_force(self) -> np.ndarray:
        tau = self._servo.read(self.data.actuator_force[:5])
        jacobian = position_jacobian(self.model, self.data)
        estimate = self._estimator.estimate(
            self.joint_positions, self.joint_velocities, tau, jacobian
        )
        return estimate.force

    def observe(self) -> Observation:
        return Observation(
            state=self.joint_positions,
            force=self.estimate_force(),
            images=self.render_all(),
        )

    def render_all(self) -> dict[str, np.ndarray]:
        if self._renderer is None:
            return {}
        images = {}
        for camera in self.cameras:
            self._renderer.update_scene(self.data, camera=camera)
            images[camera] = self._renderer.render().copy()
        return images

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> TaskEnv:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ helpers

    def site_position(self, name: str) -> np.ndarray:
        return self.data.site_xpos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        ].copy()

    def body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    @property
    def episode_parameters(self) -> dict[str, Any]:
        """Whatever the randomiser drew, for the dataset's episode metadata."""
        return dict(self._episode)
