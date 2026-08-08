"""Forward kinematics, Jacobians, and the inverse kinematics the SO-101 admits.

The SO-101 is not a 6-DoF wrist-partitioned arm and pretending otherwise is the
fastest way to write an IK solver that never converges. Its five arm joints are:

    shoulder_pan    about world z          -- selects a vertical plane
    shoulder_lift   about the plane's y    -- planar 3R chain inside that plane
    elbow_flex      about the plane's y
    wrist_flex      about the plane's y
    wrist_roll      about the tool axis    -- does not move the tool point

So the reachable task space is: tool position (3) plus the tool's pitch inside
the plane (1). Four constraints, four contributing joints. The tool cannot be
tilted sideways out of the plane at all, and wrist_roll is free.

`solve_ik` therefore targets [x, y, z, pitch] and never asks for anything else.
The pitch row of the Jacobian is exact rather than numerical: the tool axis
angle from vertical is just the sum of the three pitch joints.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

#: Joint order as the SO-101 reports it, and as LeRobot datasets store it.
JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
#: The five arm joints -- everything except the gripper jaw.
ARM_JOINTS: tuple[str, ...] = JOINT_NAMES[:5]
#: Joints that contribute to tool position and pitch (wrist_roll does neither).
IK_JOINTS: tuple[int, ...] = (0, 1, 2, 3)
#: The three pitch joints, whose angles sum to the tool axis angle from +z.
PITCH_JOINTS: tuple[int, ...] = (1, 2, 3)

TOOL_SITE = "tool_center"


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    position_error: float
    pitch_error: float
    converged: bool
    iterations: int


def site_id(model: mujoco.MjModel, name: str = TOOL_SITE) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise KeyError(f"no site named {name!r} in the model")
    return sid


def tool_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """World position of the tool centre point."""
    return data.site_xpos[site_id(model)].copy()


def tool_frame(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """World rotation matrix of the tool frame (its +z is the tool axis)."""
    return data.site_xmat[site_id(model)].reshape(3, 3).copy()


def tool_pitch(q: np.ndarray) -> float:
    """Angle of the tool axis away from world +z, in the arm's working plane.

    0 points straight up, pi points straight down at the bench.
    """
    return float(sum(q[j] for j in PITCH_JOINTS))


def forward_kinematics(model: mujoco.MjModel, q: np.ndarray) -> tuple[np.ndarray, float]:
    """(tool position, tool pitch) for a joint configuration, without stepping.

    Used to ask "where would the policy's commanded joint angles put the tool?"
    before deciding whether the arm is allowed to go there. Runs on scratch
    state so it cannot disturb the simulation.
    """
    scratch = mujoco.MjData(model)
    scratch.qpos[:6] = np.asarray(q, dtype=float)[:6]
    mujoco.mj_kinematics(model, scratch)
    return scratch.site_xpos[site_id(model)].copy(), tool_pitch(np.asarray(q, dtype=float))


def tool_axis(model: mujoco.MjModel, q: np.ndarray) -> np.ndarray:
    """Unit vector along the tool's own +z, in the base frame.

    The direction the tool points, and therefore the direction in which it can
    press something. `griff.control.guard` uses it to decide which axis the
    admittance is allowed to yield in.
    """
    scratch = mujoco.MjData(model)
    scratch.qpos[:6] = np.asarray(q, dtype=float)[:6]
    mujoco.mj_kinematics(model, scratch)
    return scratch.site_xmat[site_id(model)].reshape(3, 3)[:, 2].copy()


def position_jacobian(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """3 x 5 positional Jacobian of the tool point w.r.t. the arm joints."""
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id(model))
    return jacp[:, :5].copy()


def full_jacobian(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """6 x 5 spatial Jacobian (linear rows then angular rows) at the tool point."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id(model))
    return np.vstack([jacp[:, :5], jacr[:, :5]])


def joint_limits(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) limits for the six actuated joints, in model order."""
    lo = np.empty(6)
    hi = np.empty(6)
    for i, name in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo[i], hi[i] = model.jnt_range[jid]
    return lo, hi


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_position: np.ndarray,
    target_pitch: float,
    q_init: np.ndarray,
    *,
    max_iterations: int = 120,
    damping: float = 0.06,
    position_tolerance: float = 5e-4,
    pitch_tolerance: float = 5e-3,
    step_limit: float = 0.25,
) -> IKResult:
    """Damped least-squares IK for tool position and tool pitch.

    Runs on a scratch copy of `data` so the caller's simulation state is never
    disturbed -- IK is a planning query, not a physics step.

    `damping` is the Levenberg-Marquardt lambda. It is not small: the SO-101
    passes through a shoulder singularity whenever the arm is near full stretch,
    which is exactly where the peg-insertion fixture sits, and an undamped solve
    there produces joint steps large enough to make the follower snap.
    """
    q = np.asarray(q_init, dtype=float).copy()
    lo, hi = joint_limits(model)
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    target_position = np.asarray(target_position, dtype=float)

    pos_err = pitch_err = float("inf")
    for iteration in range(max_iterations):
        scratch.qpos[:6] = q
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)

        e_pos = target_position - scratch.site_xpos[site_id(model)]
        e_pitch = target_pitch - tool_pitch(q)
        pos_err = float(np.linalg.norm(e_pos))
        pitch_err = float(abs(e_pitch))
        if pos_err < position_tolerance and pitch_err < pitch_tolerance:
            return IKResult(q, pos_err, pitch_err, True, iteration)

        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, scratch, jacp, None, site_id(model))
        jac = np.zeros((4, len(IK_JOINTS)))
        jac[:3, :] = jacp[:, IK_JOINTS]
        for col, joint in enumerate(IK_JOINTS):
            jac[3, col] = 1.0 if joint in PITCH_JOINTS else 0.0

        error = np.concatenate([e_pos, [e_pitch]])
        jjt = jac @ jac.T + (damping**2) * np.eye(4)
        dq = jac.T @ np.linalg.solve(jjt, error)

        scale = step_limit / max(step_limit, float(np.max(np.abs(dq))))
        q[list(IK_JOINTS)] = np.clip(
            q[list(IK_JOINTS)] + scale * dq,
            lo[list(IK_JOINTS)],
            hi[list(IK_JOINTS)],
        )

    return IKResult(q, pos_err, pitch_err, False, max_iterations)
