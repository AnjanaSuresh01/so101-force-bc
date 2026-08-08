"""Kinematics: what the SO-101 can reach, and what it cannot."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from griff.kinematics import (
    ARM_JOINTS,
    IK_JOINTS,
    JOINT_NAMES,
    forward_kinematics,
    joint_limits,
    position_jacobian,
    solve_ik,
    tool_pitch,
    tool_position,
)
from griff.paths import scene


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(scene("free")))


@pytest.fixture
def data(model: mujoco.MjModel) -> mujoco.MjData:
    return mujoco.MjData(model)


def test_joint_names_match_the_model(model: mujoco.MjModel) -> None:
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
    ]
    assert names == list(JOINT_NAMES)
    assert len(ARM_JOINTS) == 5


def test_tool_pitch_is_the_sum_of_the_pitch_joints(model, data) -> None:
    """The three pitch joints share an axis; the tool angle is their sum.

    This is the identity the IK's fourth Jacobian row asserts analytically. If
    the model ever gains an offset that breaks it, IK stops converging in a way
    that looks like a tuning problem.
    """
    q = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.01])
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    frame = data.site_xmat[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_center")
    ].reshape(3, 3)
    axis = frame[:, 2]
    angle_from_vertical = np.arccos(np.clip(axis[2], -1, 1))
    assert angle_from_vertical == pytest.approx(abs(tool_pitch(q)), abs=1e-6)


@pytest.mark.parametrize(
    "target",
    [(0.19, 0.0, 0.075), (0.19, 0.03, 0.10), (0.19, -0.03, 0.10), (0.16, 0.0, 0.09),
     (0.20, 0.04, 0.085), (0.20, -0.04, 0.085)],
)
def test_ik_reaches_the_working_volume(model, data, target) -> None:
    result = solve_ik(model, data, np.array(target), np.pi, np.array([0, 0.55, 0.9, 1.05, 0, 0.01]))
    assert result.converged, f"{target} unreachable: {result.position_error * 1000:.1f} mm"
    assert result.position_error < 5e-4
    data.qpos[:6] = result.q
    mujoco.mj_forward(model, data)
    assert np.allclose(tool_position(model, data), target, atol=1e-3)


def test_ik_reports_failure_rather_than_returning_nonsense(model, data) -> None:
    """Far outside the workspace, IK must say so, not return its best guess."""
    result = solve_ik(model, data, np.array([0.6, 0.0, 0.3]), np.pi, np.zeros(6))
    assert not result.converged
    assert result.position_error > 0.05


def test_ik_respects_joint_limits(model, data) -> None:
    lower, upper = joint_limits(model)
    for target in [(0.22, 0.0, 0.13), (0.14, 0.05, 0.06), (0.20, 0.0, 0.07)]:
        result = solve_ik(model, data, np.array(target), np.pi, np.zeros(6))
        assert np.all(result.q[list(IK_JOINTS)] >= lower[list(IK_JOINTS)] - 1e-9)
        assert np.all(result.q[list(IK_JOINTS)] <= upper[list(IK_JOINTS)] + 1e-9)


def test_ik_does_not_disturb_the_caller_state(model, data) -> None:
    data.qpos[:6] = [0.1, 0.5, 0.9, 1.0, 0.0, 0.01]
    mujoco.mj_forward(model, data)
    before = data.qpos.copy()
    solve_ik(model, data, np.array([0.19, 0.0, 0.09]), np.pi, np.zeros(6))
    assert np.array_equal(data.qpos, before)


def test_jacobian_matches_finite_differences(model, data) -> None:
    q = np.array([0.2, 0.5, 0.8, 1.0, 0.3, 0.01])
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    jacobian = position_jacobian(model, data)
    assert jacobian.shape == (3, 5)

    epsilon = 1e-6
    for joint in range(5):
        perturbed = q.copy()
        perturbed[joint] += epsilon
        moved, _ = forward_kinematics(model, perturbed)
        base, _ = forward_kinematics(model, q)
        assert np.allclose((moved - base) / epsilon, jacobian[:, joint], atol=1e-4)


def test_wrist_roll_does_not_move_the_tool_point(model) -> None:
    """The reason IK solves over four joints and not five."""
    q = np.array([0.1, 0.5, 0.9, 1.0, 0.0, 0.01])
    rolled = q.copy()
    rolled[4] = 1.4
    assert np.allclose(forward_kinematics(model, q)[0], forward_kinematics(model, rolled)[0],
                       atol=1e-9)
