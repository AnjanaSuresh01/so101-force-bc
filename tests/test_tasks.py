"""The three task environments, and the properties the experiment rests on."""

from __future__ import annotations

import numpy as np
import pytest

from griff.calibrate import load_or_fit
from griff.sim import make_env
from griff.sim.tasks import SPECS
from griff.teleop import make_operator

TASKS = tuple(SPECS)


@pytest.fixture(scope="module", params=TASKS)
def task(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def env(task: str):
    environment = make_env(task, render=False, cameras=(), calibration=load_or_fit(task), seed=0)
    yield environment
    environment.close()


def test_unknown_task_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown task"):
        make_env("polish_the_silver")


def test_reset_returns_a_complete_observation(env) -> None:
    observation = env.reset(seed=0)
    assert observation.state.shape == (6,)
    assert observation.force.shape == (3,)
    assert np.isfinite(observation.state).all()
    assert np.isfinite(observation.force).all()


def test_reset_is_deterministic_given_a_seed(env) -> None:
    first = env.reset(seed=7)
    parameters = dict(env.episode_parameters)
    second = env.reset(seed=7)
    assert np.allclose(first.state, second.state)
    assert env.episode_parameters == parameters


def test_different_seeds_move_the_workpiece(env) -> None:
    env.reset(seed=1)
    first = env.episode_parameters
    env.reset(seed=2)
    second = env.episode_parameters
    assert first != second


def test_starts_out_of_contact(env) -> None:
    """Every episode must begin with the tool clear of the workpiece.

    If it does not, the first force reading is a startup transient and the
    policy's opening move is a reaction to an artefact.
    """
    env.reset(seed=3)
    assert np.linalg.norm(env.true_force) < 1.0


def test_holding_still_neither_succeeds_nor_overloads(env) -> None:
    """A do-nothing policy must score zero. Guards against a success criterion
    that is satisfied by the reset pose."""
    env.reset(seed=4)
    action = env.home_action.copy()
    for _ in range(40):
        result = env.step(action)
        assert not result.success
    assert env.peak_force_true < env.spec.overload_force


def test_actions_are_clipped_to_joint_limits(env) -> None:
    env.reset(seed=5)
    lower, upper = env.limits
    env.step(np.full(6, 50.0))
    assert np.all(env.data.ctrl <= upper + 1e-9)
    env.step(np.full(6, -50.0))
    assert np.all(env.data.ctrl >= lower - 1e-9)


def test_episode_ends_at_the_step_limit(env) -> None:
    env.reset(seed=6)
    action = env.home_action.copy()
    for _ in range(env.spec.max_steps):
        result = env.step(action)
    assert result.done
    assert env.step_index == env.spec.max_steps


def test_force_estimate_tracks_ground_truth_in_contact(env) -> None:
    """The policy's force channel has to mean something. Push, and check it does."""
    from griff.kinematics import solve_ik

    env.reset(seed=8)
    target = env.tool_point().copy()
    estimated, truth = [], []
    for _ in range(90):
        target[2] -= 0.0009
        result = solve_ik(env.model, env.data, target, np.pi, env.joint_positions)
        action = result.q.copy()
        action[5] = env.spec.gripper_closed
        step = env.step(action)
        estimated.append(step.info["estimated_force_magnitude"])
        truth.append(step.info["true_force_magnitude"])

    estimated, truth = np.array(estimated), np.array(truth)
    contact = truth > 1.0
    assert contact.sum() > 5, "the probe never made contact; the test is not testing anything"
    assert np.corrcoef(truth[contact], estimated[contact])[0, 1] > 0.7
    assert np.sqrt(np.mean((estimated[contact] - truth[contact]) ** 2)) < 4.0


def test_overload_is_reachable(env) -> None:
    """The overload threshold has to be crossable, or force-aware success is
    the same measurement as task success wearing a different name."""
    from griff.kinematics import solve_ik

    env.reset(seed=9)
    target = env.tool_point().copy()
    for _ in range(env.spec.max_steps):
        target[2] -= 0.0015
        result = solve_ik(env.model, env.data, target, np.pi, env.joint_positions)
        action = result.q.copy()
        action[5] = env.spec.gripper_closed
        env.step(action)
    assert env.peak_force_true > env.spec.overload_force


def test_controller_limit_sits_below_the_overload_threshold(task: str) -> None:
    spec = SPECS[task]
    assert spec.force_limit < spec.overload_force


@pytest.mark.slow
def test_the_operator_can_do_the_task(env, task: str) -> None:
    """The demonstrations have to demonstrate. If the scripted operator cannot
    do the task, nothing cloned from it will either."""
    operator = make_operator(task, np.random.default_rng(0))
    successes = 0
    episodes = 6
    for episode in range(episodes):
        observation = env.reset(seed=200 + episode)
        operator.reset(env, np.random.default_rng(200 + episode))
        while True:
            result = env.step(operator.act(env, observation.force))
            observation = result.observation
            if result.done:
                successes += int(result.success)
                break
        assert env.peak_force_true <= env.spec.overload_force, (
            "the operator damaged the workpiece; demonstrations must not contain overloads"
        )
    assert successes >= episodes - 1
