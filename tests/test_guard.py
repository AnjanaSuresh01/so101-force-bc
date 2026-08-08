"""The deployment guard, tested against the simulator rather than a spring model.

tests/test_admittance.py proves the controller's maths bounds force against an
idealised wall. This asks the harder question: with a real arm, real contact
dynamics, a force signal that is an *estimate* rather than a measurement, and a
policy actively trying to push through the workpiece, does the bound survive?
"""

from __future__ import annotations

import numpy as np
import pytest

from griff.calibrate import load_or_fit
from griff.control.guard import ForceGuard
from griff.kinematics import solve_ik
from griff.sim import make_env
from griff.sim.tasks import SPECS

TASKS = tuple(SPECS)


class Rammer:
    """The adversary: drive the tool straight down, hard, forever.

    This is what the guard exists for. A behaviour-cloning policy that has gone
    out of distribution does something like this, and no amount of training-time
    care prevents it.
    """

    def __init__(self, env, descent: float = 0.0025) -> None:
        self.env = env
        self.descent = descent

    def reset(self) -> None:
        self.target = self.env.tool_point().copy()
        self._last = self.env.joint_positions.copy()

    def act(self, observation) -> np.ndarray:
        self.target[2] -= self.descent
        result = solve_ik(self.env.model, self.env.data, self.target, np.pi, self._last)
        if result.converged:
            self._last = result.q.copy()
        action = self._last.copy()
        action[5] = self.env.spec.gripper_closed
        return action


def run(task: str, *, guarded: bool, seed: int = 11, steps: int = 140):
    env = make_env(task, render=False, cameras=(), calibration=load_or_fit(task), seed=seed)
    observation = env.reset(seed=seed)
    rammer = Rammer(env)
    rammer.reset()
    guard = ForceGuard(env) if guarded else None
    if guard:
        guard.reset()
    for _ in range(steps):
        action = rammer.act(observation)
        if guard:
            action = guard.apply(action, observation.force)
        result = env.step(action)
        observation = result.observation
        if result.done:
            break
    peak = env.peak_force_true
    stats = guard.stats if guard else None
    env.close()
    return peak, stats


@pytest.mark.slow
@pytest.mark.parametrize("task", TASKS)
def test_guard_cuts_the_peak_force_a_rammer_produces(task: str) -> None:
    free, _ = run(task, guarded=False)
    guarded, stats = run(task, guarded=True)
    assert free > SPECS[task].overload_force, (
        f"the unguarded rammer only reached {free:.1f} N on {task}; it is not a test of "
        "anything unless it can damage the workpiece"
    )
    assert guarded < 0.75 * free, (
        f"guarded {guarded:.1f} N was not a real reduction on unguarded {free:.1f} N"
    )
    assert stats.lead_clamped_ticks > 0, "nothing was clamped during a deliberate ram"


@pytest.mark.slow
@pytest.mark.parametrize("task", TASKS)
def test_guard_engages_its_lead_limit_against_a_rammer(task: str) -> None:
    """The mechanism that does the bounding must actually be the one firing.

    Against a rammer the *governor* often does not fire at all: the reference is
    already retreating, and it is the arm -- still catching up from a command
    issued ticks earlier -- that keeps driving in. The command-to-measurement
    lead cap is what catches that case, so a run where nothing was clamped is a
    run where this test proved nothing.
    """
    _, stats = run(task, guarded=True)
    assert stats.lead_clamped_ticks > 0


@pytest.mark.slow
@pytest.mark.parametrize("task", TASKS)
def test_guarded_rammer_stays_within_a_measured_margin(task: str) -> None:
    """What the guard actually delivers against a deliberate ram.

    Not the configured limit. Three things sit between the two: the force signal
    is an estimate that lags, the arm lags its command, and first contact is an
    impact whose magnitude is set by approach speed rather than by feedback.
    Measured peaks against the rammer are 7.7 N (peg insertion, 6 N limit,
    8 N overload), 8.5 N (wipe, 7 / 10) and 17.1 N (press fit, 12 / 14) -- so on
    two of the three fixtures the guard holds the workpiece below its damage
    threshold, and on the press fit, where the socket bottoms out on a rigid
    stop, it does not. README and docs/admittance-bound.md say so.
    """
    guarded, _ = run(task, guarded=True)
    limit = SPECS[task].force_limit
    assert guarded < 1.6 * limit, f"{task}: {guarded:.1f} N against a {limit:.1f} N limit"


def test_guard_is_transparent_in_free_space() -> None:
    """No contact, no interference. A guard that perturbs free motion would
    change what the policy does everywhere, not just at contact."""
    task = "peg_insert"
    env = make_env(task, render=False, cameras=(), calibration=load_or_fit(task), seed=3)
    env.reset(seed=3)
    guard = ForceGuard(env)
    guard.reset()

    target = env.tool_point().copy()
    last = env.joint_positions.copy()
    corrections = []
    for _ in range(25):
        target[0] += 0.0006
        result = solve_ik(env.model, env.data, target, np.pi, last)
        last = result.q.copy()
        action = last.copy()
        action[5] = env.spec.gripper_closed
        guarded = guard.apply(action, np.zeros(3))
        corrections.append(np.abs(guarded[:5] - action[:5]).max())
        env.step(guarded)
    env.close()
    assert guard.stats.governed_ticks == 0
    assert max(corrections) < 0.02, "the guard moved the arm while nothing was touching it"


def test_guard_holds_position_when_ik_fails() -> None:
    """An unreachable reference must freeze the arm, not command a partial solve."""
    task = "peg_insert"
    env = make_env(task, render=False, cameras=(), calibration=load_or_fit(task), seed=4)
    env.reset(seed=4)
    guard = ForceGuard(env)
    guard.reset()
    unreachable = np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    first = guard.apply(unreachable, np.zeros(3))
    second = guard.apply(unreachable, np.zeros(3))
    assert np.allclose(first[:5], second[:5], atol=1e-6)


def test_guard_passes_the_gripper_through() -> None:
    task = "press_fit"
    env = make_env(task, render=False, cameras=(), calibration=load_or_fit(task), seed=5)
    env.reset(seed=5)
    guard = ForceGuard(env)
    guard.reset()
    action = env.joint_positions.copy()
    action[5] = 0.0137
    assert guard.apply(action, np.zeros(3))[5] == pytest.approx(0.0137)
