"""Tests for the force-limited admittance controller.

The controller is the safety argument of this repo, so it is tested against a
closed-loop environment rather than by asserting on internal state. `press`
below is a position-controlled arm meeting a linear spring wall: the arm tracks
the commanded reference exactly, and the wall pushes back in proportion to how
far the reference has penetrated it. That is the worst case for a
position-commanded robot -- the real SO-101 has ~850 N/m of its own servo
compliance in series, which only ever reduces the force for a given command.
"""

from __future__ import annotations

import numpy as np
import pytest

from griff.control import AdmittanceConfig, AdmittanceController


def press(
    controller: AdmittanceController,
    *,
    wall_z: float = 0.0,
    env_stiffness: float = 2000.0,
    start_z: float = 0.030,
    commanded_depth: float = 0.050,
    ticks: int = 400,
) -> dict[str, np.ndarray]:
    """Drive the controller straight down into a spring wall and record it.

    The policy ramps its target from `start_z` above the wall to
    `commanded_depth` below it -- i.e. it asks for something that would be
    ruinous if executed literally -- and then holds.
    """
    controller.reset(np.array([0.0, 0.0, start_z]))
    ramp = np.linspace(start_z, -commanded_depth, ticks // 2)
    targets = np.concatenate([ramp, np.full(ticks - ramp.size, -commanded_depth)])

    force = np.zeros(3)
    forces = np.empty(ticks)
    references = np.empty(ticks)
    for i, z in enumerate(targets):
        reference = controller.step(np.array([0.0, 0.0, z]), force)
        penetration = max(0.0, wall_z - reference[2])
        # The wall pushes the tool back up: +z, away from the contact.
        force = np.array([0.0, 0.0, env_stiffness * penetration])
        forces[i] = np.linalg.norm(force)
        references[i] = reference[2]
    return {"force": forces, "reference": references}


def test_free_space_is_transparent() -> None:
    """With no contact the reference must converge onto the policy's target."""
    controller = AdmittanceController()
    controller.reset(np.array([0.0, 0.0, 0.10]))
    target = np.array([0.02, -0.01, 0.08])
    for _ in range(200):
        reference = controller.step(target, np.zeros(3))
    assert np.allclose(reference, target, atol=1e-6)
    assert controller.state.governed_ticks == 0


def test_free_space_respects_the_slew_limit() -> None:
    controller = AdmittanceController()
    controller.reset(np.array([0.0, 0.0, 0.10]))
    first = controller.step(np.array([0.0, 0.0, -0.40]), np.zeros(3))
    travelled = 0.10 - first[2]
    assert travelled == pytest.approx(controller.config.max_step, abs=1e-9)


@pytest.mark.parametrize("env_stiffness", [400.0, 1000.0, 2500.0, 8000.0, 25000.0])
def test_steady_state_force_is_bounded(env_stiffness: float) -> None:
    """However deep the policy commands, the settled force respects the limit.

    This is the claim the whole controller exists to make. The tolerance is the
    deadband: forces below it are treated as noise and not acted on, so the
    steady state is allowed to sit that much above the nominal limit.
    """
    controller = AdmittanceController()
    trace = press(controller, env_stiffness=env_stiffness)
    settled = trace["force"][-60:]
    limit = controller.config.force_limit + controller.config.deadband
    assert settled.max() <= limit, f"settled at {settled.max():.2f} N, limit {limit:.2f} N"


@pytest.mark.parametrize("env_stiffness", [400.0, 1000.0, 2500.0, 8000.0])
def test_peak_force_stays_near_the_limit(env_stiffness: float) -> None:
    """The transient is bounded too, not just the steady state.

    The stiffness estimator starts from a prior and needs a tick or two of
    contact to correct it, so a modest overshoot is expected and is allowed for
    here. docs/admittance-bound.md reports the measured curve, including the
    stiffnesses where this margin stops holding.
    """
    controller = AdmittanceController()
    trace = press(controller, env_stiffness=env_stiffness)
    assert trace["force"].max() <= 1.5 * controller.config.force_limit


def test_without_the_governor_the_same_policy_is_ruinous() -> None:
    """Control experiment: admittance alone does not bound anything.

    Disabling only the advance cap -- by giving the estimator an absurdly soft
    prior and forbidding it from learning -- leaves a pure admittance loop. The
    same commanded penetration then produces an order of magnitude more force,
    which is the reason mechanism 3 exists.
    """
    unbounded = AdmittanceController(
        AdmittanceConfig(
            stiffness_prior=250.0,
            stiffness_bounds=(250.0, 250.1),
            force_limit=1e6,
        )
    )
    trace = press(unbounded, env_stiffness=2500.0)
    guarded = press(AdmittanceController(), env_stiffness=2500.0)
    assert trace["force"].max() > 10 * guarded["force"].max()


def test_governor_never_advances_into_an_over_limit_contact() -> None:
    controller = AdmittanceController()
    controller.reset(np.array([0.0, 0.0, 0.0]))
    over_limit = np.array([0.0, 0.0, controller.config.force_limit * 3])
    previous = np.array([0.0, 0.0, 0.0])
    for _ in range(50):
        reference = controller.step(np.array([0.0, 0.0, -0.10]), over_limit)
        # Motion along -z is motion into the contact; it must never happen.
        assert reference[2] >= previous[2] - 1e-9
        previous = reference
    assert controller.state.governed


def test_contact_release_returns_the_offset_to_zero() -> None:
    controller = AdmittanceController()
    press(controller, env_stiffness=2500.0, ticks=200)
    assert np.linalg.norm(controller.state.offset) > 1e-3
    target = np.array([0.0, 0.0, 0.05])
    for _ in range(300):
        controller.step(target, np.zeros(3))
    assert np.linalg.norm(controller.state.offset) < 1e-4


def test_lower_limit_yields_lower_force() -> None:
    """Monotonicity: the limit is a knob that does what its name says."""
    peaks = []
    for limit in (3.0, 6.0, 12.0):
        controller = AdmittanceController().with_force_limit(limit)
        peaks.append(press(controller, env_stiffness=2500.0)["force"][-60:].max())
    assert peaks[0] < peaks[1] < peaks[2]


def test_integration_is_stable_at_an_absurd_admittance() -> None:
    """Backward Euler must not ring even where explicit integration would blow up."""
    controller = AdmittanceController(
        AdmittanceConfig(mass=0.05, damping=400.0, stiffness=9000.0, force_limit=8.0)
    )
    trace = press(controller, env_stiffness=25000.0, ticks=600)
    assert np.isfinite(trace["force"]).all()
    assert trace["force"].max() < 100.0
    tail = trace["force"][-100:]
    assert tail.std() < 2.0, "steady contact should not oscillate"


def test_stiffness_estimate_converges_on_the_true_wall() -> None:
    controller = AdmittanceController()
    press(controller, env_stiffness=3000.0, ticks=400)
    assert 1500.0 < controller.state.stiffness < 6000.0


def test_deadband_is_continuous_at_the_threshold() -> None:
    controller = AdmittanceController()
    band = controller.config.deadband
    just_under = controller._deadbanded(np.array([0.0, 0.0, band - 1e-6]), band - 1e-6)
    just_over = controller._deadbanded(np.array([0.0, 0.0, band + 1e-6]), band + 1e-6)
    assert np.linalg.norm(just_under - just_over) < 1e-5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0.0},
        {"mass": -1.0},
        {"force_limit": 0.1, "deadband": 0.35},
        {"max_offset": 0.0},
        {"stiffness_bounds": (5000.0, 100.0)},
        {"stiffness_prior": 1.0},
        {"stiffness_smoothing": 0.0},
    ],
)
def test_invalid_configs_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        AdmittanceConfig(**kwargs)


def test_step_rejects_wrong_shapes() -> None:
    controller = AdmittanceController()
    with pytest.raises(ValueError):
        controller.step(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError):
        controller.step(np.zeros(3), np.zeros(6))
