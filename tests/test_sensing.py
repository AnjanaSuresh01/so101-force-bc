"""The contact-force estimator, and the servo read-out it depends on."""

from __future__ import annotations

import numpy as np
import pytest

from griff.sensing import (
    FEATURE_NAMES,
    SERVO_LOAD_COUNTS,
    SERVO_STALL_TORQUE,
    ContactForceEstimator,
    ForceCalibration,
    ServoLoadModel,
    fit_calibration,
    load_features,
)


def _synthetic_sweep(rng: np.random.Generator, samples: int = 600):
    """A fake arm whose free-space load is exactly the model's own basis."""
    q = rng.uniform(-1.5, 1.5, size=(samples, 6))
    qd = rng.normal(0, 0.2, size=(samples, 6))
    qdd = rng.normal(0, 1.0, size=(samples, 6))
    truth = rng.normal(0, 0.4, size=(5, len(FEATURE_NAMES)))
    tau = np.stack([
        np.einsum("jf,jf->j", truth, load_features(a, b, c))
        for a, b, c in zip(q, qd, qdd, strict=True)
    ])
    return q, qd, qdd, tau, truth


def test_calibration_recovers_a_known_model() -> None:
    q, qd, qdd, tau, truth = _synthetic_sweep(np.random.default_rng(0))
    calibration = fit_calibration(q, qd, qdd, tau)
    assert np.allclose(calibration.coefficients, truth, atol=1e-6)
    assert np.all(calibration.residual_rms < 1e-9)


def test_calibration_round_trips_through_json(tmp_path) -> None:
    q, qd, qdd, tau, _ = _synthetic_sweep(np.random.default_rng(1))
    calibration = fit_calibration(q, qd, qdd, tau, note="unit test")
    path = tmp_path / "calibration.json"
    calibration.save(path)
    reloaded = ForceCalibration.load(path)
    assert np.allclose(reloaded.coefficients, calibration.coefficients)
    assert reloaded.note == "unit test"


def test_calibration_rejects_a_mismatched_feature_set(tmp_path) -> None:
    """A calibration fitted with different features must not load silently."""
    path = tmp_path / "stale.json"
    path.write_text(
        '{"feature_names": ["bias"], "coefficients": [[0.0]], '
        '"residual_rms_nm": [0.0], "samples": 1}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different features"):
        ForceCalibration.load(path)


def test_fit_refuses_an_underdetermined_sweep() -> None:
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError, match="at least"):
        fit_calibration(
            rng.normal(size=(4, 6)), rng.normal(size=(4, 6)),
            rng.normal(size=(4, 6)), rng.normal(size=(4, 5)),
        )


def test_estimator_recovers_an_applied_force() -> None:
    """With a perfect model and a well-conditioned Jacobian, F comes back exactly."""
    estimator = ContactForceEstimator(ForceCalibration.identity(), smoothing=1.0, damping=1e-8)
    jacobian = np.array([
        [0.20, 0.0, 0.05, 0.0, 0.0],
        [0.0, 0.18, 0.0, 0.04, 0.0],
        [0.0, 0.0, 0.16, 0.0, 0.03],
    ])
    applied = np.array([1.5, -2.0, 4.0])
    estimate = estimator.estimate(
        np.zeros(6), np.zeros(6), jacobian.T @ applied, jacobian
    )
    assert np.allclose(estimate.force, applied, atol=1e-3)
    assert estimate.magnitude == pytest.approx(np.linalg.norm(applied), rel=1e-3)


def test_estimator_reports_a_blind_direction() -> None:
    """A Jacobian that cannot transmit z must show up as a large condition number."""
    estimator = ContactForceEstimator(ForceCalibration.identity())
    jacobian = np.array([
        [0.20, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.20, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1e-6, 0.0, 0.0],
    ])
    estimate = estimator.estimate(np.zeros(6), np.zeros(6), np.zeros(5), jacobian)
    assert estimate.condition_number > 1e6


def test_estimator_smoothing_reduces_noise() -> None:
    rng = np.random.default_rng(3)
    jacobian = np.eye(3, 5) * 0.2
    heavy = ContactForceEstimator(ForceCalibration.identity(), smoothing=0.1)
    light = ContactForceEstimator(ForceCalibration.identity(), smoothing=1.0)
    heavy_out, light_out = [], []
    for _ in range(200):
        tau = rng.normal(0, 0.05, size=5)
        heavy_out.append(heavy.estimate(np.zeros(6), np.zeros(6), tau, jacobian).magnitude)
        light_out.append(light.estimate(np.zeros(6), np.zeros(6), tau, jacobian).magnitude)
    assert np.std(heavy_out) < np.std(light_out)


def test_estimator_rejects_a_wrong_shaped_jacobian() -> None:
    estimator = ContactForceEstimator()
    with pytest.raises(ValueError, match="3x5"):
        estimator.estimate(np.zeros(6), np.zeros(6), np.zeros(5), np.eye(3))


@pytest.mark.parametrize("smoothing", [0.0, 1.5])
def test_estimator_rejects_an_invalid_smoothing(smoothing: float) -> None:
    with pytest.raises(ValueError):
        ContactForceEstimator(smoothing=smoothing)


def test_servo_load_is_quantised_to_the_bus_resolution() -> None:
    """The STS3215 reports 1/1000 of stall torque. That is the noise floor."""
    servo = ServoLoadModel(noise_nm=0.0, rng=np.random.default_rng(0))
    step = SERVO_STALL_TORQUE / SERVO_LOAD_COUNTS
    reading = servo.read(np.array([0.5001, -1.2345, 0.0, 2.0, -0.00001]))
    assert np.allclose(reading / step, np.round(reading / step))
    assert np.max(np.abs(reading - np.array([0.5001, -1.2345, 0.0, 2.0, -0.00001]))) <= step


def test_servo_load_saturates_at_stall_torque() -> None:
    servo = ServoLoadModel(noise_nm=0.0, rng=np.random.default_rng(0))
    reading = servo.read(np.array([100.0, -100.0, 0.0, 0.0, 0.0]))
    assert reading[0] == pytest.approx(SERVO_STALL_TORQUE, abs=1e-3)
    assert reading[1] == pytest.approx(-SERVO_STALL_TORQUE, abs=1e-3)


def test_load_features_are_finite_at_zero_velocity() -> None:
    features = load_features(np.zeros(6), np.zeros(6), np.zeros(6))
    assert features.shape == (5, len(FEATURE_NAMES))
    assert np.isfinite(features).all()
