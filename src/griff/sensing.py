"""Contact-force estimation from servo load.

The SO-101 has no force/torque sensor. What it has is six STS3215 servos, each
of which will report a present-load figure over the serial bus. Everything in
this repo that calls itself a force -- the policy's force channel, the
admittance controller's input, the peak-force metric that decides whether a run
counts as a success -- comes from that, and from nothing else.

The estimate is the textbook joint-torque residual:

    tau_ext  =  tau_measured  -  tau_model(q, qdot)
    F        =  argmin_F || J(q)^T F - tau_ext ||^2 + lambda ||F||^2

with ``tau_model`` a static model of everything that loads the servos when
nothing is touching the tool: gravity on the links and payload, Coulomb
friction, and viscous friction. That model is *fitted*, not derived -- exactly
as it would have to be on hardware, where the link inertias are unknown and the
printed gearboxes have friction no datasheet lists. `fit_calibration` runs the
arm through free-space motion, records what the servos report, and solves a
linear least-squares problem for the coefficients.

Two consequences worth being explicit about, because they bound what any claim
built on this signal can mean:

* **The estimate is only as good as the residual.** Anything the static model
  fails to explain shows up as phantom force. The fitted residual RMS is stored
  in the calibration file and reported by `griff sensing calibrate`.
* **It cannot see forces the Jacobian cannot transmit.** A force applied along a
  direction the arm is singular in produces no joint torque and is invisible.
  Near full extension -- where the peg-insertion fixture sits -- the arm is
  poorly conditioned for vertical force, which is precisely the direction that
  matters. `estimate` returns the condition number so this is measurable rather
  than a footnote.

The MuJoCo scenes carry a real F/T sensor at the tool mount as ground truth.
Nothing in the control or policy path reads it; it exists so that
`griff sensing validate` can report the error of this estimator against it
instead of assuming it away.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Feature names of the load model, in fit order.
FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "sin_q1",
    "cos_q1",
    "sin_q12",
    "cos_q12",
    "sin_q123",
    "cos_q123",
    "coulomb",
    "viscous",
    "accel",
)

#: STS3215 stall torque at 12 V, from the datasheet's 30 kgf.cm.
SERVO_STALL_TORQUE = 2.94  # N.m
#: The bus reports load as a signed 0-1000 count of stall torque.
SERVO_LOAD_COUNTS = 1000


def load_features(q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
    """Regressor row for the free-space load model.

    The three pitch joints share an axis, so the gravity torque on each depends
    only on the cumulative pitch angles q1, q1+q2 and q1+q2+q3. Sine terms carry
    the gravity; the cosine terms are there to absorb mounting offsets, and the
    fit is free to zero them.

    Coulomb friction uses tanh rather than sign. A hard sign flips the predicted
    torque discontinuously at zero velocity, which on a servo that dithers around
    a held position injects a spike into the force estimate several times a
    second -- as phantom contact, at rest.

    The `accel` term is a diagonal inertia model, and it earns its place. Without
    it, the servo torque needed to accelerate a link is unexplained, and the
    estimator reports it as external force: a 3 cm/s tool move leaves an 80 mrad
    tracking error at the shoulder, which reads as several newtons of contact
    that nothing is touching. Off-diagonal coupling stays unmodelled and is part
    of what the reported residual measures. It is fittable on hardware for the
    same reason it is here -- differencing the position reads twice gives you
    `qdd` without any sensor the arm does not already have.
    """
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    qdd = np.asarray(qdd, dtype=float)
    c1 = q[1]
    c12 = q[1] + q[2]
    c123 = q[1] + q[2] + q[3]
    return np.stack(
        [
            np.ones(5),
            np.full(5, np.sin(c1)),
            np.full(5, np.cos(c1)),
            np.full(5, np.sin(c12)),
            np.full(5, np.cos(c12)),
            np.full(5, np.sin(c123)),
            np.full(5, np.cos(c123)),
            np.tanh(qd[:5] / 0.02),
            qd[:5],
            qdd[:5],
        ],
        axis=1,
    )


@dataclass(frozen=True)
class ForceCalibration:
    """Fitted static load model, one row of coefficients per arm joint."""

    coefficients: np.ndarray  # (5, len(FEATURE_NAMES))
    residual_rms: np.ndarray  # (5,) N.m, from the fit
    samples: int
    note: str = ""

    def predict(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        """Servo torque expected with nothing touching the tool."""
        return np.einsum("jf,jf->j", self.coefficients, load_features(q, qd, qdd))

    def to_json(self) -> str:
        return json.dumps(
            {
                "feature_names": list(FEATURE_NAMES),
                "coefficients": self.coefficients.tolist(),
                "residual_rms_nm": self.residual_rms.tolist(),
                "samples": self.samples,
                "note": self.note,
            },
            indent=2,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ForceCalibration:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if tuple(payload["feature_names"]) != FEATURE_NAMES:
            raise ValueError(
                "calibration was fitted with different features "
                f"({payload['feature_names']}); refit it with `griff sensing calibrate`"
            )
        return cls(
            coefficients=np.asarray(payload["coefficients"], dtype=float),
            residual_rms=np.asarray(payload["residual_rms_nm"], dtype=float),
            samples=int(payload["samples"]),
            note=payload.get("note", ""),
        )

    @classmethod
    def identity(cls) -> ForceCalibration:
        """A do-nothing calibration, for tests that want the raw residual."""
        return cls(
            coefficients=np.zeros((5, len(FEATURE_NAMES))),
            residual_rms=np.zeros(5),
            samples=0,
            note="uncalibrated",
        )


def fit_calibration(
    q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, tau: np.ndarray, *, note: str = ""
) -> ForceCalibration:
    """Least-squares fit of the free-space load model.

    `q`, `qd`, `qdd` and `tau` are (N, >=5) arrays recorded with nothing in
    contact. The sweep behind them has to contain both quasi-static poses and
    genuine acceleration, or the gravity and inertia terms are not separately
    identifiable and the fit will trade one off against the other.
    """
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    qdd = np.asarray(qdd, dtype=float)
    tau = np.asarray(tau, dtype=float)
    if not (len(q) == len(qd) == len(qdd) == len(tau)):
        raise ValueError("q, qd, qdd and tau must have the same number of samples")
    if len(q) < len(FEATURE_NAMES) * 5:
        raise ValueError(
            f"need at least {len(FEATURE_NAMES) * 5} samples to fit "
            f"{len(FEATURE_NAMES)} features per joint; got {len(q)}"
        )

    design = np.stack(
        [load_features(a, b, c) for a, b, c in zip(q, qd, qdd, strict=True)]
    )
    coefficients = np.empty((5, len(FEATURE_NAMES)))
    residual_rms = np.empty(5)
    for joint in range(5):
        a = design[:, joint, :]
        b = tau[:, joint]
        solution, *_ = np.linalg.lstsq(a, b, rcond=None)
        coefficients[joint] = solution
        residual_rms[joint] = float(np.sqrt(np.mean((a @ solution - b) ** 2)))
    return ForceCalibration(coefficients, residual_rms, samples=len(q), note=note)


@dataclass(frozen=True)
class ForceEstimate:
    """One tick of the estimator."""

    force: np.ndarray  # (3,) N, base frame
    residual_torque: np.ndarray  # (5,) N.m
    condition_number: float  # of J J^T; large means a blind direction

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self.force))


class ContactForceEstimator:
    """Servo load -> Cartesian contact force at the tool.

    `smoothing` is a first-order low-pass on the output. The raw residual is
    noisy for an unglamorous reason: the STS3215 reports load as one part in a
    thousand of stall torque, so the quantisation step alone is ~3 mN.m, which
    the Jacobian transpose turns into tenths of a newton.
    """

    def __init__(
        self,
        calibration: ForceCalibration | None = None,
        *,
        # The Jacobian of a 0.2 m arm has singular values around 0.2, so the
        # Gram matrix eigenvalues sit near 0.04. Damping of comparable size
        # shrinks the solution by tens of percent -- an estimator that reads
        # half the true force and looks plausible while doing it. Keep it two
        # orders below the smallest eigenvalue it is there to protect against.
        damping: float = 5e-4,
        smoothing: float = 0.4,
        dt: float = 1.0 / 30.0,
    ) -> None:
        if not 0 < smoothing <= 1:
            raise ValueError("smoothing must be in (0, 1]")
        if damping <= 0:
            raise ValueError("damping must be positive")
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.calibration = calibration or ForceCalibration.identity()
        self.damping = damping
        self.smoothing = smoothing
        self.dt = dt
        self._filtered: np.ndarray = np.zeros(3)
        self._previous_qd: np.ndarray | None = None

    def reset(self) -> None:
        self._filtered = np.zeros(3)
        self._previous_qd = None

    def estimate(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        tau_measured: np.ndarray,
        jacobian: np.ndarray,
    ) -> ForceEstimate:
        """`jacobian` is the 3x5 positional Jacobian of the tool point.

        Joint acceleration is differenced from the velocity of the previous
        tick, which is what you can do on hardware and is why the model uses it.
        The first tick after a reset has no previous velocity and is treated as
        stationary.
        """
        jacobian = np.asarray(jacobian, dtype=float)
        if jacobian.shape != (3, 5):
            raise ValueError(f"expected a 3x5 positional Jacobian, got {jacobian.shape}")

        qd = np.asarray(qd, dtype=float)
        previous = self._previous_qd if self._previous_qd is not None else qd
        qdd = (qd[:5] - previous[:5]) / self.dt
        self._previous_qd = qd.copy()

        residual = np.asarray(tau_measured, dtype=float)[:5] - self.calibration.predict(q, qd, qdd)
        gram = jacobian @ jacobian.T
        regularised = gram + self.damping * np.eye(3)
        raw = np.linalg.solve(regularised, jacobian @ residual)

        self._filtered = (1 - self.smoothing) * self._filtered + self.smoothing * raw
        eigenvalues = np.linalg.eigvalsh(gram)
        condition = float(eigenvalues.max() / max(eigenvalues.min(), 1e-12))
        return ForceEstimate(self._filtered.copy(), residual, condition)


class ServoLoadModel:
    """What the STS3215 bus actually returns, given a true joint torque.

    Simulating the read-out rather than handing the estimator exact torques is
    the difference between a force channel that works in this repo and one that
    would work on the bench. The bus reports load as a signed count out of 1000
    of stall torque, so the resolution is ~3 mN.m however clean the physics is,
    and the reading carries noise on top of that.
    """

    def __init__(
        self,
        *,
        noise_nm: float = 0.012,
        quantisation_counts: int = SERVO_LOAD_COUNTS,
        stall_torque: float = SERVO_STALL_TORQUE,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.noise_nm = noise_nm
        self.step = stall_torque / quantisation_counts
        self.stall_torque = stall_torque
        self.rng = rng or np.random.default_rng(0)

    def read(self, tau_true: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau_true, dtype=float)
        noisy: np.ndarray = tau + self.rng.normal(0.0, self.noise_nm, size=tau.shape)
        clipped: np.ndarray = np.clip(noisy, -self.stall_torque, self.stall_torque)
        return np.asarray(np.round(clipped / self.step) * self.step)
