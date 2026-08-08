"""The three contact-rich tasks, and what makes each of them a *force* task.

Every one of them is randomised along an axis that the 96x96 camera views
cannot resolve but the force channel can. That is deliberate and it is the
experiment: if the vision-only and vision+force ablations are to say anything,
the tasks have to contain information that only force carries. Otherwise the
ablation measures nothing but parameter count.

    peg_insert   a fine position offset of +/- 3 mm on top of a visible coarse
                 one. At 96x96 over a ~0.3 m field of view, one pixel is about
                 3 mm, so the fine term is at the resolution limit -- and at the
                 moment it matters the peg is inside the bore, hidden from both
                 cameras.
    wipe         a plate tilt of +/- 4 degrees. Six millimetres of height change
                 across the stroke, and a couple of pixels of foreshortening.
    press_fit    retainer stiffness, +/- 20%. Inside the socket. Not observable
                 by any camera at any resolution.

Overload thresholds are set below what the arm can actually produce -- the
STS3215's 2.94 N.m at a 0.2 m reach caps contact around 15 N -- so "succeeded by
overloading the workpiece" is a reachable outcome rather than a hypothetical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from griff.sim.env import TaskEnv


@dataclass(frozen=True)
class TaskSpec:
    name: str
    scene: str
    description: str  # the natural-language task string stored in the dataset
    held_body: str
    held_joint: str
    weld_offset: float  # m along the tool axis
    max_steps: int
    overload_force: float  # N; above this the workpiece is considered damaged
    gripper_open: float = 0.016
    gripper_closed: float = 0.002
    #: Force limit handed to the admittance controller for this task. Below the
    #: overload threshold, so the controller has margin to work in.
    force_limit: float = 6.0
    #: Virtual spring stiffness of the admittance, N/m. This does NOT set the
    #: force bound -- the reference governor does that -- but it decides how much
    #: of the policy's positional authority the compliance consumes, because the
    #: steady-state offset is F / K_a. A task that must *sustain* force needs a
    #: stiff virtual spring or the compliance simply undoes the press: at
    #: K_a = 250 N/m the scripted operator seats the press-fit part 0/6 times,
    #: at 2200 N/m it seats it 6/6, and the governor fires in neither case.
    admittance_stiffness: float = 250.0
    #: Where the workpiece nominally sits, before per-episode randomisation.
    #: Set by reach, not by convenience: with the tool vertical the SO-101 can
    #: hold a point about 0.24 m out, and the wipe stroke alone spans 80 mm.
    workpiece_xy: tuple[float, float] = (0.185, 0.0)
    metrics: tuple[str, ...] = ()


PEG_INSERT = TaskSpec(
    name="peg_insert",
    scene="peg_insert",
    description="Insert the peg into the hole",
    held_body="peg",
    held_joint="peg_free",
    weld_offset=0.044,
    max_steps=170,
    workpiece_xy=(0.190, 0.0),
    overload_force=8.0,
    force_limit=6.0,
    metrics=("insertion_depth_mm", "lateral_error_mm"),
)

WIPE = TaskSpec(
    name="wipe",
    scene="wipe",
    description="Wipe the surface of the plate end to end",
    held_body="pad",
    held_joint="pad_free",
    weld_offset=0.050,
    max_steps=210,
    workpiece_xy=(0.200, 0.0),
    overload_force=10.0,
    force_limit=7.0,
    # Sustains 2-6 N for the whole stroke.
    admittance_stiffness=900.0,
    metrics=("coverage", "in_band_fraction"),
)

PRESS_FIT = TaskSpec(
    name="press_fit",
    scene="press_fit",
    description="Press the part into the socket until it seats",
    held_body="part",
    held_joint="part_free",
    weld_offset=0.054,
    max_steps=190,
    workpiece_xy=(0.190, 0.0),
    overload_force=14.0,
    force_limit=12.0,
    # Has to hold 6-9 N against the retainer to seat at all.
    admittance_stiffness=2200.0,
    metrics=("seat_depth_mm", "seating_force_n"),
)


class PegInsertEnv(TaskEnv):
    """Insert a 17 mm peg into a 20 mm bore with 1.5 mm of radial clearance."""

    spec = PEG_INSERT

    #: Top face of the bore, in world z.
    BORE_TOP = 0.048
    #: How far below the bore top the peg tip must reach to count as inserted.
    INSERT_DEPTH = 0.026
    #: Coarse offset -- several pixels at 96x96, so a camera can resolve it.
    COARSE_OFFSET = 0.012
    #: Fine offset -- about one pixel. It cannot.
    FINE_OFFSET = 0.003

    def _randomise(self, rng: np.random.Generator) -> dict[str, Any]:
        coarse = rng.uniform(-self.COARSE_OFFSET, self.COARSE_OFFSET, size=2)
        fine = rng.uniform(-self.FINE_OFFSET, self.FINE_OFFSET, size=2)
        base = np.array(self.spec.workpiece_xy)
        self._hole_xy = base + coarse + fine
        self.model.body_pos[self.body_id("fixture")][:2] = self._hole_xy
        return {
            "hole_x": float(self._hole_xy[0]),
            "hole_y": float(self._hole_xy[1]),
            "coarse_offset_mm": [float(v * 1000) for v in coarse],
            "unresolvable_offset_mm": [float(v * 1000) for v in fine],
        }

    def _home_target(self, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        # Start above the fixture with the peg tip clear of the entry lip, with
        # enough jitter that the policy cannot memorise a single opening move.
        jitter = rng.uniform(-0.009, 0.009, size=2)
        target = np.array([self._hole_xy[0] + jitter[0], self._hole_xy[1] + jitter[1], 0.108])
        return target, float(np.pi + rng.uniform(-0.04, 0.04))

    def _peg_tip(self) -> np.ndarray:
        return self.site_position("peg_tip")

    def _success(self) -> bool:
        tip = self._peg_tip()
        deep_enough = tip[2] <= self.BORE_TOP - self.INSERT_DEPTH
        centred = np.linalg.norm(tip[:2] - self._hole_xy) < 0.006
        return bool(deep_enough and centred)

    def _task_info(self) -> dict[str, Any]:
        tip = self._peg_tip()
        return {
            "insertion_depth_mm": float(max(0.0, self.BORE_TOP - tip[2]) * 1000),
            "lateral_error_mm": float(np.linalg.norm(tip[:2] - self._hole_xy) * 1000),
        }


class WipeEnv(TaskEnv):
    """Sweep a compliant pad across a randomly tilted plate, on the force band."""

    spec = WIPE

    #: The pad must be pressed at least this hard to count as wiping, not hovering.
    CONTACT_MIN = 1.5  # N
    #: Above this the wipe is scrubbing rather than wiping. Used for the band metric.
    CONTACT_MAX = 6.0  # N
    #: Stroke checkpoints along the plate's long axis (y), as offsets from its
    #: centre. Lateral, so the whole stroke sits at constant radius from the
    #: shoulder -- see the comment in task_wipe.xml.
    CHECKPOINTS = (-0.040, -0.020, 0.0, 0.020, 0.040)
    CHECKPOINT_RADIUS = 0.011  # m
    MAX_TILT = np.deg2rad(4.0)

    def _randomise(self, rng: np.random.Generator) -> dict[str, Any]:
        shift = rng.uniform(-0.008, 0.008, size=2)
        self._plate_xy = np.array(self.spec.workpiece_xy) + shift
        plate = self.body_id("plate")
        self.model.body_pos[plate][:2] = self._plate_xy

        roll, pitch = rng.uniform(-self.MAX_TILT, self.MAX_TILT, size=2)
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, np.array([roll, pitch, 0.0]), "xyz")
        self.model.body_quat[plate] = quat
        self._tilt = (roll, pitch)

        self._visited = [False] * len(self.CHECKPOINTS)
        self._band_ticks = 0
        self._contact_ticks = 0
        return {
            "plate_x": float(self._plate_xy[0]),
            "plate_y": float(self._plate_xy[1]),
            "tilt_roll_deg": float(np.rad2deg(roll)),
            "tilt_pitch_deg": float(np.rad2deg(pitch)),
        }

    def _home_target(self, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        start_y = self._plate_xy[1] + self.CHECKPOINTS[0] + rng.uniform(-0.005, 0.005)
        target = np.array([self._plate_xy[0] + rng.uniform(-0.006, 0.006), start_y, 0.098])
        return target, float(np.pi + rng.uniform(-0.03, 0.03))

    def _update_coverage(self) -> None:
        face = self.site_position("pad_face")
        magnitude = float(np.linalg.norm(self.true_force))
        if magnitude >= self.CONTACT_MIN:
            self._contact_ticks += 1
            if magnitude <= self.CONTACT_MAX:
                self._band_ticks += 1
            for index, offset in enumerate(self.CHECKPOINTS):
                centre = np.array([self._plate_xy[0], self._plate_xy[1] + offset])
                if np.linalg.norm(face[:2] - centre) <= self.CHECKPOINT_RADIUS:
                    self._visited[index] = True

    def _success(self) -> bool:
        self._update_coverage()
        return all(self._visited)

    def _task_info(self) -> dict[str, Any]:
        contact = max(1, self._contact_ticks)
        return {
            "coverage": sum(self._visited) / len(self.CHECKPOINTS),
            "in_band_fraction": self._band_ticks / contact,
            "contact_ticks": self._contact_ticks,
        }


class PressFitEnv(TaskEnv):
    """Press a part into a socket against a spring retainer of unknown stiffness."""

    spec = PRESS_FIT

    #: Retainer travel that counts as seated. Set against the 14 mm hard stop
    #: so seating needs 6-9 N depending on the episode's stiffness draw --
    #: inside what the operator will press, and well under the 14 N overload,
    #: so both failure modes stay reachable rather than one dominating.
    SEAT_DEPTH = 0.011  # m
    DWELL_TICKS = 6  # how long it must stay seated (0.2 s at 30 Hz)
    NOMINAL_STIFFNESS = 700.0  # N/m, as authored in the scene
    STIFFNESS_SPREAD = 0.20

    def _randomise(self, rng: np.random.Generator) -> dict[str, Any]:
        shift = rng.uniform(-0.008, 0.008, size=2)
        self._socket_xy = np.array(self.spec.workpiece_xy) + shift
        self.model.body_pos[self.body_id("socket")][:2] = self._socket_xy

        scale = rng.uniform(1 - self.STIFFNESS_SPREAD, 1 + self.STIFFNESS_SPREAD)
        stiffness = self.NOMINAL_STIFFNESS * scale
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "retainer_slide")
        self.model.jnt_stiffness[joint] = stiffness
        self._retainer_qpos_adr = self.model.jnt_qposadr[joint]
        self._seated_ticks = 0
        self._seating_force = float(stiffness * self.SEAT_DEPTH)
        return {
            "socket_x": float(self._socket_xy[0]),
            "socket_y": float(self._socket_xy[1]),
            "retainer_stiffness_n_per_m": float(stiffness),
            # The force the part must sustain to seat. Not observable by camera.
            "seating_force_n": self._seating_force,
        }

    def _home_target(self, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        jitter = rng.uniform(-0.008, 0.008, size=2)
        target = np.array([self._socket_xy[0] + jitter[0], self._socket_xy[1] + jitter[1], 0.106])
        return target, float(np.pi + rng.uniform(-0.03, 0.03))

    def _retainer_travel(self) -> float:
        return float(-self.data.qpos[self._retainer_qpos_adr])

    def _success(self) -> bool:
        if self._retainer_travel() >= self.SEAT_DEPTH:
            self._seated_ticks += 1
        else:
            self._seated_ticks = 0
        return self._seated_ticks >= self.DWELL_TICKS

    def _task_info(self) -> dict[str, Any]:
        return {
            "seat_depth_mm": self._retainer_travel() * 1000,
            "seating_force_n": self._seating_force,
            "seated_ticks": self._seated_ticks,
        }


TASKS: dict[str, type[TaskEnv]] = {
    PEG_INSERT.name: PegInsertEnv,
    WIPE.name: WipeEnv,
    PRESS_FIT.name: PressFitEnv,
}

SPECS: dict[str, TaskSpec] = {
    PEG_INSERT.name: PEG_INSERT,
    WIPE.name: WIPE,
    PRESS_FIT.name: PRESS_FIT,
}


def make_env(task: str, **kwargs: Any) -> TaskEnv:
    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}; known tasks: {sorted(TASKS)}")
    return TASKS[task](**kwargs)
