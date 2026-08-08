# so101-force-bc

**Contact-rich behaviour cloning on an SO-101 — and the scoring rule that stops
a policy from passing by crushing the part.**

[![CI](https://github.com/AnjanaSuresh01/so101-force-bc/actions/workflows/ci.yml/badge.svg)](https://github.com/AnjanaSuresh01/so101-force-bc/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![ROS 2 Jazzy](https://img.shields.io/badge/ROS%202-Jazzy-22314E.svg)](docs/ros2.md)

Three contact-rich tasks on a 6-joint SO-101: peg insertion, surface wiping, and
a force-limited press fit. Demonstrations captured through a leader-follower
teleoperation rig into LeRobot v2.1 datasets. ACT and Diffusion Policy cloned
from them, ablated on whether they can feel contact. Deployed behind a
force-limited admittance controller. Scored on task success **and** peak contact
force, with runs that succeed by overloading the workpiece counted as failures.

<!-- HEADLINE-PLACEHOLDER: replace with real `griff evaluate` output before publishing -->

```console
$ griff evaluate --task peg_insert
```

See [results/RESULTS.md](results/RESULTS.md) for the measured tables.

---

## The problem this is built around

Manipulation policies are reported on success rate. On contact-rich tasks that
is a metric with a cheap exploit: **press harder.** A peg that will not align
goes in if you drive it through the chamfer. A part that will not seat seats if
you bottom out the retainer. Both count as successes, and both would have
destroyed the workpiece on real hardware.

So this repo measures two things and reports both:

- **task success** — the task's own criterion
- **force-aware success** — that, *and* peak contact force stayed under the
  threshold at which the workpiece is considered damaged

The second is not a stricter version of the first. It can reorder the ranking,
and in [results/RESULTS.md](results/RESULTS.md) it does.

Two supporting decisions make that measurement mean something:

**Peak force is graded from ground truth, not from the robot's own signal.** The
policy and the controller both run on a force *estimate* derived from servo
load. The metric reads the simulator's F/T sensor, which nothing in the control
path can see. Grading a safety property with the same noisy signal the system
used to try to satisfy it would let estimator error hide exactly the failures
being looked for.

**Overload thresholds are reachable.** They sit below what the arm can produce —
an STS3215 delivers 2.94 N·m, which at a 0.19 m reach caps contact near 15 N.
`tests/test_tasks.py` rams each fixture and requires the threshold to be crossed,
so "succeeded by overloading" is an outcome that can actually happen rather than
a hypothetical the metric is safe from.

---

## There is no force sensor on this robot

The SO-101 has six STS3215 servos and nothing else. Every number here that calls
itself a force comes from `Present_Load` — an 11-bit field, about 3 mN·m of
resolution — put through a fitted static model and a Jacobian transpose:

```
tau_ext = tau_measured - tau_model(q, qdot, qddot)
F       = argmin_F || J(q)^T F - tau_ext ||^2 + lambda ||F||^2
```

`tau_model` is fitted, not derived, because on hardware you do not know the link
inertias and printed gearboxes have friction no datasheet lists. `griff calibrate`
runs the arm through free space with the workpiece removed and solves for it.

Measured against ground truth: **0.96–3.12 N RMSE** in each task's operating
band, r ≥ 0.97, and 0.07–1.09 N with nothing touching the tool. It saturates
around 30 N because the servos do. Full numbers and failure modes in
[docs/force-estimation.md](docs/force-estimation.md).

The `qddot` term in that model was not there at first, and its absence is worth
recording because it looked like a policy bug rather than a sensing bug: without
an inertia term, the torque needed to accelerate a link is unexplained, so a
3 cm/s tool move left an 80 mrad tracking error at the shoulder that read out as
**3–6 N of contact force with nothing in the scene**. The scripted operator,
regulating on that, lifted away from workpieces it had never touched.

---

## The ablation has three arms, not two

`vision_force` carries ~17k more parameters than `vision` — small, but not
nothing. A gain over vision-only could be capacity rather than information. So
there is a third arm:

| arm | inputs | parameters |
|---|---|---|
| `vision` | images + joint state | 1,059,878 |
| `vision_zeroforce` | images + joint state + a force branch fed a constant zero | 1,077,030 |
| `vision_force` | images + joint state + the force estimate | 1,077,030 |

`vision_force − vision_zeroforce` isolates what the *signal* contributes with
capacity held exactly fixed. `vision_zeroforce − vision` shows what those
parameters buy on their own. Everything else — backbone, width, depth, chunk
length, optimiser, seed, data — is identical across the three, and
`tests/test_policies.py` asserts that the blinded arm's output is byte-identical
when the force input is multiplied by 100.

That only measures anything if the tasks contain information force alone
carries, so each is randomised along an axis the 96×96 cameras cannot resolve:
a ±3 mm hole offset (about one pixel, and occluded at the moment it matters), a
±4° plate tilt, and a ±20% retainer stiffness that is inside the socket.
[docs/tasks.md](docs/tasks.md) has the details.

All 18 cells were trained identically: 3,000 steps at batch 32, AdamW, cosine
schedule, seed 0, on 60 demonstrations per task, CPU-only. That is a modest
budget — ACT's own experiments run far longer — and the absolute success rates
should be read as "what this much data and compute buys", not as a ceiling. The
comparison between cells is what the budget was spent to make fair.

---

## What the safety layer does and does not guarantee

The policy's output never reaches the arm. It is a *request*:

```
q_policy --FK--> Cartesian target
                 admittance + reference governor, on the force estimate
                 admissible target --IK--> q_command
```

Three mechanisms, and only one of them bounds anything:

1. **Admittance** — a virtual mass-damper-spring makes the arm yield on contact.
   On its own this does *not* bound force. Steady state settles at
   `d · K_env·K_a/(K_env+K_a)` where `d` is how deep the policy commanded, so a
   policy asking for 5 cm of penetration still generates a large force, just
   more gently.
2. **A reference governor** — while the force is at the limit, the reference may
   not advance into the contact. This bounds the steady state whatever the
   policy asks for, because it is a projection on the command itself.
3. **Online stiffness estimation** — a governor that fires only *after* an
   over-limit tick still lets one unthrottled step slam into the fixture, so
   each tick's advance is capped at the distance predicted to consume the
   remaining force headroom.

Compliance acts along the **tool axis only**, and that is not a refinement.
Yielding in all three axes means yielding to friction, and friction opposes
motion: with three-axis compliance the scripted operator scored 0/20 on the
wiping task — dragged several millimetres behind its own reference for the whole
stroke, with the governor firing on exactly zero ticks. The virtual spring's
stiffness is per task for the same class of reason: at 250 N/m, holding the 9 N
a press fit needs costs more offset than the controller has, and the part never
seats. Neither changes the bound; both decide whether the arm can still do the
job. [docs/admittance-bound.md](docs/admittance-bound.md) has the measurements.

Measured: **settled force lands exactly on the limit at every environment
stiffness from 400 to 60,000 N/m**, and the worst impact transient across that
sweep is 1.51× the limit. First contact is an impact, and impact is set by
approach speed and stiffness rather than by feedback — no controller commanding
position at 30 Hz avoids that, and [docs/admittance-bound.md](docs/admittance-bound.md)
publishes the curve rather than claiming otherwise.

The guard is not free. On peg insertion it costs task success and buys peak
force; both directions are in [results/RESULTS.md](results/RESULTS.md).

---

## Results

Full tables, with 95% Wilson intervals and paired episode seeds, in
[results/RESULTS.md](results/RESULTS.md) — generated by `griff report`, never
hand-edited, and CI fails on a diff.

---

## The ROS 2 stack

`ros2_ws/` holds five packages: messages, description (URDF + SRDF), the
force-limited admittance controller as a `controller_interface` plugin, the
policy and force-estimator nodes, and bringup.

**It builds in CI against Jazzy and has never run on hardware.** That is the
whole claim, stated in [docs/ros2.md](docs/ros2.md) before anything else.

What makes it more than a stub: the controller's maths core is a line-for-line
C++ port of the Python one, and `test/test_admittance_core.cpp` asserts the same
bound against the same spring-wall model with the same constants. If the
implementation that would run on the robot and the implementation that produced
the published numbers ever disagree, CI says so. And
`tests/test_urdf_matches_mjcf.py` keeps the URDF and the MuJoCo model describing
the same robot — joint limits, axes, link offsets and the tool centre point —
because the policies are trained against one and the controller computes
kinematics from the other.

---

## Reproducing

```console
$ pip install -e '.[dev]'
$ griff reproduce          # calibrate, record, train the grid, evaluate. Hours.
```

Or the steps separately:

```console
$ griff calibrate                    # fit the force estimator, per task
$ griff validate-sensing             # measure it against ground truth
$ griff record --episodes 60         # teleoperate into LeRobot datasets
$ griff validate-dataset datasets/peg_insert
$ griff train-grid --steps 3000      # 3 tasks x 2 policies x 3 conditionings
$ griff evaluate --episodes 25       # scored on success and peak force
```

Recording and training write large, regenerable artefacts. Point `GRIFF_DATASETS`
and `GRIFF_RUNS` somewhere outside a cloud-synced folder — on the machine this
was built on, OneDrive indexing 30,000 recorded frames took more CPU than
training did.

---

## What is measured, and what is not

**Measured, on this machine:** the force estimator against ground truth; the
admittance bound across a stiffness sweep; dataset conformance; every number in
`results/RESULTS.md`, from 25 paired rollouts per policy.

**Written and tested, but never executed against the thing it models:** the
Feetech STS3215 driver. Packet framing, checksums, position encoding and the
signed load field are unit-tested through a fake port; no servo has answered it.

**Written and built, but never run:** the ROS 2 stack. It compiles in CI against
Jazzy and its C++ controller core passes the same tests as the Python one. There
is no controller manager, no MoveIt instance, and no arm behind any of it. There
is deliberately no Feetech hardware interface — the URDF names
`griff_hardware/NotImplemented` so a non-mock launch fails loudly rather than
looking like it might work.

**Simulated, not real:** everything. There is no SO-101 here. The arm is a
kinematically faithful stand-in — SO-101 joint set, link lengths, joint limits
and the STS3215's torque limit, with primitive collision geometry. Tools are
welded into the jaws rather than grasped, because grasping is not what is being
studied. Demonstrations come from a scripted operator with a per-episode aiming
error it cannot see, not from a human.

**A known confound:** ACT re-plans every tick with temporal ensembling and
Diffusion Policy re-plans every 8 with a receding horizon, because that is what
each paper specifies. It makes the *between-policy* comparison imperfect. The
force ablation, which is what this repo is built around, holds the policy and
its inference scheme fixed and changes only whether force is in the observation.

---

## Layout

```
src/griff/
  kinematics.py     FK, Jacobians, and the 4-DoF IK the SO-101 actually admits
  sensing.py        contact-force estimation from servo load
  calibrate.py      fitting that model, and validating it against ground truth
  control/          the force-limited admittance controller and the deploy guard
  sim/              MuJoCo environments for the three tasks
  teleop/           leader-follower rig, Feetech protocol, scripted operators
  data/             LeRobot v2.1 writer, reader, validator, episode recorder
  policies/         ACT, Diffusion Policy, and the conditioning switch
  train.py          one behaviour-cloning loop for both
  evaluate.py       rollouts scored on success and peak contact force
ros2_ws/src/        five ROS 2 packages, built in CI
docs/               how the force estimate works, what the bound covers, the tasks
results/            results.json and the generated RESULTS.md
```

MIT licensed.
