# The ROS 2 stack

**Status: built in CI against Jazzy. Never run on hardware, never run against a
live controller manager, never run against MoveIt.** The CI badge on the README
is the entire claim this stack makes about itself. Everything below describes
what the code does when read; nothing below has been observed.

That is stated first because it is the thing a reader most needs to know, and
because the alternative -- a launch file and a controller that *look* deployed --
is the failure mode this repo is trying not to have.

## Shape

```
policy_node ──/griff/policy_command──┐
                                     ├─► force_limited_admittance_controller ──► arm
force_estimator_node ──/griff/contact_force──┘
```

The policy has **no path to the arm that does not pass through the admittance
controller**. That is the design, and it is why `policy_node` publishes a
request topic instead of claiming a command interface. A policy cannot express a
command the guard has not approved, because the guard re-derives the joint
command from a Cartesian reference it has authority over.

MoveIt Servo, when enabled with `use_servo:=true`, publishes into the *same*
guarded topic. A human jogging the arm into a fixture is subject to the same
force limit the policy is.

## Packages

| package | build type | what it is |
|---|---|---|
| `griff_msgs` | ament_cmake | `ContactForce`, `GuardStatus` |
| `griff_description` | ament_cmake | URDF/xacro and SRDF, dimensions locked to the MuJoCo model |
| `griff_control` | ament_cmake | the force-limited admittance controller, a `controller_interface` plugin |
| `griff_policy` | ament_python | the policy node and the force-estimator node |
| `griff_bringup` | ament_cmake | launch and configuration |

## The controller

`griff_control::ForceLimitedAdmittanceController` claims position command
interfaces for all six joints and does, every cycle:

```
q_policy  --FK-->  x_policy
                   admittance + reference governor, on the force estimate
                   x_ref
dq = J^T (J J^T + lambda^2 I)^-1 (x_ref - x_policy)
command = q_policy + dq
```

Kinematics are KDL, built from the controller's own `robot_description`, so the
controller does not depend on a running MoveIt instance.

`include/griff_control/admittance_core.hpp` is a line-for-line port of
`griff.control.admittance` in Python. The duplication is deliberate: the Python
version produced every number in `results/RESULTS.md`, and this version is what
would run on the robot. `test/test_admittance_core.cpp` asserts the same
properties as `tests/test_admittance.py`, against the same spring-wall model
with the same constants -- so if the two implementations ever disagree, CI says
so rather than the workpiece.

**One difference from the simulation guard, stated because it is otherwise
discovered in a lab:** `griff.control.guard` runs a full iterative IK solve to
convergence each tick; the controller runs a single damped-least-squares step.
That is the right trade in a real-time `update()` -- an unbounded iteration
count has no place in a control loop -- but it means large Cartesian corrections
are tracked over several cycles rather than in one. The force bound is
unaffected: it is enforced on the Cartesian reference, before any of this.

## What is deliberately missing

**A hardware interface.** There is no `SystemInterface` for the Feetech bus.
`griff.teleop.feetech` implements the protocol in Python and its framing is
tested, but it has never been run against a servo and there is no C++ port. The
URDF names `griff_hardware/NotImplemented` in the non-mock branch rather than a
plugin that does not exist, so a `use_mock_hardware:=false` launch fails loudly
instead of looking like it might work.

**`griff_policy` in the ROS CI job.** It imports `griff`, which needs torch and
MuJoCo in the ROS Python environment. Installing a training stack into a ROS CI
job to lint two nodes is not a trade worth making; those nodes are linted by
ruff in the Python job, and their ROS-specific parts are unverified.

**Anything about latency.** The controller is configured at 30 Hz to match the
rate the force estimate arrives at and the rate the policies were trained at.
Whether ACT's ~90 ms CPU inference actually fits in a 33 ms budget on the target
machine is not something this repo can answer -- on the machine it was developed
on, it does not, which is why the evaluation harness runs faster than real time
rather than claiming a real-time result.

## Running it, if you had an arm

```console
$ colcon build --packages-up-to griff_bringup
$ ros2 launch griff_bringup griff.launch.py \
    checkpoint:=/path/to/runs/peg_insert/act-vision_force/policy.pt \
    task:=peg_insert
```

Watch `/force_limited_admittance_controller/status`. `governed` going true
during contact is the guard working; `governed` never going true during a run
that overloaded the workpiece means the guard was not what failed -- the force
estimate was.
