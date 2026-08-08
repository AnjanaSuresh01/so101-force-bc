# Where the force numbers come from

The SO-101 has no force/torque sensor. Every quantity in this repository that
calls itself a force is an **estimate** derived from what six STS3215 servos
report over a serial bus, and the difference matters enough to spell out.

## The signal

`Present_Load`, control-table address `0x3C`. Eleven bits: a magnitude of 0-1023
in bits 0-9, and direction in bit 10. As a fraction of the servo's 2.94 N.m
stall torque that is a resolution of about **3 mN.m**, and that is the noise
floor of everything downstream. Reading the field as a plain unsigned 16-bit
value -- which is the natural mistake, because every other field on this bus is
one -- makes a joint pushing one way report a load 1024 counts higher than the
same effort the other way. `griff.teleop.feetech.decode_load` handles it and
`tests/test_feetech.py` pins the behaviour.

## The estimator

Textbook joint-torque residual:

```
tau_ext = tau_measured - tau_model(q, qdot, qddot)
F       = argmin_F || J(q)^T F - tau_ext ||^2 + lambda ||F||^2
```

`tau_model` is everything that loads the servos when nothing is touching the
tool. It is **fitted, not derived** -- on hardware you do not know the link
inertias, and printed gearboxes have friction no datasheet lists. The regressor
is per joint:

| term | why it is there |
|---|---|
| `bias` | mounting offset, servo zero |
| `sin/cos` of the cumulative pitch angles | gravity. The three pitch joints share an axis, so the gravity torque on each depends only on `q1`, `q1+q2`, `q1+q2+q3` |
| `coulomb` = `tanh(qdot / 0.02)` | dry friction. `tanh` rather than `sign`: a hard sign flips the prediction discontinuously at zero velocity, and a servo dithering around a held position then injects phantom contact several times a second |
| `viscous` = `qdot` | velocity-proportional drag |
| `accel` = `qddot` | diagonal inertia |

The `accel` term was not in the first version and its absence was a real bug, of
the kind that looks like a policy failure rather than a sensing failure. Without
it, the torque needed to accelerate a link is unexplained and the estimator
reports it as external force: a 3 cm/s tool move leaves an 80 mrad tracking
error at the shoulder, which reads as **3-6 N of contact that nothing is
touching**. The scripted operator, regulating on that reading, would lift away
from a workpiece it had never met. `qddot` is obtainable on hardware from
successive position reads, which is why it is allowed to be in the model.

## Calibration

`griff calibrate` does on the simulated arm what you would do on the bench:

1. Clear the bench, so nothing can touch the tool. In simulation that is zeroing
   every geom's collision mask; the weld holding the tool is an equality
   constraint and is unaffected, so the payload stays in the jaws.
2. Sweep the working volume with the task's tool still held -- **calibration is
   per task payload, not per robot**, because the peg, the pad and the part have
   different masses.
3. Fit by least squares.

The sweep has to excite all three effects or the fit cannot separate them:
dwells for gravity, creeps at several speeds for friction, and darts for
inertia. Quasi-static data alone leaves the `accel` coefficient unidentifiable.

The fit residual lands at **13-17 mN.m per joint**, which is the servo noise
floor plus quantisation -- the model is fitting to the limit of what the bus can
report, and there is nothing left in the free-space signal to explain.

## Measured accuracy

`griff validate-sensing` runs contact-rich motion with the workpiece back in
place and compares the estimate against the simulator's ground-truth F/T sensor
-- a sensor no policy and no controller in this repo can read.

| task | in-range RMSE | contact RMSE | correlation | free space | peak true / est |
|---|---|---|---|---|---|
| peg_insert | 0.96 N | 1.49 N | 0.994 | 1.09 N | 36.1 / 37.6 N |
| wipe | 3.12 N | 1.75 N | 0.972 | 0.07 N | 23.4 / 22.6 N |
| press_fit | 2.02 N | 1.60 N | 0.993 | 0.07 N | 36.6 / 37.4 N |

*In-range* is the error over contacts at or below the task's overload threshold
-- the band the success criterion is decided in. *Free space* is what the
estimate reads with nothing touching the tool.

## What it cannot do

**It cannot see forces the Jacobian cannot transmit.** A force along a direction
the arm is singular in produces no joint torque and is invisible. `estimate`
returns the condition number of `J J^T` so this is measurable rather than a
footnote: a small reading at a large condition number means the estimate is
blind, not that nothing is being touched.

**It saturates when the servos do.** Above roughly 30 N of contact the STS3215s
are at their 2.94 N.m limit, measured torque stops rising, and so does the
estimate. Every force this repo *acts* on is well below that; the ceiling is
visible in the peak columns above and is a property of the arm, not of the
estimator.

**It lags.** The output is low-passed (`smoothing = 0.4`, roughly a two-tick
time constant at 30 Hz) because the raw residual is noisy at the bus's
resolution. Fast force transients are reported late and smaller than they were.

## Why the metrics are not graded with it

`results/RESULTS.md` reports peak contact force from the simulator's
ground-truth sensor, not from this estimate. Grading a safety property with the
same signal the system used to try to satisfy it would let estimator error hide
exactly the failures being looked for: a policy that overloads a workpiece while
its own force channel under-reads would score as safe. The estimate drives the
controller and conditions the policy; the ground truth decides whether the run
counted.

That is also why every task's overload threshold sits above its controller
limit. The controller bounds the *estimated* force; the margin between the two
is the estimator's error, and the table above is what sizes it.
