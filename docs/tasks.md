# The three tasks, and what makes each one a *force* task

An ablation between vision-only and vision+force policies measures nothing
unless the tasks contain information that only force carries. Each of these is
randomised along an axis the 96x96 camera views cannot resolve and the force
channel can. That is the experiment, and it is the first thing to check if the
results ever look uninteresting.

Overload thresholds sit below what the arm can produce -- the STS3215's 2.94 N.m
at a 0.19 m reach caps contact around 15 N -- so "succeeded by overloading the
workpiece" is a reachable outcome rather than a hypothetical. `tests/test_tasks.py`
asserts this directly by ramming each fixture and requiring the threshold to be
crossed.

---

## peg_insert

A 17 mm peg into a 20 mm square bore: **1.5 mm radial clearance**, 36 mm deep,
with a coarse entry funnel.

**The hidden variable.** The fixture is re-placed every episode, and the offset
is drawn in two parts: a coarse ±12 mm, which is about four pixels at the policy
input and is therefore visible, and a fine ±3 mm, which is about one pixel and
is not. Vision gets the peg to the lip. What tells you it is jammed on the
chamfer rather than sliding down the bore is the contact force -- and at the
moment it matters, the wrist camera is looking at the top face of the fixture
with the peg hidden inside it.

**Success.** Peg tip 26 mm below the bore top and laterally within 6 mm.
**Overload.** 8 N. **Controller limit.** 6 N.

The demonstrations contain the force-guided behaviour this is meant to teach:
the scripted operator carries a ±3.2 mm aiming error it cannot see, lands on the
lip, and spirals outward around *where it believes the hole is* while pressing
lightly, until the peg drops. An earlier version spiralled around the true hole
centre instead, which silently cancelled the aim error and produced 12/12
successes at 0.3 N peak force -- demonstrations that never touched anything.

---

## wipe

Sweep a compliant pad across a plate, covering the full stroke while holding the
normal force in a band.

**The hidden variable.** The plate is **tilted ±4 degrees** about both horizontal
axes every episode. Across the 80 mm stroke that is 6 mm of height change --
enough to lift the pad clear at one end or bury it at the other. From an
overhead camera at 96x96, a 4 degree tilt is a couple of pixels of
foreshortening. A policy that only sees pixels has to guess the plane; a policy
that feels the normal force can servo on it.

**Success.** All five stroke checkpoints visited while pressed at 1.5 N or more.
**Overload.** 10 N. **Controller limit.** 7 N.

The stroke runs laterally rather than radially, which is a reachability decision
and not an aesthetic one: sweeping 80 mm in and out with the tool held vertical
puts the elbow against its 1.69 rad limit at the near end, the IK stops
converging, and the pad never touches the plate at all. A lateral stroke is a
pure `shoulder_pan` rotation at constant radius, so every point on it is equally
reachable.

The pad is deliberately soft (`solref 0.02`) -- a sponge, not a rigid puck. Soft
contact turns the force channel into a usable proxy for penetration depth rather
than an on/off spike, which is what makes force conditioning learnable at 30 Hz.

---

## press_fit

Press a part into a socket against a spring-loaded retainer with 14 mm of travel
and then a rigid stop.

**The hidden variable.** Retainer stiffness, randomised ±20%. It is inside the
socket. No camera at any resolution can see it, which makes this the cleanest of
the three ablations: the force required to seat the part is, by construction,
observable only through the force channel.

**The window.** Below the seating force (6-9 N depending on the draw) the part
never seats and the run fails on the task criterion. Above the overload
threshold the retainer has bottomed out and the load is going straight into the
workpiece, so the run fails on the force criterion *even though the part is
seated*. This is the task where the two success definitions come apart most
sharply.

**Success.** 11 mm of retainer travel, held for 6 ticks.
**Overload.** 14 N. **Controller limit.** 12 N.

---

## Did the design work?

Partly. Each task *does* contain an axis the cameras cannot resolve — that is a
property of the fixtures, and it is verifiable by reading the randomisation draw
in `meta/episodes_griff.jsonl`. What the results do not show is policies
exploiting it: measured against the capacity-matched control arm, the force
signal is worth 0 pp of force-aware success on average across the six cells
(see the README's ablation table).

Two readings, and this repo cannot separate them:

* the tasks are solvable from vision and proprioception alone at these
  tolerances, so the force channel is redundant; or
* 3,000 CPU steps on 60 demonstrations is not enough for a 1.08 M-parameter
  policy to learn to use a noisy 3-vector, and a longer run would find it.

If you want to push on this, the cheapest discriminating experiment is
press_fit: its hidden variable (retainer stiffness) is the one that is
*provably* unobservable from pixels, so a policy that never beats the blinded
control there is not being denied information by the camera.

## What is simplified, and why

**Tools are welded, not grasped.** Peg, pad and part are held to the tool by an
equality constraint rather than by friction between the jaws, and the jaw geoms
do not collide. Grasping is not what is being studied, and a friction grasp
would spend every episode slipping -- turning the force signal into an artefact
of the grip rather than of the task. Leaving jaw collision on produced exactly
that: the jaw actuator squeezing the peg against a compliant weld showed up at
the joints as several newtons of contact that no fixture explained.

**The arm is a kinematically faithful stand-in, not a vendor model.** Joint set,
link lengths, joint limits and the STS3215's torque limit are the SO-101's;
collision geometry is primitives. `tests/test_urdf_matches_mjcf.py` keeps the
MuJoCo model and the URDF describing the same robot, because the policies are
trained against one and the ROS controller computes kinematics from the other.

**The demonstrations come from a scripted operator, not a human.** There is no
SO-101 on the machine this was built on. The operator is a state machine with a
per-episode aiming error, a randomised speed, and reference noise, regulating on
the same force estimate a human would read off a screen -- and rate-limited to
3.6 cm/s, because commanding the tool faster than the servos can follow leaves a
tracking error the force estimator reports as contact.
