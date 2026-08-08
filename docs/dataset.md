# The datasets

180 teleoperated episodes, 15,391 frames, three tasks, written in the LeRobot
v2.1 layout.

| task | episodes | frames | mean length | contact ticks | peak force mean / max |
|---|---|---|---|---|---|
| peg_insert | 60 | 3,556 | 59 | 35% | 1.9 N / 4.5 N |
| wipe | 60 | 7,673 | 128 | 71% | 4.2 N / 5.6 N |
| press_fit | 60 | 4,162 | 69 | 62% | 8.7 N / 10.3 N |

Recorded by `griff record`, validated by `griff validate-dataset`, and
regenerable from seed: `griff record --episodes 60 --seed 1` reproduces them.

## Layout

```
meta/info.json              feature schema, fps, chunk and path templates
meta/tasks.jsonl            task index -> natural-language task string
meta/episodes.jsonl         episode index -> tasks, length
meta/episodes_stats.jsonl   per-episode per-feature min/max/mean/std
meta/episodes_griff.jsonl   this repo's extras (see below)
data/chunk-000/episode_000000.parquet
images/<key>/episode_000000/frame_000000.png
```

Written against the format rather than against the `lerobot` package. That is a
deliberate trade: importing lerobot would pull a second training stack into a
repo that already has one, and the format is small enough to implement exactly.
The cost is that conformance is a claim rather than a consequence, so
`griff.data.lerobot.validate` exists to check it -- structure, required metadata
keys, per-episode row counts, monotone global indices, and that every frame's
image file is actually on disk. `tests/test_dataset.py` checks that the
validator catches each of those failures rather than only passing the happy path.

## Features

| key | dtype | shape | what it is |
|---|---|---|---|
| `observation.state` | float32 | (6,) | joint positions, rad (gripper in m) |
| `observation.force` | float32 | (3,) | **estimated** contact force at the tool, N, base frame |
| `observation.images.top` | image | (96, 96, 3) | fixed overhead camera |
| `observation.images.wrist` | image | (96, 96, 3) | wrist-mounted camera |
| `action` | float32 | (6,) | joint position targets -- what the leader arm commanded |

`observation.force` is not a standard LeRobot key, because most LeRobot robots
have nothing to put in it. It is an estimate from servo load, not a measurement;
see [force-estimation.md](force-estimation.md).

## Frames are PNGs, not video

LeRobot supports both. Frames are used here because there is no ffmpeg on the
machine that recorded them, and a dataset that cannot be read back without a
system binary is not reproducible in the way this repo needs it to be. The cost
is size and a slow first read, which `griff.policies.dataset.ChunkDataset`
mitigates by decoding once into a cached `.npy` per camera.

## The extras file

`meta/episodes_griff.jsonl` carries what this project needs and the spec has no
field for: whether the episode succeeded, its peak force estimated and true, the
operator's speed and aiming error for that episode, and the full randomisation
draw (hole position, plate tilt, retainer stiffness). It is a separate file on
purpose -- everything a LeRobot loader reads stays exactly as specified, and
anything extra lives where it cannot corrupt the former.

The randomisation draw is what makes the ablation auditable after the fact. If
force conditioning helps, `unresolvable_offset_mm` and `tilt_roll_deg` are the
columns to correlate it against.

## What is in the observation and action pairing

One frame per 30 Hz control tick, holding the observation the operator acted on
and the joint command they produced from it. That pairing is the whole content
of a behaviour-cloning dataset and it is easy to get subtly wrong: recording the
observation *after* stepping pairs each command with the state it caused rather
than the state that caused it, and trains a policy to predict the past.

## Failed episodes are dropped

Demonstration datasets are meant to demonstrate; a policy cloned from a mixture
of successes and timeouts learns the average of the two. The counts are reported
either way, so the yield of the teleoperation session is visible rather than
implied -- one episode was discarded across the 181 attempted.

## Committed vs regenerable

`datasets/demo-<task>` (3 episodes each) is committed, and is what CI trains a
30-step smoke test on. The full 60-episode datasets are not committed -- 1.2 GB
of PNGs -- and are regenerated from seed. `GRIFF_DATASETS` redirects where they
are written; see `griff.paths`.
