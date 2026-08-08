"""LeRobot v2.1 dataset writer, reader and validator.

Written against the format rather than against the `lerobot` package. That is a
deliberate trade: importing lerobot would pull a second training stack into a
repo that already has one, and the format is small enough to implement exactly.
The cost is that conformance is a claim rather than a consequence, so
`validate` exists to check it -- structure, required metadata keys, per-episode
row counts, monotone global indices, and that every frame's image file is
actually on disk.

Layout produced:

    meta/info.json              feature schema, fps, chunk and path templates
    meta/tasks.jsonl            task index -> natural-language task string
    meta/episodes.jsonl         episode index -> tasks, length
    meta/episodes_stats.jsonl   per-episode per-feature min/max/mean/std
    meta/episodes_griff.jsonl   this repo's extras: outcome, peak force, the
                                per-episode randomisation draw
    data/chunk-000/episode_000000.parquet
    images/<key>/episode_000000/frame_000000.png

Images are PNG frames rather than encoded video. LeRobot supports both; frames
are chosen here because there is no ffmpeg on the machine this was recorded on,
and a dataset that cannot be read back without a system binary is not
reproducible in the way this repo needs it to be.

`meta/episodes_griff.jsonl` is kept separate on purpose. Everything a LeRobot
loader reads stays exactly as specified; anything this project needs that the
spec has no field for lives in its own file where it cannot corrupt the former.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

CODEBASE_VERSION = "v2.1"
CHUNK_SIZE = 1000

#: Columns every LeRobot dataset carries, whatever the robot.
BOOKKEEPING_FEATURES: dict[str, dict[str, Any]] = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}


@dataclass(frozen=True)
class Feature:
    name: str
    dtype: str  # "float32" | "int64" | "image"
    shape: tuple[int, ...]
    names: list[str] | None = None

    @property
    def is_image(self) -> bool:
        return self.dtype == "image"

    def to_info(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "shape": list(self.shape), "names": self.names}


def so101_features(
    cameras: tuple[str, ...], image_size: int, *, joint_names: list[str]
) -> list[Feature]:
    """The feature set these recordings use."""
    features = [
        Feature("observation.state", "float32", (6,), joint_names),
        # Not a standard LeRobot key. There is no standard key for this, because
        # most LeRobot robots have nothing to put in it.
        Feature("observation.force", "float32", (3,), ["fx", "fy", "fz"]),
        Feature("action", "float32", (6,), joint_names),
    ]
    features += [
        Feature(
            f"observation.images.{camera}",
            "image",
            (image_size, image_size, 3),
            ["height", "width", "channel"],
        )
        for camera in cameras
    ]
    return features


def _stats(values: np.ndarray) -> dict[str, list[float]]:
    flat = values.reshape(len(values), -1).astype(np.float64)
    return {
        "min": flat.min(axis=0).tolist(),
        "max": flat.max(axis=0).tolist(),
        "mean": flat.mean(axis=0).tolist(),
        "std": flat.std(axis=0).tolist(),
        "count": [int(len(flat))],
    }


def _image_stats(frames: list[np.ndarray]) -> dict[str, list[float]]:
    """Per-channel statistics on [0, 1] pixels, as LeRobot stores them.

    Subsampled: the mean of every tenth frame of an episode is the same number
    to well past the precision anyone normalises with, and reading every PNG
    back to compute it is the slowest part of recording.
    """
    sampled = np.stack(frames[::10] if len(frames) > 10 else frames).astype(np.float64) / 255.0
    per_channel = sampled.reshape(-1, sampled.shape[-1])
    return {
        "min": per_channel.min(axis=0).tolist(),
        "max": per_channel.max(axis=0).tolist(),
        "mean": per_channel.mean(axis=0).tolist(),
        "std": per_channel.std(axis=0).tolist(),
        "count": [int(len(sampled))],
    }


@dataclass
class _EpisodeBuffer:
    index: int
    task: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    images: dict[str, list[np.ndarray]] = field(default_factory=dict)


class DatasetWriter:
    """Accumulates episodes and writes a LeRobot v2.1 dataset."""

    def __init__(
        self,
        root: str | Path,
        *,
        fps: int,
        features: list[Feature],
        robot_type: str = "so101",
        overwrite: bool = False,
    ) -> None:
        self.root = Path(root)
        if self.root.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{self.root} already exists; pass overwrite=True to replace it"
                )
            import shutil

            shutil.rmtree(self.root)
        self.fps = fps
        self.features = features
        self.robot_type = robot_type

        self._tasks: dict[str, int] = {}
        self._episodes: list[dict[str, Any]] = []
        self._episode_stats: list[dict[str, Any]] = []
        self._extras: list[dict[str, Any]] = []
        self._global_index = 0
        self._buffer: _EpisodeBuffer | None = None

    # ------------------------------------------------------------------ writing

    def start_episode(self, task: str) -> None:
        if self._buffer is not None:
            raise RuntimeError("an episode is already open; call end_episode first")
        if task not in self._tasks:
            self._tasks[task] = len(self._tasks)
        self._buffer = _EpisodeBuffer(index=len(self._episodes), task=task)

    def add_frame(self, values: dict[str, Any]) -> None:
        if self._buffer is None:
            raise RuntimeError("no episode open; call start_episode first")
        buffer = self._buffer
        frame_index = len(buffer.rows)
        row: dict[str, Any] = {
            "timestamp": np.float32(frame_index / self.fps),
            "frame_index": np.int64(frame_index),
            "episode_index": np.int64(buffer.index),
            "index": np.int64(self._global_index),
            "task_index": np.int64(self._tasks[buffer.task]),
        }
        for feature in self.features:
            if feature.name not in values:
                raise KeyError(f"frame is missing feature {feature.name!r}")
            value = values[feature.name]
            if feature.is_image:
                array = np.asarray(value, dtype=np.uint8)
                if array.shape != feature.shape:
                    raise ValueError(
                        f"{feature.name}: expected shape {feature.shape}, got {array.shape}"
                    )
                buffer.images.setdefault(feature.name, []).append(array)
            else:
                vector = np.asarray(value, dtype=np.float32).reshape(-1)
                if vector.shape != feature.shape:
                    raise ValueError(
                        f"{feature.name}: expected shape {feature.shape}, got {vector.shape}"
                    )
                row[feature.name] = vector
        buffer.rows.append(row)
        self._global_index += 1

    def end_episode(self, extras: dict[str, Any] | None = None) -> None:
        if self._buffer is None:
            raise RuntimeError("no episode open")
        buffer = self._buffer
        if not buffer.rows:
            raise ValueError(f"episode {buffer.index} has no frames")

        chunk = buffer.index // CHUNK_SIZE
        data_dir = self.root / "data" / f"chunk-{chunk:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(buffer.rows)
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False),
                       data_dir / f"episode_{buffer.index:06d}.parquet")

        stats: dict[str, Any] = {}
        for feature in self.features:
            if feature.is_image:
                frames = buffer.images[feature.name]
                episode_dir = self.root / "images" / feature.name / f"episode_{buffer.index:06d}"
                episode_dir.mkdir(parents=True, exist_ok=True)
                for i, image in enumerate(frames):
                    Image.fromarray(image).save(episode_dir / f"frame_{i:06d}.png")
                stats[feature.name] = _image_stats(frames)
            else:
                stats[feature.name] = _stats(np.stack([r[feature.name] for r in buffer.rows]))
        for name in BOOKKEEPING_FEATURES:
            stats[name] = _stats(np.array([[r[name]] for r in buffer.rows], dtype=np.float64))

        self._episodes.append(
            {"episode_index": buffer.index, "tasks": [buffer.task], "length": len(buffer.rows)}
        )
        self._episode_stats.append({"episode_index": buffer.index, "stats": stats})
        self._extras.append({"episode_index": buffer.index, **(extras or {})})
        self._buffer = None

    def finalise(self) -> Path:
        if self._buffer is not None:
            raise RuntimeError("an episode is still open")
        if not self._episodes:
            raise ValueError("refusing to write a dataset with no episodes")

        meta = self.root / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        total_frames = sum(e["length"] for e in self._episodes)
        features = {f.name: f.to_info() for f in self.features}
        features.update(BOOKKEEPING_FEATURES)

        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "total_episodes": len(self._episodes),
            "total_frames": total_frames,
            "total_tasks": len(self._tasks),
            "total_videos": 0,
            "total_chunks": (len(self._episodes) - 1) // CHUNK_SIZE + 1,
            "chunks_size": CHUNK_SIZE,
            "fps": self.fps,
            "splits": {"train": f"0:{len(self._episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": None,
            "image_path": "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png",
            "features": features,
        }
        _write_json(meta / "info.json", info)
        _write_jsonl(
            meta / "tasks.jsonl",
            [{"task_index": i, "task": t} for t, i in sorted(self._tasks.items(), key=lambda kv: kv[1])],
        )
        _write_jsonl(meta / "episodes.jsonl", self._episodes)
        _write_jsonl(meta / "episodes_stats.jsonl", self._episode_stats)
        _write_jsonl(meta / "episodes_griff.jsonl", self._extras)
        return self.root


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, default=_json_default) + "\n" for row in rows), encoding="utf-8"
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class LeRobotDataset:
    """Read side. Loads metadata eagerly, frames on demand."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.info = json.loads((self.root / "meta" / "info.json").read_text(encoding="utf-8-sig"))
        self.episodes = _read_jsonl(self.root / "meta" / "episodes.jsonl")
        self.tasks = _read_jsonl(self.root / "meta" / "tasks.jsonl")
        extras = self.root / "meta" / "episodes_griff.jsonl"
        self.extras = _read_jsonl(extras) if extras.exists() else []

    @property
    def fps(self) -> int:
        return int(self.info["fps"])

    @property
    def total_frames(self) -> int:
        return int(self.info["total_frames"])

    @property
    def image_keys(self) -> list[str]:
        return [k for k, v in self.info["features"].items() if v["dtype"] == "image"]

    def episode_frame(self, episode: int) -> pd.DataFrame:
        chunk = episode // int(self.info["chunks_size"])
        path = self.root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet"
        return pq.read_table(path).to_pandas()

    def image_path(self, key: str, episode: int, frame: int) -> Path:
        return self.root / "images" / key / f"episode_{episode:06d}" / f"frame_{frame:06d}.png"

    def load_image(self, key: str, episode: int, frame: int) -> np.ndarray:
        with Image.open(self.image_path(key, episode, frame)) as handle:
            return np.asarray(handle.convert("RGB"), dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.episodes)


def validate(root: str | Path, *, check_images: bool = True) -> list[str]:
    """Structural conformance check. Returns a list of problems; empty is a pass."""
    root = Path(root)
    problems: list[str] = []

    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return [f"missing {info_path}"]
    info = json.loads(info_path.read_text(encoding="utf-8-sig"))

    for key in ("codebase_version", "fps", "features", "total_episodes", "total_frames", "data_path"):
        if key not in info:
            problems.append(f"info.json is missing required key {key!r}")
    if info.get("codebase_version") != CODEBASE_VERSION:
        problems.append(
            f"codebase_version is {info.get('codebase_version')!r}, expected {CODEBASE_VERSION!r}"
        )
    for name in BOOKKEEPING_FEATURES:
        if name not in info.get("features", {}):
            problems.append(f"info.json features is missing bookkeeping column {name!r}")

    for name in ("tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"):
        if not (root / "meta" / name).exists():
            problems.append(f"missing meta/{name}")
    if problems:
        return problems

    dataset = LeRobotDataset(root)
    if len(dataset.episodes) != info["total_episodes"]:
        problems.append(
            f"episodes.jsonl has {len(dataset.episodes)} rows but info.json says "
            f"{info['total_episodes']}"
        )

    stats = {row["episode_index"] for row in _read_jsonl(root / "meta" / "episodes_stats.jsonl")}
    known_tasks = {row["task"] for row in dataset.tasks}
    expected_index = 0
    counted_frames = 0

    for episode in dataset.episodes:
        index = episode["episode_index"]
        if index not in stats:
            problems.append(f"episode {index} has no row in episodes_stats.jsonl")
        for task in episode["tasks"]:
            if task not in known_tasks:
                problems.append(f"episode {index} references task {task!r} not in tasks.jsonl")
        try:
            frame = dataset.episode_frame(index)
        except FileNotFoundError:
            problems.append(f"episode {index} has no parquet file")
            continue

        if len(frame) != episode["length"]:
            problems.append(
                f"episode {index}: parquet has {len(frame)} rows, episodes.jsonl says "
                f"{episode['length']}"
            )
        if not (frame["frame_index"].to_numpy() == np.arange(len(frame))).all():
            problems.append(f"episode {index}: frame_index is not 0..N-1")
        if not (frame["index"].to_numpy() == expected_index + np.arange(len(frame))).all():
            problems.append(f"episode {index}: global index is not contiguous with the previous episode")
        expected_index += len(frame)
        counted_frames += len(frame)

        if check_images:
            for key in dataset.image_keys:
                missing = [
                    i for i in range(len(frame)) if not dataset.image_path(key, index, i).exists()
                ]
                if missing:
                    problems.append(
                        f"episode {index}: {len(missing)} frames of {key} have no PNG on disk "
                        f"(first missing: {missing[0]})"
                    )

    if counted_frames != info["total_frames"]:
        problems.append(
            f"parquet files hold {counted_frames} frames but info.json says {info['total_frames']}"
        )
    return problems
