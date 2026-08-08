"""Turning a LeRobot dataset into (observation, action-chunk) training samples.

Images are decoded once into a uint8 array in RAM rather than read from PNG per
batch. For these datasets that is a few hundred megabytes and it removes the
only real bottleneck in CPU training -- decoding the same few thousand PNGs
several hundred times over the course of a run.

Chunks that run off the end of an episode are padded by repeating the final
action and masked out of the loss. Padding without masking is the quiet version
of this bug: the policy learns that every task ends by holding still, and at
evaluation time it holds still early.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from griff.data.lerobot import LeRobotDataset
from griff.policies.config import PolicyConfig
from griff.policies.encoders import Normaliser


@dataclass
class Normalisers:
    state: Normaliser
    force: Normaliser
    action: Normaliser


class ChunkDataset(Dataset):
    def __init__(self, root: str | Path, config: PolicyConfig) -> None:
        self.config = config
        self.source = LeRobotDataset(root)
        self.image_keys = [f"observation.images.{camera}" for camera in config.cameras]
        missing = [key for key in self.image_keys if key not in self.source.image_keys]
        if missing:
            raise KeyError(
                f"dataset {root} has no {missing}; it holds {self.source.image_keys}"
            )

        states: list[np.ndarray] = []
        forces: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        self.episode_bounds: list[tuple[int, int]] = []

        cursor = 0
        for episode in self.source.episodes:
            index = episode["episode_index"]
            frame = self.source.episode_frame(index)
            states.append(np.stack(frame["observation.state"].to_numpy()))
            forces.append(np.stack(frame["observation.force"].to_numpy()))
            actions.append(np.stack(frame["action"].to_numpy()))
            self.episode_bounds.append((cursor, cursor + len(frame)))
            cursor += len(frame)

        self.states = np.concatenate(states).astype(np.float32)
        self.forces = np.concatenate(forces).astype(np.float32)
        self.actions = np.concatenate(actions).astype(np.float32)
        self.images = {key: self._images_for(key, cursor) for key in self.image_keys}
        self.episode_of = np.zeros(cursor, dtype=np.int64)
        for episode_index, (start, end) in enumerate(self.episode_bounds):
            self.episode_of[start:end] = episode_index

    def _images_for(self, key: str, total: int) -> np.ndarray:
        """Decode a camera's PNGs once, then reuse a single cached array.

        Reading tens of thousands of individually tiny PNGs is the slowest part
        of training by a wide margin -- worse on a cloud-synced folder, where
        each open goes through the sync filter and costs tens of milliseconds.
        One contiguous .npy read costs a fraction of a second for the same
        bytes, so the first run pays the decode and every run after it does not.

        The cache is keyed on the frame count. It is regenerated if the dataset
        grows or shrinks, so a stale cache cannot silently train on old data.
        """
        cache = self.source.root / "cache" / f"{key}.npy"
        if cache.exists():
            # Memory-mapped first so a stale cache can be checked without
            # reading it, then copied: a read-only mmap handed to
            # torch.from_numpy produces a tensor PyTorch warns about and will
            # not guarantee the behaviour of.
            cached = np.load(cache, mmap_mode="r")
            if cached.shape[0] == total:
                return np.array(cached)

        frames: list[np.ndarray] = []
        for episode, (start, end) in zip(self.source.episodes, self.episode_bounds, strict=True):
            index = episode["episode_index"]
            frames.extend(self.source.load_image(key, index, i) for i in range(end - start))
        stacked = np.stack(frames)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, stacked)
        return stacked

    def __len__(self) -> int:
        return len(self.states)

    @property
    def megabytes(self) -> float:
        return sum(a.nbytes for a in self.images.values()) / 1e6

    def normalisers(self) -> Normalisers:
        def make(values: np.ndarray) -> Normaliser:
            return Normaliser(
                torch.from_numpy(values.mean(axis=0)), torch.from_numpy(values.std(axis=0))
            )

        return Normalisers(make(self.states), make(self.forces), make(self.actions))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        start, end = self.episode_bounds[self.episode_of[index]]
        chunk = self.config.chunk
        indices = np.minimum(index + np.arange(chunk), end - 1)
        mask = (index + np.arange(chunk) < end).astype(np.float32)

        images = {
            camera: torch.from_numpy(
                self.images[f"observation.images.{camera}"][index]
            ).permute(2, 0, 1).float()
            / 255.0
            for camera in self.config.cameras
        }
        return {
            "images": images,
            "state": torch.from_numpy(self.states[index]),
            "force": torch.from_numpy(self.forces[index]),
            "actions": torch.from_numpy(self.actions[indices]),
            "mask": torch.from_numpy(mask),
        }


def collate(batch: list[dict]) -> dict:
    cameras = list(batch[0]["images"].keys())
    return {
        "images": {
            camera: torch.stack([item["images"][camera] for item in batch]) for camera in cameras
        },
        "state": torch.stack([item["state"] for item in batch]),
        "force": torch.stack([item["force"] for item in batch]),
        "actions": torch.stack([item["actions"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
    }
