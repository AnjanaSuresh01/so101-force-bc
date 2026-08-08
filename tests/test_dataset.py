"""LeRobot v2.1 dataset writing, reading and validation.

The validator is the thing that makes "conforms to LeRobot v2.1" a claim rather
than an intention, so most of these tests are about it catching damage rather
than about the happy path.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from griff.data.lerobot import (
    BOOKKEEPING_FEATURES,
    CODEBASE_VERSION,
    DatasetWriter,
    LeRobotDataset,
    so101_features,
    validate,
)
from griff.kinematics import JOINT_NAMES

CAMERAS = ("top", "wrist")
SIZE = 8


def write_dataset(root, episodes: int = 2, length: int = 5) -> LeRobotDataset:
    writer = DatasetWriter(
        root, fps=30, features=so101_features(CAMERAS, SIZE, joint_names=list(JOINT_NAMES))
    )
    rng = np.random.default_rng(0)
    for _ in range(episodes):
        writer.start_episode("Insert the peg into the hole")
        for _ in range(length):
            writer.add_frame({
                "observation.state": rng.normal(size=6),
                "observation.force": rng.normal(size=3),
                "action": rng.normal(size=6),
                **{
                    f"observation.images.{camera}": rng.integers(
                        0, 255, size=(SIZE, SIZE, 3), dtype=np.uint8
                    )
                    for camera in CAMERAS
                },
            })
        writer.end_episode({"success": True, "peak_force_true_n": 3.5})
    writer.finalise()
    return LeRobotDataset(root)


def test_round_trip(tmp_path) -> None:
    dataset = write_dataset(tmp_path / "d")
    assert len(dataset) == 2
    assert dataset.fps == 30
    assert dataset.total_frames == 10
    assert set(dataset.image_keys) == {f"observation.images.{c}" for c in CAMERAS}

    frame = dataset.episode_frame(0)
    assert len(frame) == 5
    assert frame["observation.state"][0].shape == (6,)
    assert dataset.load_image("observation.images.top", 0, 0).shape == (SIZE, SIZE, 3)
    assert dataset.extras[0]["success"] is True


def test_written_layout_matches_the_spec(tmp_path) -> None:
    root = tmp_path / "d"
    write_dataset(root)
    for relative in (
        "meta/info.json",
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
        "meta/episodes_stats.jsonl",
        "data/chunk-000/episode_000000.parquet",
        "images/observation.images.top/episode_000000/frame_000000.png",
    ):
        assert (root / relative).exists(), relative

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["codebase_version"] == CODEBASE_VERSION
    assert info["robot_type"] == "so101"
    for name in BOOKKEEPING_FEATURES:
        assert name in info["features"]
    assert info["features"]["observation.state"]["names"] == list(JOINT_NAMES)
    assert info["features"]["observation.images.top"]["dtype"] == "image"


def test_global_index_is_contiguous_across_episodes(tmp_path) -> None:
    dataset = write_dataset(tmp_path / "d", episodes=3, length=4)
    indices = np.concatenate(
        [dataset.episode_frame(i)["index"].to_numpy() for i in range(3)]
    )
    assert np.array_equal(indices, np.arange(12))


def test_validate_accepts_a_good_dataset(tmp_path) -> None:
    root = tmp_path / "d"
    write_dataset(root)
    assert validate(root) == []


def test_validate_catches_a_missing_parquet(tmp_path) -> None:
    root = tmp_path / "d"
    write_dataset(root)
    (root / "data" / "chunk-000" / "episode_000001.parquet").unlink()
    assert any("no parquet" in problem for problem in validate(root))


def test_validate_catches_a_missing_image(tmp_path) -> None:
    root = tmp_path / "d"
    write_dataset(root)
    (root / "images" / "observation.images.wrist" / "episode_000000" / "frame_000002.png").unlink()
    problems = validate(root)
    assert any("have no PNG on disk" in problem for problem in problems)


def test_validate_catches_a_frame_count_disagreement(tmp_path) -> None:
    root = tmp_path / "d"
    write_dataset(root)
    meta = root / "meta" / "episodes.jsonl"
    rows = [json.loads(line) for line in meta.read_text(encoding="utf-8").splitlines()]
    rows[0]["length"] = 99
    meta.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert any("episodes.jsonl says" in problem for problem in validate(root))


def test_validate_catches_a_wrong_codebase_version(tmp_path) -> None:
    root = tmp_path / "d"
    write_dataset(root)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["codebase_version"] = "v1.6"
    info_path.write_text(json.dumps(info), encoding="utf-8")
    assert any("codebase_version" in problem for problem in validate(root))


def test_validate_reports_a_missing_dataset(tmp_path) -> None:
    assert validate(tmp_path / "nothing") == [f"missing {tmp_path / 'nothing/meta/info.json'}"]


def test_writer_rejects_a_wrong_shaped_feature(tmp_path) -> None:
    writer = DatasetWriter(
        tmp_path / "d", fps=30,
        features=so101_features(CAMERAS, SIZE, joint_names=list(JOINT_NAMES)),
    )
    writer.start_episode("task")
    with pytest.raises(ValueError, match="expected shape"):
        writer.add_frame({
            "observation.state": np.zeros(4),
            "observation.force": np.zeros(3),
            "action": np.zeros(6),
            **{f"observation.images.{c}": np.zeros((SIZE, SIZE, 3), np.uint8) for c in CAMERAS},
        })


def test_writer_rejects_a_missing_feature(tmp_path) -> None:
    writer = DatasetWriter(
        tmp_path / "d", fps=30,
        features=so101_features(CAMERAS, SIZE, joint_names=list(JOINT_NAMES)),
    )
    writer.start_episode("task")
    with pytest.raises(KeyError, match="observation.force"):
        writer.add_frame({"observation.state": np.zeros(6), "action": np.zeros(6)})


def test_writer_refuses_to_overwrite_without_being_told(tmp_path) -> None:
    root = tmp_path / "d"
    write_dataset(root)
    with pytest.raises(FileExistsError):
        DatasetWriter(root, fps=30, features=so101_features(CAMERAS, SIZE, joint_names=[]))


def test_writer_refuses_an_empty_dataset(tmp_path) -> None:
    writer = DatasetWriter(
        tmp_path / "d", fps=30, features=so101_features(CAMERAS, SIZE, joint_names=[])
    )
    with pytest.raises(ValueError, match="no episodes"):
        writer.finalise()


def test_writer_refuses_an_empty_episode(tmp_path) -> None:
    writer = DatasetWriter(
        tmp_path / "d", fps=30, features=so101_features(CAMERAS, SIZE, joint_names=[])
    )
    writer.start_episode("task")
    with pytest.raises(ValueError, match="no frames"):
        writer.end_episode()


def test_episode_stats_cover_every_feature(tmp_path) -> None:
    root = tmp_path / "d"
    write_dataset(root)
    rows = [
        json.loads(line)
        for line in (root / "meta" / "episodes_stats.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    stats = rows[0]["stats"]
    for name in ("observation.state", "observation.force", "action", *BOOKKEEPING_FEATURES):
        assert name in stats
        assert set(stats[name]) == {"min", "max", "mean", "std", "count"}
    # Images are stored as [0, 1] per-channel statistics, as LeRobot does.
    image_stats = stats["observation.images.top"]
    assert len(image_stats["mean"]) == 3
    assert 0.0 <= image_stats["mean"][0] <= 1.0
