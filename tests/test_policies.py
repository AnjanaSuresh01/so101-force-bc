"""ACT, Diffusion Policy, and the ablation switch that has to be airtight."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from griff.policies import (
    CONDITIONINGS,
    POLICY_KINDS,
    ChunkDataset,
    Normaliser,
    PolicyConfig,
    SpatialSoftmax,
    TemporalEnsemble,
    build_policy,
    collate,
    load_policy,
    save_checkpoint,
)
from griff.policies.diffusion import cosine_beta_schedule

BATCH = 4


def make_batch(config: PolicyConfig, seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {
        "images": {
            camera: torch.rand(
                BATCH, 3, config.image_size, config.image_size, generator=generator
            )
            for camera in config.cameras
        },
        "state": torch.randn(BATCH, config.state_dim, generator=generator),
        "force": torch.randn(BATCH, config.force_dim, generator=generator),
        "actions": torch.randn(BATCH, config.chunk, config.action_dim, generator=generator),
        "mask": torch.ones(BATCH, config.chunk),
    }


def make_model(kind: str, conditioning: str):
    config = PolicyConfig(kind=kind, conditioning=conditioning, image_size=32)
    identity6 = Normaliser(torch.zeros(6), torch.ones(6))
    identity3 = Normaliser(torch.zeros(3), torch.ones(3))
    return config, build_policy(config, identity6, identity3, identity6)


@pytest.mark.parametrize("kind", POLICY_KINDS)
@pytest.mark.parametrize("conditioning", CONDITIONINGS)
def test_loss_and_chunk_shapes(kind: str, conditioning: str) -> None:
    config, model = make_model(kind, conditioning)
    batch = make_batch(config)
    loss, parts = model.loss(
        batch["images"], batch["state"], batch["force"], batch["actions"], batch["mask"]
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert parts
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

    chunk = model.predict_chunk(batch["images"], batch["state"], batch["force"])
    assert chunk.shape == (BATCH, config.chunk, config.action_dim)
    assert torch.isfinite(chunk).all()


@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_blinded_and_sighted_have_identical_capacity(kind: str) -> None:
    """The control arm exists to hold parameter count fixed. Check that it does."""
    _, blinded = make_model(kind, "vision_zeroforce")
    _, sighted = make_model(kind, "vision_force")
    assert sum(p.numel() for p in blinded.parameters()) == sum(
        p.numel() for p in sighted.parameters()
    )
    _, plain = make_model(kind, "vision")
    assert sum(p.numel() for p in plain.parameters()) < sum(
        p.numel() for p in sighted.parameters()
    )


@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_blinded_policy_cannot_see_the_force_channel(kind: str) -> None:
    """The whole control depends on this. Feed it two very different forces and
    require byte-identical output."""
    config, model = make_model(kind, "vision_zeroforce")
    model.eval()
    batch = make_batch(config)
    torch.manual_seed(0)
    first = model.predict_chunk(batch["images"], batch["state"], batch["force"])
    torch.manual_seed(0)
    second = model.predict_chunk(
        batch["images"], batch["state"], batch["force"] * 100 + 7.0
    )
    assert torch.equal(first, second)


@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_sighted_policy_does_see_the_force_channel(kind: str) -> None:
    config, model = make_model(kind, "vision_force")
    model.eval()
    batch = make_batch(config)
    torch.manual_seed(0)
    first = model.predict_chunk(batch["images"], batch["state"], batch["force"])
    torch.manual_seed(0)
    second = model.predict_chunk(
        batch["images"], batch["state"], batch["force"] * 100 + 7.0
    )
    assert not torch.allclose(first, second)


def test_vision_only_policy_has_no_force_branch() -> None:
    _, model = make_model("act", "vision")
    assert model.observation_encoder.force_proj is None


def test_mask_excludes_padded_actions_from_the_loss() -> None:
    """Padding without masking teaches every task to end by holding still."""
    config, model = make_model("act", "vision_force")
    batch = make_batch(config)
    masked = dict(batch)
    masked["mask"] = torch.ones(BATCH, config.chunk)
    masked["mask"][:, config.chunk // 2 :] = 0.0

    corrupted = batch["actions"].clone()
    corrupted[:, config.chunk // 2 :] += 1000.0
    torch.manual_seed(0)
    with_padding, _ = model.loss(
        batch["images"], batch["state"], batch["force"], corrupted, masked["mask"]
    )
    torch.manual_seed(0)
    without_padding, _ = model.loss(
        batch["images"], batch["state"], batch["force"], batch["actions"], masked["mask"]
    )
    assert with_padding.item() == pytest.approx(without_padding.item(), rel=1e-3)


def test_spatial_softmax_returns_the_activation_centroid() -> None:
    softmax = SpatialSoftmax(8, 8, temperature=0.01)
    features = torch.full((1, 1, 8, 8), -10.0)
    features[0, 0, 0, 7] = 10.0  # top-right corner
    out = softmax(features)
    assert out.shape == (1, 2)
    assert out[0, 0].item() == pytest.approx(1.0, abs=0.05)  # x -> +1
    assert out[0, 1].item() == pytest.approx(-1.0, abs=0.05)  # y -> -1


def test_normaliser_round_trips_and_survives_a_constant_channel() -> None:
    normaliser = Normaliser(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 0.0]))
    values = torch.tensor([[3.0, 2.0], [-1.0, 2.0]])
    assert torch.allclose(normaliser.denormalise(normaliser.normalise(values)), values)
    assert torch.isfinite(normaliser.normalise(values)).all()


def test_cosine_schedule_is_monotone_and_bounded() -> None:
    betas = cosine_beta_schedule(100)
    assert betas.shape == (100,)
    assert (betas > 0).all() and (betas < 1).all()
    alphas_cumprod = torch.cumprod(1 - betas, dim=0)
    assert alphas_cumprod[0] > 0.99
    assert alphas_cumprod[-1] < 0.05


def test_temporal_ensemble_averages_overlapping_chunks() -> None:
    ensemble = TemporalEnsemble(chunk=4, action_dim=2, weight_decay=0.0)
    first = ensemble.add(np.tile(np.array([1.0, 1.0]), (4, 1)))
    assert np.allclose(first, [1.0, 1.0])
    # With zero decay this is a plain mean of the two chunks' predictions.
    second = ensemble.add(np.tile(np.array([3.0, 3.0]), (4, 1)))
    assert np.allclose(second, [2.0, 2.0])


def test_temporal_ensemble_forgets_chunks_that_have_run_out() -> None:
    ensemble = TemporalEnsemble(chunk=2, action_dim=1)
    for _ in range(5):
        ensemble.add(np.array([[1.0], [1.0]]))
    assert len(ensemble._predictions) <= 2


def test_temporal_ensemble_resets_between_episodes() -> None:
    ensemble = TemporalEnsemble(chunk=3, action_dim=1)
    ensemble.add(np.array([[5.0], [5.0], [5.0]]))
    ensemble.reset()
    out = ensemble.add(np.array([[1.0], [1.0], [1.0]]))
    assert out == pytest.approx([1.0])


@pytest.mark.parametrize("kind", POLICY_KINDS)
def test_checkpoint_round_trips_without_the_dataset(tmp_path, kind: str) -> None:
    """A checkpoint must be self-contained: normalisation travels with it."""
    config = PolicyConfig(kind=kind, conditioning="vision_force", image_size=32)
    state = Normaliser(torch.arange(6, dtype=torch.float), torch.full((6,), 2.0))
    force = Normaliser(torch.zeros(3), torch.ones(3))
    model = build_policy(config, state, force, state)
    save_checkpoint(tmp_path / "policy.pt", model, config)

    runner = load_policy(tmp_path / "policy.pt")
    assert runner.config.kind == kind
    assert torch.allclose(runner.model.state_norm.mean, state.mean)
    assert torch.allclose(runner.model.action_norm.std, state.std)


def test_runner_emits_one_action_per_tick(tmp_path) -> None:
    config = PolicyConfig(kind="diffusion", conditioning="vision_force", image_size=32)
    identity6 = Normaliser(torch.zeros(6), torch.ones(6))
    identity3 = Normaliser(torch.zeros(3), torch.ones(3))
    save_checkpoint(
        tmp_path / "policy.pt", build_policy(config, identity6, identity3, identity6), config
    )
    runner = load_policy(tmp_path / "policy.pt")
    runner.reset()

    class Observation:
        state = np.zeros(6, dtype=np.float32)
        force = np.zeros(3, dtype=np.float32)
        images = {c: np.zeros((32, 32, 3), np.uint8) for c in config.cameras}

    actions = [runner.act(Observation()) for _ in range(config.execute + 2)]
    assert all(a.shape == (6,) for a in actions)
    assert all(np.isfinite(a).all() for a in actions)


def test_chunk_dataset_pads_and_masks_at_the_episode_end(tmp_path) -> None:
    from tests.test_dataset import write_dataset

    write_dataset(tmp_path / "d", episodes=1, length=6)
    config = PolicyConfig(chunk=4, cameras=("top", "wrist"), image_size=8)
    dataset = ChunkDataset(tmp_path / "d", config)
    assert len(dataset) == 6

    last = dataset[5]
    assert last["mask"].tolist() == [1.0, 0.0, 0.0, 0.0]
    # Padded entries repeat the final action rather than being zeros.
    assert torch.allclose(last["actions"][1], last["actions"][0])

    middle = dataset[0]
    assert middle["mask"].tolist() == [1.0, 1.0, 1.0, 1.0]
    batch = collate([dataset[i] for i in range(3)])
    assert batch["actions"].shape == (3, 4, 6)
    assert batch["images"]["top"].shape == (3, 3, 8, 8)


def test_chunk_dataset_rejects_a_missing_camera(tmp_path) -> None:
    from tests.test_dataset import write_dataset

    write_dataset(tmp_path / "d", episodes=1, length=3)
    with pytest.raises(KeyError, match="observation.images.side"):
        ChunkDataset(tmp_path / "d", PolicyConfig(cameras=("side",), image_size=8))


def test_config_round_trips(tmp_path) -> None:
    config = PolicyConfig(kind="diffusion", conditioning="vision_zeroforce", chunk=9)
    config.save(tmp_path / "config.json")
    reloaded = PolicyConfig.load(tmp_path / "config.json")
    assert reloaded.kind == "diffusion"
    assert reloaded.conditioning == "vision_zeroforce"
    assert reloaded.chunk == 9
    assert reloaded.cameras == config.cameras
    assert reloaded.force_is_blinded
