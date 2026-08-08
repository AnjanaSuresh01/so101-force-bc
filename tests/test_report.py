"""The results renderer.

results/RESULTS.md is generated and CI fails on a diff, so the renderer is on
the critical path for every published number. These tests check that the
arithmetic shown in the tables is the arithmetic the harness computed, and that
the force ablation columns are subtracted in the direction their headings claim.
"""

from __future__ import annotations

import json

import pytest

from griff.evaluate import PolicyEvaluation, wilson_interval
from griff.report import render, write_report


def evaluation(policy: str, conditioning: str, *, task_rate: float, safe_rate: float,
               guarded: bool = True, peak: float = 4.0) -> dict:
    return PolicyEvaluation(
        task="peg_insert",
        policy=policy,
        conditioning=conditioning,
        guarded=guarded,
        episodes=20,
        task_success_rate=task_rate,
        task_success_ci=wilson_interval(round(task_rate * 20), 20),
        force_aware_success_rate=safe_rate,
        force_aware_success_ci=wilson_interval(round(safe_rate * 20), 20),
        overload_rate=task_rate - safe_rate,
        overload_share_of_successes=(task_rate - safe_rate) / task_rate if task_rate else 0.0,
        peak_force_mean_n=peak,
        peak_force_p95_n=peak + 1.0,
        peak_force_max_n=peak + 2.0,
        overload_threshold_n=8.0,
        mean_steps=90.0,
        governed_fraction=0.1,
        ik_failure_rate=0.0,
        seconds=1.0,
    ).to_dict()


@pytest.fixture
def payload() -> dict:
    return {
        "generated_by": "griff evaluate",
        "seed_base": 500000,
        "tasks": {
            "peg_insert": [
                evaluation("operator", "scripted", task_rate=1.0, safe_rate=1.0),
                evaluation("act", "vision", task_rate=0.50, safe_rate=0.40),
                evaluation("act", "vision_zeroforce", task_rate=0.55, safe_rate=0.45),
                evaluation("act", "vision_force", task_rate=0.80, safe_rate=0.75),
                evaluation("act", "vision_force", task_rate=0.90, safe_rate=0.60,
                           guarded=False, peak=9.0),
            ]
        },
    }


def test_render_includes_both_success_definitions(payload) -> None:
    text = render(payload)
    assert "task success" in text
    assert "force-aware success" in text
    assert "act / vision + force" in text
    assert "act / vision + blinded force" in text


def test_force_ablation_subtracts_in_the_stated_direction(payload) -> None:
    """information = force - blinded; capacity = blinded - vision."""
    text = render(payload)
    assert "What the force channel contributes" in text
    # 75% - 45% = +30 pp of information; 45% - 40% = +5 pp of capacity.
    assert "+30 pp" in text
    assert "+5 pp" in text


def test_guard_comparison_appears_when_unguarded_runs_exist(payload) -> None:
    text = render(payload)
    assert "With and without the admittance guard" in text
    assert "11.0" in text  # unguarded peak max = 9.0 + 2.0


def test_guard_comparison_is_omitted_when_there_is_nothing_to_compare(payload) -> None:
    payload["tasks"]["peg_insert"] = [
        e for e in payload["tasks"]["peg_insert"] if e["guarded"]
    ]
    assert "With and without the admittance guard" not in render(payload)


def test_render_states_that_peak_force_is_ground_truth(payload) -> None:
    """The reader has to know the safety metric is not graded by the estimate."""
    text = render(payload).lower()
    assert "ground-truth" in text
    assert "estimate" in text


def test_thresholds_table_lists_every_task(payload) -> None:
    from griff.sim.tasks import SPECS

    text = render(payload)
    for task, spec in SPECS.items():
        assert task in text
        assert f"{spec.overload_force:.0f} N" in text


def test_write_report_round_trips(tmp_path, payload) -> None:
    results = tmp_path / "results.json"
    results.write_text(json.dumps(payload), encoding="utf-8")
    out = write_report(results, tmp_path / "RESULTS.md")
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("# Results")


def test_report_is_deterministic(tmp_path, payload) -> None:
    """CI diffs this file. A renderer with unstable ordering would fail forever."""
    assert render(payload) == render(payload)


@pytest.mark.parametrize(
    ("successes", "total"), [(0, 20), (20, 20), (10, 20), (1, 3), (0, 0)]
)
def test_wilson_interval_stays_inside_the_unit_range(successes: int, total: int) -> None:
    low, high = wilson_interval(successes, total)
    assert 0.0 <= low <= high <= 1.0
    if total:
        assert low <= successes / total <= high


def test_wilson_interval_narrows_with_more_episodes() -> None:
    narrow = wilson_interval(80, 100)
    wide = wilson_interval(8, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])
