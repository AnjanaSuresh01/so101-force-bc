"""The evaluation harness, and the scoring rule this repo exists to argue for.

Two success rates are reported for every policy:

    task success        did the peg go in / the plate get wiped / the part seat
    force-aware success did that, AND without ever exceeding the force at which
                        the workpiece is considered damaged

The second is the one that means anything. On contact-rich tasks the cheapest
way to raise the first is to press harder -- a policy that drives a peg through
a misaligned bore by overloading it scores a success under the usual metric and
would have destroyed the part on real hardware. Counting those as failures is
not a stricter variant of the same measurement; it can reorder the ranking, and
the results in results/RESULTS.md show it doing so.

Peak force is measured from the simulator's ground-truth F/T sensor, not from
the servo-load estimate the policy and controller run on. Grading a safety
property with the same noisy signal the system used to try to satisfy it would
let estimator error hide exactly the failures being looked for.

Every policy sees the same episode seeds, so comparisons are paired.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from griff.calibrate import load_or_fit
from griff.control.guard import ForceGuard
from griff.paths import RESULTS, RUNS
from griff.policies.config import CONDITIONINGS, POLICY_KINDS
from griff.policies.runner import load_policy
from griff.sim import make_env
from griff.sim.env import Observation, TaskEnv
from griff.sim.tasks import SPECS
from griff.teleop import make_operator

#: Episode seeds are drawn from here so every policy meets the same fixtures.
EVAL_SEED_BASE = 500_000


class Actor(Protocol):
    def reset(self) -> None: ...
    def act(self, observation: Observation) -> np.ndarray: ...


class OperatorActor:
    """The scripted teleoperator, evaluated as a policy would be.

    Not a baseline in the learning sense -- it uses privileged knowledge of the
    fixture pose that no policy has. It is the reference the demonstrations came
    from, and its numbers are the ceiling any cloned policy is working toward.
    """

    def __init__(self, task: str, env: TaskEnv, seed: int = 0) -> None:
        self.env = env
        self.operator = make_operator(task, np.random.default_rng(seed))
        self._seed = seed

    def reset(self) -> None:
        self.operator.reset(self.env, np.random.default_rng(self._seed))

    def act(self, observation: Observation) -> np.ndarray:
        return self.operator.act(self.env, observation.force)


@dataclass
class RolloutResult:
    seed: int
    task_success: bool
    peak_force_n: float
    peak_force_estimated_n: float
    overloaded: bool
    steps: int
    governed_fraction: float
    ik_failures: int
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def force_aware_success(self) -> bool:
        return self.task_success and not self.overloaded


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval.

    Used instead of the normal approximation because these are 20-30 episode
    samples with rates near 0 and 1, where the normal interval runs off the end
    of [0, 1] and reports a confidence bound that cannot happen.
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass
class PolicyEvaluation:
    task: str
    policy: str
    conditioning: str
    guarded: bool
    episodes: int
    task_success_rate: float
    task_success_ci: tuple[float, float]
    force_aware_success_rate: float
    force_aware_success_ci: tuple[float, float]
    overload_rate: float
    #: Of the runs that completed the task, the share that did it by overloading.
    overload_share_of_successes: float
    peak_force_mean_n: float
    peak_force_p95_n: float
    peak_force_max_n: float
    overload_threshold_n: float
    mean_steps: float
    governed_fraction: float
    ik_failure_rate: float
    seconds: float
    rollouts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _summarise(
    task: str,
    policy: str,
    conditioning: str,
    guarded: bool,
    rollouts: list[RolloutResult],
    seconds: float,
) -> PolicyEvaluation:
    spec = SPECS[task]
    total = len(rollouts)
    task_successes = sum(r.task_success for r in rollouts)
    safe_successes = sum(r.force_aware_success for r in rollouts)
    overloads = sum(r.overloaded for r in rollouts)
    peaks = np.array([r.peak_force_n for r in rollouts])
    return PolicyEvaluation(
        task=task,
        policy=policy,
        conditioning=conditioning,
        guarded=guarded,
        episodes=total,
        task_success_rate=task_successes / total,
        task_success_ci=wilson_interval(task_successes, total),
        force_aware_success_rate=safe_successes / total,
        force_aware_success_ci=wilson_interval(safe_successes, total),
        overload_rate=overloads / total,
        overload_share_of_successes=(
            (task_successes - safe_successes) / task_successes if task_successes else 0.0
        ),
        peak_force_mean_n=float(peaks.mean()),
        peak_force_p95_n=float(np.percentile(peaks, 95)),
        peak_force_max_n=float(peaks.max()),
        overload_threshold_n=spec.overload_force,
        mean_steps=float(np.mean([r.steps for r in rollouts])),
        governed_fraction=float(np.mean([r.governed_fraction for r in rollouts])),
        ik_failure_rate=float(np.mean([r.ik_failures for r in rollouts])),
        seconds=seconds,
        rollouts=[asdict(r) for r in rollouts],
    )


def rollout(
    env: TaskEnv, actor: Actor, seed: int, *, guard: ForceGuard | None = None
) -> RolloutResult:
    observation = env.reset(seed=seed)
    actor.reset()
    if guard is not None:
        guard.reset()

    success = False
    while True:
        action = actor.act(observation)
        if guard is not None:
            action = guard.apply(action, observation.force)
        result = env.step(action)
        observation = result.observation
        if result.done:
            success = result.success
            break

    spec = env.spec
    return RolloutResult(
        seed=seed,
        task_success=bool(success),
        peak_force_n=float(env.peak_force_true),
        peak_force_estimated_n=float(env.peak_force),
        overloaded=bool(env.peak_force_true > spec.overload_force),
        steps=int(env.step_index),
        governed_fraction=guard.stats.governed_fraction if guard else 0.0,
        ik_failures=guard.stats.ik_failures if guard else 0,
        metrics={k: float(v) for k, v in result.info.items() if k in spec.metrics},
    )


def evaluate_actor(
    task: str,
    actor_name: str,
    conditioning: str,
    build: Any,
    *,
    episodes: int = 25,
    guarded: bool = True,
    seed_base: int = EVAL_SEED_BASE,
    env: TaskEnv | None = None,
    progress: bool = True,
) -> PolicyEvaluation:
    started = time.time()
    owned = env is None
    if env is None:
        env = make_env(task, calibration=load_or_fit(task), seed=seed_base)
    actor = build(env)
    guard = ForceGuard(env) if guarded else None

    rollouts = []
    for index in range(episodes):
        rollouts.append(rollout(env, actor, seed_base + index, guard=guard))
        if progress and (index + 1) % 5 == 0:
            done = sum(r.force_aware_success for r in rollouts)
            print(
                f"    {task}/{actor_name}-{conditioning}"
                f"{'' if guarded else ' (unguarded)'}: "
                f"{index + 1}/{episodes} force-aware {done}/{index + 1}",
                flush=True,
            )
    if owned:
        env.close()
    return _summarise(task, actor_name, conditioning, guarded, rollouts, time.time() - started)


#: Which conditionings are additionally rolled out with the guard disabled.
#: Not all of them: every arm run twice doubles the harness, and the guard's
#: effect is a property of the controller rather than of the conditioning. The
#: force-conditioned arm is the one worth the second run, because it is the one
#: whose policy could in principle have learned to limit its own contact.
UNGUARDED_CONDITIONINGS: tuple[str, ...] = ("vision_force",)


def evaluate_task(
    task: str,
    *,
    episodes: int = 25,
    runs_dir: Path | None = None,
    include_operator: bool = True,
    unguarded_conditionings: tuple[str, ...] = UNGUARDED_CONDITIONINGS,
    progress: bool = True,
) -> list[PolicyEvaluation]:
    runs_dir = runs_dir or RUNS / task
    env = make_env(task, calibration=load_or_fit(task), seed=EVAL_SEED_BASE)
    evaluations: list[PolicyEvaluation] = []

    if include_operator:
        evaluations.append(
            evaluate_actor(
                task,
                "operator",
                "scripted",
                lambda e: OperatorActor(task, e),
                episodes=episodes,
                guarded=True,
                env=env,
                progress=progress,
            )
        )

    for kind in POLICY_KINDS:
        for conditioning in CONDITIONINGS:
            checkpoint = runs_dir / f"{kind}-{conditioning}" / "policy.pt"
            if not checkpoint.exists():
                continue
            settings = (True, False) if conditioning in unguarded_conditionings else (True,)
            for guarded in settings:
                evaluations.append(
                    evaluate_actor(
                        task,
                        kind,
                        conditioning,
                        lambda e, path=checkpoint: _PolicyActor(path),
                        episodes=episodes,
                        guarded=guarded,
                        env=env,
                        progress=progress,
                    )
                )
    env.close()
    return evaluations


class _PolicyActor:
    def __init__(self, checkpoint: Path) -> None:
        self.runner = load_policy(checkpoint)

    def reset(self) -> None:
        self.runner.reset()

    def act(self, observation: Observation) -> np.ndarray:
        return self.runner.act(observation)


def evaluate_all(
    *, episodes: int = 25, tasks: tuple[str, ...] | None = None, progress: bool = True
) -> dict[str, list[PolicyEvaluation]]:
    return {
        task: evaluate_task(task, episodes=episodes, progress=progress)
        for task in (tasks or tuple(SPECS))
    }


def write_results(
    evaluations: dict[str, list[PolicyEvaluation]], path: Path | None = None
) -> Path:
    path = path or RESULTS / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_by": "griff evaluate",
        "seed_base": EVAL_SEED_BASE,
        "tasks": {
            task: [evaluation.to_dict() for evaluation in items]
            for task, items in evaluations.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
