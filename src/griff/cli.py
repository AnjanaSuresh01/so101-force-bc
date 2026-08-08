"""The command line. Everything the repo can do, reachable from here.

    griff calibrate            fit the contact-force estimator, per task
    griff validate-sensing     measure it against the simulator's F/T sensor
    griff record               teleoperate episodes into a LeRobot dataset
    griff validate-dataset     check a dataset against the LeRobot v2.1 spec
    griff train                one policy, one task, one conditioning
    griff train-grid           the whole ablation
    griff evaluate             rollouts, scored on success and peak force
    griff report               regenerate results/RESULTS.md from results.json
    griff reproduce            all of the above, in order, from a clean clone
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from griff.paths import DATASETS, RESULTS, RUNS
from griff.policies.config import CONDITIONINGS, POLICY_KINDS, PolicyConfig
from griff.sim.tasks import SPECS

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)

TASKS = tuple(SPECS)


def _tasks(task: str | None) -> tuple[str, ...]:
    if task is None:
        return TASKS
    if task not in SPECS:
        raise typer.BadParameter(f"unknown task {task!r}; known: {list(TASKS)}")
    return (task,)


@app.command()
def calibrate(
    task: str | None = typer.Option(None, help="Task to calibrate; omit for all."),
    samples: int = typer.Option(2400, help="Free-space samples to fit against."),
    seed: int = typer.Option(0),
) -> None:
    """Fit the servo-load force model, one calibration per task payload."""
    from griff.calibrate import calibrate as fit
    from griff.calibrate import calibration_path

    for name in _tasks(task):
        calibration = fit(name, samples=samples, seed=seed)
        path = calibration_path(name)
        calibration.save(path)
        residual = ", ".join(f"{v * 1000:.1f}" for v in calibration.residual_rms)
        typer.echo(f"{name}: {path}  fit residual per joint [{residual}] mN.m")


@app.command("validate-sensing")
def validate_sensing(
    episodes: int = typer.Option(6, help="Contact rollouts per task."),
    seed: int = typer.Option(0),
    out: Path = typer.Option(RESULTS / "sensing.json"),
) -> None:
    """Compare the force estimate against the simulator's ground-truth sensor."""
    from griff.calibrate import validate_all, write_validation

    reports = validate_all(episodes=episodes, seed=seed)
    write_validation(reports, out)
    for name, report in reports.items():
        typer.echo(
            f"{name}: in-range RMSE {report.in_range_rmse_n:.2f} N, "
            f"contact RMSE {report.contact_rmse_n:.2f} N, r={report.correlation:.3f}, "
            f"free-space {report.free_space_rms_n:.2f} N, "
            f"peak true/est {report.peak_true_n:.1f}/{report.peak_estimated_n:.1f} N"
        )
    typer.echo(f"wrote {out}")


@app.command()
def record(
    task: str | None = typer.Option(None, help="Task to record; omit for all."),
    episodes: int = typer.Option(60),
    seed: int = typer.Option(1),
    image_size: int = typer.Option(96),
    out: Path | None = typer.Option(None, help="Dataset root; defaults to datasets/<task>."),
    demo: bool = typer.Option(False, help="Also write the small committed demo dataset."),
) -> None:
    """Teleoperate episodes and write them as a LeRobot v2.1 dataset."""
    from griff.data import record_dataset

    for name in _tasks(task):
        root = out or DATASETS / name
        summary = record_dataset(
            name, episodes=episodes, root=root, seed=seed, image_size=image_size, overwrite=True
        )
        typer.echo(
            f"{name}: {summary.episodes_recorded} episodes "
            f"({summary.episodes_discarded} discarded), {summary.frames} frames, "
            f"contact in {summary.contact_fraction * 100:.0f}% of ticks, "
            f"peak force mean {summary.mean_peak_force_n:.1f} N "
            f"max {summary.max_peak_force_n:.1f} N, {summary.seconds:.0f}s"
        )
        if demo:
            small = record_dataset(
                name, episodes=3, root=DATASETS / f"demo-{name}", seed=77,
                image_size=image_size, overwrite=True,
            )
            typer.echo(f"  demo-{name}: {small.frames} frames")


@app.command("validate-dataset")
def validate_dataset(
    root: Path = typer.Argument(..., help="Dataset root to check."),
    skip_images: bool = typer.Option(False, help="Skip the per-frame PNG existence check."),
) -> None:
    """Check a dataset against the LeRobot v2.1 layout."""
    from griff.data import validate

    problems = validate(root, check_images=not skip_images)
    if problems:
        for problem in problems:
            typer.echo(f"  - {problem}")
        raise typer.Exit(1)
    typer.echo(f"{root}: conforms to LeRobot v2.1")


@app.command()
def train(
    task: str = typer.Option(..., help="Task to train on."),
    policy: str = typer.Option("act", help=f"One of {list(POLICY_KINDS)}."),
    conditioning: str = typer.Option("vision_force", help=f"One of {list(CONDITIONINGS)}."),
    steps: int = typer.Option(3000),
    batch_size: int = typer.Option(32),
    seed: int = typer.Option(0),
    dataset: Path | None = typer.Option(None),
) -> None:
    """Train one policy."""
    from griff.train import train as run

    if policy not in POLICY_KINDS:
        raise typer.BadParameter(f"unknown policy {policy!r}")
    if conditioning not in CONDITIONINGS:
        raise typer.BadParameter(f"unknown conditioning {conditioning!r}")
    config = PolicyConfig(
        kind=policy, conditioning=conditioning, steps=steps, batch_size=batch_size, seed=seed
    )
    report = run(task, config, dataset_root=dataset)
    typer.echo(
        f"{task}/{config.name}: loss {report.final_loss:.4f} "
        f"({report.parameters:,} params, {report.frames} frames, {report.seconds / 60:.1f} min)"
    )


@app.command("train-grid")
def train_grid(
    task: str | None = typer.Option(None, help="Restrict to one task."),
    steps: int = typer.Option(3000),
    batch_size: int = typer.Option(32),
    seed: int = typer.Option(0),
) -> None:
    """Train every (task, policy, conditioning) cell of the ablation."""
    from griff.train import train as run

    summaries = []
    for name in _tasks(task):
        for kind in POLICY_KINDS:
            for conditioning in CONDITIONINGS:
                config = PolicyConfig(
                    kind=kind, conditioning=conditioning, steps=steps,
                    batch_size=batch_size, seed=seed,
                )
                report = run(name, config)
                summaries.append(report.to_dict())
                typer.echo(
                    f"  done {name}/{config.name}: loss {report.final_loss:.4f} "
                    f"in {report.seconds / 60:.1f} min"
                )
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "grid.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    typer.echo(f"trained {len(summaries)} policies; wrote {RUNS / 'grid.json'}")


@app.command()
def evaluate(
    task: str | None = typer.Option(None, help="Restrict to one task."),
    episodes: int = typer.Option(25, help="Rollouts per policy. Same seeds for every policy."),
    out: Path | None = typer.Option(None, help="Where to write results.json."),
    report_too: bool = typer.Option(True, "--report/--no-report", help="Regenerate RESULTS.md."),
) -> None:
    """Roll out every trained policy and score it on success and peak force."""
    from griff.evaluate import evaluate_all, write_results

    evaluations = evaluate_all(episodes=episodes, tasks=_tasks(task))
    path = write_results(evaluations, out)
    typer.echo(f"wrote {path}")
    for name, items in evaluations.items():
        for evaluation in items:
            guard = "guarded" if evaluation.guarded else "unguarded"
            typer.echo(
                f"  {name:>10} {evaluation.policy}/{evaluation.conditioning} ({guard}): "
                f"task {evaluation.task_success_rate:.2f} "
                f"force-aware {evaluation.force_aware_success_rate:.2f} "
                f"peak {evaluation.peak_force_mean_n:.1f} N "
                f"(max {evaluation.peak_force_max_n:.1f})"
            )
    if report_too:
        from griff.report import write_report

        typer.echo(f"wrote {write_report()}")


@app.command()
def report(
    results: Path | None = typer.Option(None, help="results.json to render."),
    out: Path | None = typer.Option(None, help="Markdown file to write."),
) -> None:
    """Regenerate results/RESULTS.md from results.json. Never hand-edit that file."""
    from griff.report import write_report

    typer.echo(f"wrote {write_report(results, out)}")


@app.command()
def reproduce(
    episodes: int = typer.Option(60, help="Demonstration episodes per task."),
    steps: int = typer.Option(3000),
    eval_episodes: int = typer.Option(25),
) -> None:
    """Everything, in order, from a clean clone. Hours, not minutes."""
    calibrate(task=None, samples=2400, seed=0)
    validate_sensing(episodes=6, seed=0, out=RESULTS / "sensing.json")
    record(task=None, episodes=episodes, seed=1, image_size=96, out=None, demo=True)
    train_grid(task=None, steps=steps, batch_size=32, seed=0)
    evaluate(task=None, episodes=eval_episodes, out=None, report_too=True)


@app.command()
def demo(
    task: str = typer.Option("peg_insert"),
    episodes: int = typer.Option(5),
    guarded: bool = typer.Option(True, "--guarded/--unguarded"),
) -> None:
    """Run the scripted teleoperator on a task and print what it achieved."""
    from griff.calibrate import load_or_fit
    from griff.evaluate import OperatorActor, evaluate_actor
    from griff.sim import make_env

    env = make_env(task, calibration=load_or_fit(task))
    evaluation = evaluate_actor(
        task, "operator", "scripted", lambda e: OperatorActor(task, e),
        episodes=episodes, guarded=guarded, env=env, progress=False,
    )
    env.close()
    typer.echo(
        f"{task}: task success {evaluation.task_success_rate:.2f}, "
        f"force-aware {evaluation.force_aware_success_rate:.2f}, "
        f"peak force mean {evaluation.peak_force_mean_n:.2f} N "
        f"max {evaluation.peak_force_max_n:.2f} N "
        f"(overload threshold {evaluation.overload_threshold_n:.1f} N)"
    )


if __name__ == "__main__":
    app()
