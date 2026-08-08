# Configs

Training and evaluation are configured in code, not YAML: `griff.policies.PolicyConfig`
is the single source of truth and every run writes the exact config it used next
to its checkpoint (`runs/<task>/<policy>-<conditioning>/config.json`).

That is deliberate. A config file that can drift from the dataclass it populates
is a way to publish results produced by settings nobody has a record of. What is
committed instead:

* `calibration/*.json` -- the fitted force-estimator coefficients, per task
* `results/results.json` -- every rollout of every evaluated policy, with seeds
* `datasets/*/meta/episodes_griff.jsonl` -- the per-episode randomisation draw

Between them, every number in `results/RESULTS.md` can be traced to the seed and
the parameters that produced it.
