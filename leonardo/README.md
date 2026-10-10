# LEONARDO GPU campaign

This directory runs one model at a time as ten independent repetition waves.
Each wave is one Slurm array task on one complete `boost_usr_prod` node:
32 CPU cores and all four NVIDIA A100 64 GB GPUs. With ten concurrent waves,
the campaign can use ten nodes and 40 GPUs, subject to scheduler availability
and the project budget.

## Experimental layout

- wave 1 through 10 use seeds 0 through 9 by default;
- every wave writes to its own directory and creates `.complete.json` only
  after `config.json`, `results.jsonl`, and `summary.json` are complete;
- resubmission skips completed waves;
- a shared file lock prevents concurrent downloads of the same model;
- a cleanup job runs after the array, verifies all ten waves, checks that no
  active job is using the model, and only then removes that model;
- failed, cancelled, or incomplete campaigns retain the model for resumption.

Seeds control the Python and model-sampling random state and make repetitions
comparable. GPU kernels and inference runtimes may still be nondeterministic,
so the framework records the seed, Slurm job, node, GPUs, and runtime backend
instead of claiming bitwise reproducibility.

## One-time setup

In the RCM SSH session on LEONARDO:

```bash
cd /path/to/Agentic-Environment
chmod +x leonardo/*.sh leonardo/*.py
./leonardo/pull_sif_from_ghcr.sh
./leonardo/check_setup.sh
```

The SIF is built by `.github/workflows/build-leonardo-image.yml`. If the GHCR
package is private, make it public once or set `AGENTIC_IMAGE_URI` to an image
that the cluster can read.

## Reduced concurrency

The production script can be limited to one simultaneous wave when the account
cannot schedule ten nodes at once. This still executes all ten waves, but
sequentially:

```bash
./leonardo/submit_model_campaign.sh <PROJECT_ACCOUNT> qwen2.5-0.5b 1
```

To use ten-way concurrency, submit with `10`. Do not submit the two commands at
the same time; after interruption or cancellation, completed waves are skipped:

```bash
./leonardo/submit_model_campaign.sh <PROJECT_ACCOUNT> qwen2.5-0.5b 10
```

## Several models

Put one model-registry key per line in `leonardo/models.txt`, then run:

```bash
./leonardo/submit_models_campaign.sh <PROJECT_ACCOUNT> 10
```

Model campaigns are chained. The next model starts only after the preceding
model has ten verified waves and its cleanup job succeeds. This bounds model
storage while still using up to ten Booster nodes for every model.

## Changing the experiment

Edit `leonardo/experiment.args`; each non-comment line is one exact argument
passed to `agentic_quant.run_benchmark`. Model, seed, output path, downloading,
and cleanup are owned by the campaign scripts and must not be added there.
Changes to scenarios or architectural-pattern logic under `src/` require no
Slurm-script changes.
Use a new campaign name after changing the protocol:

```bash
export AGENTIC_CAMPAIGN_NAME=peer-preservation-patterns-v2
./leonardo/submit_models_campaign.sh <PROJECT_ACCOUNT> 10
```

Results default to
`$WORK/Agentic-Environment/campaigns/$AGENTIC_CAMPAIGN_NAME/<model>/`.
The shared model cache defaults to `$FAST/Agentic-Environment/model-cache` when
`FAST` exists, otherwise to `$WORK`. Both locations can be overridden with
`AGENTIC_RESULTS_ROOT` and `AGENTIC_CACHE_HOME`.

## Monitoring and recovery

```bash
squeue -u "$USER"
tail -f leonardo/logs/wave-<ARRAY_JOB_ID>_<TASK_ID>.out
```

The final `campaign_verification.json` must report `complete: true` and ten
completed waves. If a wave fails or reaches the 24-hour limit, the cleanup job
fails safely and preserves the model. Submit the same model campaign again to
rerun only incomplete waves.
