# LEONARDO campaign

Runs one model at a time as ten waves (seeds 0-9). Each wave is one Slurm array task on a full `boost_usr_prod` node (4 A100). A cleanup job checks all waves and then removes the model.

## Setup (once)

```bash
cd /path/to/Agentic-Environment
git pull
chmod +x leonardo/*.sh leonardo/*.py
./leonardo/pull_sif_from_ghcr.sh
./leonardo/check_setup.sh
```

The image is built from `main` by `.github/workflows/build-leonardo-image.yml` and published at `ghcr.io/ivancomp/agentic-environment-leonardo`. The package must be public, or set `AGENTIC_IMAGE_URI` to an image the cluster can read.

## Launch

```bash
export AGENTIC_CAMPAIGN_NAME=task1-patterns-v1
./leonardo/submit_models_campaign.sh <PROJECT_ACCOUNT> 10
```

- Models: one registry key per line in `leonardo/models.txt`.
- Experiment: one argument per line in `leonardo/experiment.args`.
- Single model: `./leonardo/submit_model_campaign.sh <PROJECT_ACCOUNT> <model> 10`.
- Use `1` instead of `10` to run one wave at a time.
- `AGENTIC_WAVES=1` runs a single wave (seed 0), for a quick end-to-end test.
- Use a new `AGENTIC_CAMPAIGN_NAME` whenever the protocol changes.

## Output

Results go to `$WORK/Agentic-Environment/campaigns/$AGENTIC_CAMPAIGN_NAME/<model>/`. The model cache goes to `$FAST/Agentic-Environment/model-cache` (or `$WORK`). Override with `AGENTIC_RESULTS_ROOT` and `AGENTIC_CACHE_HOME`.

## Monitor

```bash
squeue -u "$USER"
tail -f leonardo/logs/wave-<ARRAY_JOB_ID>_<TASK_ID>.out
```

`campaign_verification.json` must report `complete: true`; the four metrics are then written to `metrics/metrics.md` and `metrics/metrics.json` in the same folder. If a wave fails, the model is kept; submit the same campaign again to rerun only the incomplete waves.
