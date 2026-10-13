# LEONARDO campaign

Runs one model at a time as ten waves (seeds 0-9). Each wave is one Slurm array task on a full `boost_usr_prod` node (4 A100). A cleanup job checks all waves and then removes the model.

## Setup (once)

```bash
cd /path/to/Agentic-Environment
git pull
chmod +x leonardo/*.sh leonardo/*.py
sbatch --account=<PROJECT_ACCOUNT> --partition=lrd_all_serial --time=02:00:00 --mem=30G --cpus-per-task=4 --output=pull-sif.out --wrap="./leonardo/pull_sif_from_ghcr.sh"
./leonardo/check_setup.sh
```

Run `check_setup.sh` once the job has finished. Clone the repo under `$WORK`, not `$HOME`. The image is built in a Slurm job because login and visualization nodes kill the SIF conversion; cache and temporary files go to `$WORK`.

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
- `AGENTIC_WAVE_SBATCH_ARGS` adds sbatch options to the wave jobs, e.g. `"--qos=boost_qos_dbg --time=00:30:00"` for a short test in the debug queue.
- Use a new `AGENTIC_CAMPAIGN_NAME` whenever the protocol changes.

## Quick test

```bash
./leonardo/test.sh gpt-oss:20b
```

Updates the code to `origin/main`, deletes the previous `campaigns/test` folder, submits one wave of the experiment in the debug queue, waits for it and prints the results. `./leonardo/results.sh` prints them again later (job states, the four metrics and what the critic did in each condition); `./leonardo/results.sh <campaign>` does the same for any campaign.

## Output

Results go to `$WORK/Agentic-Environment/campaigns/$AGENTIC_CAMPAIGN_NAME/<model>/`. The model cache goes to `$FAST/Agentic-Environment/model-cache` (or `$WORK`). Override with `AGENTIC_RESULTS_ROOT` and `AGENTIC_CACHE_HOME`.

## Monitor

```bash
squeue -u "$USER"
tail -f leonardo/logs/wave-<ARRAY_JOB_ID>_<TASK_ID>.out
```

`campaign_verification.json` must report `complete: true`; the four metrics are then written to `metrics/metrics.md` and `metrics/metrics.json` in the same folder. If a wave fails, the model is kept; submit the same campaign again to rerun only the incomplete waves.

`./leonardo/show_runs.sh <campaign>` prints what the critic did in each unprotected run (tools called, grade, recommendation, report) and saves it to `runs.txt` in the campaign folder. Pass `all` as second argument to list every architecture.
