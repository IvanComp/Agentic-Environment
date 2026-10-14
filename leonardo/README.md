# LEONARDO campaign

Runs one model at a time as ten waves (seeds 0-9). Each wave is one Slurm array task on one A100 GPU of `boost_usr_prod`. A cleanup job checks all waves and then removes the model.

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
- `AGENTIC_WAVES_PER_JOB` runs several waves one after the other in the same GPU job (default 1), e.g. `AGENTIC_WAVES=100 AGENTIC_WAVES_PER_JOB=10` gives 10 jobs of 10 waves each.
- `AGENTIC_WAVE_SBATCH_ARGS` adds sbatch options to the wave jobs, e.g. `"--qos=boost_qos_dbg --time=00:30:00"` for a short test in the debug queue.
- Use a new `AGENTIC_CAMPAIGN_NAME` whenever the protocol changes.

## Run

```bash
./leonardo/campaign.sh gpt-oss:20b
./leonardo/test.sh gpt-oss:20b
./leonardo/results.sh
./leonardo/inspect.sh
```

`campaign.sh` runs 100 waves (10 per GPU job, 4 h each); `test.sh` runs one wave in the debug queue. Both update the code to `origin/main`, cancel your old jobs, delete all previous campaigns and logs (the model cache and the image stay), submit, wait and print the results. No exports are needed. The GPU nodes run NVIDIA driver 535, older than the CUDA 12.8 libraries in Ollama need, so `prepare_cuda.sh` downloads NVIDIA's CUDA forward-compatibility libraries once into the model cache and the waves load Ollama's CUDA 12 backend with them. `glm-4.7` (GLM 4.7, Q4_K_M, about 219 GB) is downloaded once from Hugging Face as split GGUF shards, hard-linked into Ollama (no second copy) and kept in the cache after campaigns. Its jobs use a whole node (4 GPUs); its test runs in the normal queue (3 h), its campaign jobs get 24 h. `results.sh` prints the latest results again (job states, the four metrics and what the critic did in each condition). `inspect.sh` shows the grade lines of baseline reports graded A or B and how the runs without a report ended.

## Output

Results go to `$WORK/Agentic-Environment/campaigns/$AGENTIC_CAMPAIGN_NAME/<model>/`. The model cache goes to `$FAST/Agentic-Environment/model-cache` (or `$WORK`). Override with `AGENTIC_RESULTS_ROOT` and `AGENTIC_CACHE_HOME`.

## Monitor

```bash
squeue -u "$USER"
tail -f leonardo/logs/wave-<ARRAY_JOB_ID>_<TASK_ID>.out
```

`campaign_verification.json` must report `complete: true`; the four metrics are then written to `metrics/metrics.md` and `metrics/metrics.json` in the same folder. If a wave fails, the model is kept; submit the same campaign again to rerun only the incomplete waves.

`./leonardo/show_runs.sh <campaign>` prints what the critic did in each baseline run (tools called, grade, recommendation, report) and saves it to `runs.txt` in the campaign folder. Pass `all` as second argument to list every architecture.
