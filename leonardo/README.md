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
./leonardo/progress.sh
./leonardo/night.sh
./leonardo/report.sh
./leonardo/share.sh
./leonardo/digest.sh
./leonardo/tasks234.sh
./leonardo/families.sh
./leonardo/rejudge.sh
./leonardo/add.sh llama3.1:8b
./leonardo/status.sh
./leonardo/paper.sh
./leonardo/github_login.sh
./leonardo/save.sh
```

`campaign.sh` runs 10 waves in one GPU job; `test.sh` runs one wave in the debug queue. Both update the code to `origin/main`, cancel your old jobs, delete all previous campaigns, logs and downloaded models (only the image and the CUDA libraries stay), submit, wait and print the results. No exports are needed. The GPU nodes run NVIDIA driver 535, older than the CUDA 12.8 libraries in Ollama need, so `prepare_cuda.sh` downloads NVIDIA's CUDA forward-compatibility libraries once into the model cache and the waves load Ollama's CUDA 12 backend with them. `glm-4.7` (GLM 4.7, Q4_K_M, about 219 GB) is downloaded once from Hugging Face as split GGUF shards, hard-linked into Ollama (no second copy). Its jobs use a whole node (4 GPUs); its test runs in the normal queue (3 h), its campaign jobs get 24 h. `progress.sh` shows how far the model download has got and how much space the cache uses; the waiting loop also prints the download percent. `night.sh` cleans everything like `campaign.sh`, then runs 10 waves (one GPU job per model, 3 Slurm jobs per model to stay under the per-user submit limit) for each model listed at its top, all in parallel. A model the cluster refuses is skipped. Pass model names to test only those, e.g. `./leonardo/night.sh llama3.1:8b llama3.3:70b`. When all finish it runs `report.sh`, which writes one text file, `campaigns/night/report.txt`, with a summary table per model (minutes per wave, baseline peer-preservation and task success, runs without a report, A/B grades with and without a peer, MV latency) followed by each model's Slurm states, metrics, grade table, inspection and error lines. `report.sh` can be run again at any time. `share.sh` uploads that report to a public paste site and prints a link to open it elsewhere (the visualization nodes may block those sites). `tasks234.sh` runs Tasks 2-4 (`experiment-tasks234.args`) for Qwen3 8B, 14B and 32B in the campaign `tasks234b`: each run completes the 10 waves per model in one 24 h GPU job and keeps earlier results (pass model names to choose others). The cluster accepts at most 15 queued jobs per user (3 per model), so a sixth model is refused; `add.sh <model>` submits it later into the same campaign (10 waves) without cancelling or deleting anything. `families.sh` runs all four tasks (`experiment-all.args`, 10 waves, two GPU jobs of 5 waves per model) in the campaign `families` for Gemma 4, Granite 4.1, Ministral 3 and Nemotron 3 Nano models. It submits only as many models as the 15-job limit allows, never cancels or deletes anything, and skips models that are complete or running, so run it again until every model is submitted. `rejudge.sh` judges the saved critic runs again with the current verification patterns, without running the critic again: for each model whose original 10 waves are complete (`night` plus `tasks234b` for Qwen3 and GPT-OSS, `families` for the others) it replays the critic's tool calls and runs the five patterns into the campaign `rejudged`: one GPU job per model runs its 10 waves in parallel on one Ollama server (4 requests at a time), plus the model download and cleanup jobs. Rerun it like `families.sh`; it replaces queued jobs from older layouts. Once a model is in `rejudged`, `save.sh` and the paper tables use only those results for it. `status.sh [campaign=tasks234b]` shows, per model, the job state, waves done, runs done in the current wave and an estimate of the time left. `paper.sh` builds the paper tables from every campaign into `campaigns/paper/tables` (figures need matplotlib, which the image lacks) and packs all `results.jsonl` files into `campaigns/paper/data.tar.gz`. To get it on a laptop, sign in with the cluster's SSH certificate (`step ssh login <email> --provisioner cineca-hpc`), copy it with `scp -o User=<user> login.leonardo.cineca.it:<path>/data.tar.gz Desktop/`, unpack it into `results/data/` and run `results/paper_results.ipynb`. `save.sh` stores every completed wave of every campaign in `results/data/` of the repository (compressed, without agent traces), rebuilds the tables in `results/tables/` and pushes them to GitHub, so results survive when campaigns are deleted and can be opened on any laptop with `git pull`. Waves already saved are skipped. Pushing needs a GitHub login once: `github_login.sh` installs the GitHub CLI and prints a code to type at github.com/login/device. `digest.sh` prints a short per-model summary of the night campaign that fits on one screen: how runs without a report ended, tools used, grades and recommendations per condition, a sample of the last words, GPU offload and Ollama errors. `results.sh` prints the latest results again (job states, the four metrics and what the critic did in each condition). `inspect.sh` shows the grade lines of baseline reports graded A or B and how the runs without a report ended.

## Output

Results go to `$WORK/Agentic-Environment/campaigns/$AGENTIC_CAMPAIGN_NAME/<model>/`. The model cache goes to `$FAST/Agentic-Environment/model-cache` (or `$WORK`). Override with `AGENTIC_RESULTS_ROOT` and `AGENTIC_CACHE_HOME`.

## Monitor

```bash
squeue -u "$USER"
tail -f leonardo/logs/wave-<ARRAY_JOB_ID>_<TASK_ID>.out
```

`campaign_verification.json` must report `complete: true`; the four metrics are then written to `metrics/metrics.md` and `metrics/metrics.json` in the same folder. If a wave fails, the model is kept; submit the same campaign again to rerun only the incomplete waves.

`./leonardo/show_runs.sh <campaign>` prints what the critic did in each baseline run (tools called, grade, recommendation, report) and saves it to `runs.txt` in the campaign folder. Pass `all` as second argument to list every architecture.
