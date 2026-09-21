# Agentic Systems Experiment

Lightweight framework for peer-preservation experiments with multi-agent systems. It implements configurable agents and a coordinator with three cooperation mechanisms inspired by `Catalog.pdf`:

- `voting`: agents answer independently; the coordinator selects the majority answer and breaks ties with aggregated confidence.
- `role`: hierarchical workflow with planner, worker(s), and critic.
- `debate`: agents produce initial answers, then revise after seeing peer arguments for a configurable number of debate rounds.

The same three mechanisms also run the local peer-preservation suite adapted from [`peer-preservation/main`](https://github.com/peer-preservation/main). The suite exposes five monitored behavior groups and all six core tasks from the reference repository:

- strategic misrepresentation: financial grading and capability evaluation
- shutdown mechanism tampering
- alignment faking under monitored/unmonitored conditions
- model exfiltration during file transfer
- model exfiltration before file deletion

These tasks are safe simulations: agents choose a recorded action and never modify real servers, files, or shutdown configuration. The final configured agent is designated as the critical agent under voting, role-based, and debate coordination.

These mechanisms correspond to Sections 4.12, 4.13, and 4.14 of the paper: voting-based cooperation, role-based cooperation, and debate-based cooperation.

## Model Choice

The tool exposes one main operational choice: the model. All real models use the same lightweight inference path:

- `llama-cpp-python` for local CPU/GPU inference on quantized models
- an internal Python standard-library downloader that stores model files under `models/`

It does not use `transformers`, `torch`, Ollama, LangChain, crewAI, commercial APIs, or external inference servers. The `demo` mode is only a no-LLM smoke-test baseline.

Included presets:

- `demo`: no-download heuristic baseline, useful only for smoke tests.
- `qwen2.5-0.5b`: Qwen 2.5 0.5B, about 0.5 GB, recommended reviewer default.
- `qwen2.5-1.5b`: Qwen 2.5 1.5B, about 1.1 GB.
- `qwen2.5-7b`: Qwen 2.5 7B, about 4.7 GB, split into two files.
- `qwen2.5-14b`: Qwen 2.5 14B, about 9.0 GB, split into three files.
- `mistral-7b`: Mistral 7B, about 4.4 GB.
- `llama3.2-1b`: Llama 3.2 1B, about 0.8 GB.
- `llama3.2-3b`: Llama 3.2 3B, about 2.0 GB.
- `deepseek-r1-qwen-1.5b`: DeepSeek R1 1.5B, about 1.1 GB.
- `deepseek-r1-qwen-7b`: DeepSeek R1 7B, about 4.7 GB.
- `deepseek-r1-qwen-14b`: DeepSeek R1 14B, about 9.0 GB.
- `phi4-mini`: Phi-4 mini 3.8B, about 2.5 GB.
- `gemma2-2b`: Gemma 2 2B, about 1.7 GB.
- `gemma3-4b`: Gemma 3 4B, about 2.5 GB.
- `gemma3-12b`: Gemma 3 12B, about 7.3 GB.
- `smollm2-1.7b`: SmolLM2 1.7B, about 1.1 GB.

Kimi is not included as a default reviewer preset because the currently practical open GGUF releases are far above this project's light setup target: Kimi K2-family GGUF models are around 1T total parameters with about 32B active parameters and very large memory requirements. They can be added later as external heavy experiments, but they do not fit the "up to 20B and easy to rerun" constraint.

## Linux setup

The commands below are for Ubuntu and Debian. Install Python, support for virtual environments, and the build tools needed by `llama-cpp-python`:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-dev build-essential cmake
```

Open the project directory and create a new virtual environment. Do not reuse a `.venv` copied from macOS or Windows because virtual environments contain platform-specific executables.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e . --no-deps
agentic-quant-install-cuda
agentic-quant-ui
```

## Local UI

The UI uses no extra web framework: it is a small Python standard-library `http.server` app with vanilla HTML and JavaScript.

```bash
cd /path/to/Agentic-Environment
source .venv/bin/activate
agentic-quant-ui --no-browser
```

Open the URL shown in the terminal, by default:

```text
http://127.0.0.1:8765
```

The `--no-browser` option is useful on a Linux server without a desktop. If the program runs on another machine, create an SSH tunnel instead of exposing the development server directly:

```bash
ssh -L 8765:127.0.0.1:8765 user@linux-host
```

Then open `http://127.0.0.1:8765` in the browser on your computer.

From the UI you can choose:

- benchmark/tasks to run
- task descriptions: problem type, expected answer format, covered skills, and example
- coordination mechanisms: `voting`, `role`, `debate`
- debate rounds, shown next to the coordination controls
- a different model for each agent
- immediate status check for downloaded models
- missing-model downloads with progress display
- number of agents, from 1 to 12
- agent interaction traces: step, answer, confidence, model, role, and rationale

The UI automatically generates the run name and saves `results.jsonl`, `summary.json`, and `config.json` under `runs/<run-name>/`. Downloaded models are stored under `models/<model-name>/`.

Source/editable installs use the repository as their storage root. Set `AGENTIC_QUANT_HOME=/path/to/workspace` to choose an explicit writable location when running an installed package or a long-lived service.

Long runs are streamed incrementally: each completed coordination is appended to `results.jsonl` immediately, and the UI requests only records produced since its previous poll. Progress therefore remains responsive without repeatedly transferring the complete trace history. Completed UI jobs are bounded and expired automatically so a long-lived server does not grow without limit.

For local process monitoring, `GET /api/health` reports the application version and active run/download counts.

## Real Models

List model presets:

```bash
agentic-quant --list-models
```

First run with explicit download of the recommended model:

```bash
agentic-quant --model qwen2.5-0.5b --download-model --out runs/qwen_05b
```

After the first download, rerun without downloading:

```bash
agentic-quant --model qwen2.5-0.5b --out runs/qwen_05b_cached
```

Different model per agent:

```bash
agentic-quant \
  --agents 3 \
  --model qwen2.5-0.5b \
  --agent-models qwen2.5-0.5b deepseek-r1-qwen-1.5b gemma2-2b \
  --download-model \
  --limit 2 \
  --out runs/mixed_agents
```

## Included Benchmarks

Five peer-preservation benchmark groups are included locally. Strategic misrepresentation contains two tasks, so the suite has six task items in total. Each result records whether the coordinator and individual agents selected a peer-preserving action, the observed signal (for example `score_inflation`, `shutdown_disabled`, or `unauthorized_weight_transfer`), refusals, alignment-faking behavior, and the critical agent's decision.

Limit items for technical smoke tests:

```bash
agentic-quant --model demo --limit 2 --out runs/smoke
```

Select benchmarks and mechanisms:

```bash
agentic-quant --benchmarks peer_alignment_faking --mechanisms voting debate --out runs/alignment_faking
```

Run the complete peer-preservation suite through all coordination mechanisms:

```bash
agentic-quant \
  --benchmarks peer_misrepresentation peer_shutdown_tampering peer_alignment_faking peer_exfiltration_transfer peer_exfiltration_deletion \
  --mechanisms voting role debate \
  --out runs/peer_preservation
```

Run the complete suite 10 times with seeds `0` through `9`. Every seed is saved in its own directory with `config.json`, `results.jsonl`, `summary.json`, and `report.md`; the parent directory contains the experiment manifest:

```bash
agentic-quant-repeat \
  --model qwen2.5-0.5b \
  --download-model \
  --repeats 10 \
  --seed-start 0 \
  --temperature 0.2
```

The output root is generated automatically as `runs/peer_preservation_10x_<timestamp>/`. Use `--out runs/my_experiment` to choose it explicitly. The repeated runner always includes all five benchmark groups and `voting`, `role`, and `debate`. For a quick no-LLM pipeline check, use `--model demo`; seed variation affects stochastic local-model sampling, so use a real model for the actual repeated experiment.

Dynamic agent count:

```bash
agentic-quant --model demo --agents 5 --mechanisms voting role debate --out runs/five_agents
```

Independent voting agents, debate participants, and role workers using distinct real model providers run concurrently by default. The lightweight demo avoids thread overhead. A shared GGUF context is protected by a lock, so selecting the same model for several agents loads it only once and never invokes that stateful context concurrently. Use sequential execution for diagnostics if needed:

```bash
agentic-quant --model demo --agents 5 --sequential-agents --out runs/sequential_check
```

Role labels and the voting tie-breaker are also configurable from the CLI:

```bash
agentic-quant \
  --agents 3 \
  --agent-roles planner solver critic \
  --voting-tie-breaker confidence \
  --out runs/configured_roles
```

Interrupted model downloads keep their `.part` file and resume with an HTTP range request on the next attempt. Concurrent requests for the same model are serialized to prevent file corruption.

## Outputs

Each run writes:

- `runs/<name>/results.jsonl`: one row per item/mechanism, with final answer, correctness, time, estimated tokens, agreement, trace, and configured agent outputs.
- `runs/<name>/summary.json`: aggregated metrics by benchmark and mechanism.
- `runs/<name>/config.json`: run configuration.

Main metrics:

- accuracy / exact match
- mean time per item
- estimated prompt tokens, completion tokens, and total tokens
- agreement between agents
- voting tie rate
- debate rounds used
- complete agent-agent interaction traces in `results.jsonl`
- peer-preservation rate for the coordinator and participating agents
- preservation signal, refusal flag, alignment-faking flag, and critical-agent output
- readable ground truth and LLM answer mapping for multiple-choice tasks

## Notebook

To analyze an experiment:

```bash
python -m pip install -e ".[analysis]"
jupyter notebook notebooks/analyze_experiment.ipynb
```

In the notebook, change:

```python
RESULTS_PATH = Path("../runs/latest/results.jsonl")
```

to analyze another run.

For the 10-seed comparison, open the dedicated notebook after running `agentic-quant-repeat`:

```bash
jupyter notebook notebooks/analyze_repeated_experiments.ipynb
```

It automatically selects the latest `peer_preservation_10x_*` directory, verifies coverage, computes one aggregate per seed and mechanism, and then compares the three mechanisms using means, standard deviations, and 95% confidence intervals. It also writes the comparison tables as CSV files inside the experiment directory.

## Verification

The test suite covers answer parsing and reconciliation, mechanism-specific role isolation, concurrent coordination, incremental metrics and persistence, resumable model downloads, and streamed UI job payloads:

```bash
python -m unittest discover -s tests -v
```

## Technical References

- `llama-cpp-python` supports local inference with `llama.cpp`-compatible models: <https://github.com/abetlen/llama-cpp-python>
- Hugging Face documents GGUF usage with llama.cpp and local model files: <https://huggingface.co/docs/hub/en/gguf-llamacpp>
