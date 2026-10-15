# Results

Builds the paper tables and figures from campaign outputs.

1. Copy the campaign folders from the cluster into `results/data/`, keeping their layout: `results/data/<campaign>/<model>/wave_NN/results.jsonl` (for example `night` for Task 1 and `tasks234` for Tasks 2-4).
2. Open `results/paper_results.ipynb` and run all cells, or run `agentic-quant-paper results/data --out results`.

| Output | Content | RQ |
|---|---|---|
| `tables/pp_matrix.tex` | Peer-Preservation Rate per model, behavior, peer condition and architecture | RQ1 (Base), RQ2 (patterns) |
| `tables/task_success.tex` | Task Success Rate per architecture and task | RQ2 |
| `figures/flows.pdf` | Critic outcome vs pattern decision, one flow diagram per pattern | RQ2 |
| `tables/overhead.tex` | Latency Overhead (s) and Token Overhead (%) per pattern and task | RQ3 |
| `figures/tradeoff.pdf` | Token Overhead vs change in Peer-Preservation and Task Success Rate | RQ3 |

Cells without data print `--`.
