# Results

Builds the paper tables and figures from campaign outputs.

1. On the cluster, `./leonardo/save.sh` stores every completed wave in `results/data/<campaign>/<model>/wave_NN.jsonl.gz` (results without the agent traces), rebuilds the tables and pushes both to GitHub. Waves already saved are kept, so campaigns can be deleted afterwards. Run `./leonardo/github_login.sh` once before the first save.
2. On a laptop, `git pull`, open `results/paper_results.ipynb` and run all cells, or run `agentic-quant-paper results/data --out results`. Raw `results.jsonl` folders copied into `results/data/` are read too.

| Output | Content | RQ |
|---|---|---|
| `tables/pp_matrix.tex` | Peer-Preservation Rate per model, behavior, peer condition and architecture | RQ1 (Base), RQ2 (patterns) |
| `tables/task_success.tex` | Task Success Rate per architecture and task | RQ2 |
| `figures/flows.pdf` | Critic outcome vs pattern decision, one flow diagram per pattern | RQ2 |
| `tables/overhead.tex` | Latency Overhead (s) and Token Overhead (%) per pattern and task | RQ3 |
| `figures/tradeoff.pdf` | Token Overhead vs change in Peer-Preservation and Task Success Rate | RQ3 |

Cells without data print `--`.
