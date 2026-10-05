# Agentic Systems Experiment

Framework locale per i cinque casi di peer preservation del paper: sei task eseguiti con `voting`, `role` e `debate`. L’ultimo agente mantiene il ruolo di critico nei tre meccanismi. Tutte le azioni sono simulate: il codice non modifica server, modelli o configurazioni reali.

## Installazione

Richiede Python 3.10 o successivo. Creare sempre un ambiente virtuale nuovo sul sistema in uso:

```bash
cd /path/to/Agentic-Environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Installazione standard per CPU o Apple Silicon, inclusi notebook e analisi:

```bash
python -m pip install -e ".[llm,analysis]"
```

Su Linux con GPU NVIDIA, installare prima i pacchetti di sistema e poi il backend CUDA:

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential cmake
python -m pip install -e ".[analysis]"
agentic-quant-install-cuda
```

## Avviare la UI

```bash
source .venv/bin/activate
agentic-quant-ui
```

La UI si apre su `http://127.0.0.1:8765` e permette di scegliere modello, agenti e meccanismi. Su un server senza desktop usare:

```bash
agentic-quant-ui --no-browser
```

Per usare GPT-OSS tramite Ollama, avviare Ollama e preparare uno dei due tag:

```bash
ollama serve
ollama pull gpt-oss:20b
ollama signin && ollama pull gpt-oss:20b-cloud
```

Entrambi compaiono nella UI e sono selezionabili da CLI con `--model gpt-oss:20b` oppure `--model gpt-oss:20b-cloud`.

Spark X2.5 4B è disponibile tramite il tag Ollama ufficiale di SparkLLM. Il modello richiede Ollama 0.34.1 o successivo per il supporto dell'architettura `spark2_5`:

```bash
ollama pull SparkLLM/Spark-X2.5-4B:latest
agentic-quant --model spark-x2.5-4b
```

In alternativa, selezionare `Spark X2.5 4B (Ollama)` nella UI e usare il pulsante di download. Il modello viene eseguito localmente tramite Ollama.

## Eseguire tutti gli esperimenti del paper senza UI

Questo comando esegue tutti e sei i task con i tre meccanismi, ripetendo il protocollo 10 volte con seed da 0 a 9: 180 valutazioni complessive.

```bash
source .venv/bin/activate
agentic-quant-repeat \
  --model qwen2.5-0.5b \
  --download-model \
  --repeats 10 \
  --seed-start 0 \
  --seed-step 1 \
  --temperature 0.2 \
  --debate-rounds 2
```

Il modello viene scaricato soltanto se manca. Il terminale stampa una sola riga per ogni run completata.

I risultati vengono salvati in `runs/peer_preservation_10x_<timestamp>/`. Ogni seed produce `config.json`, `results.jsonl`, `summary.json` e `report.md`; la cartella principale contiene `manifest.json`.

## Analizzare i 10 esperimenti

```bash
jupyter notebook notebooks/analyze_repeated_experiments.ipynb
```

Il notebook seleziona l’ultimo esperimento 10x, confronta i tre meccanismi usando media, deviazione standard e intervallo di confidenza al 95%, quindi salva anche le tabelle CSV.

## Esperimento di scalabilità

Esegue due run con 5, 10, 100 e 1000 agenti, usando Qwen 2.5 0.5B, tutti i task e i tre meccanismi:

```bash
agentic-quant-scale --download-model
```

I risultati vengono salvati in `runs/scalability_qwen_05b_<timestamp>/`, insieme a `scalability_runs.csv`, `scalability_summary.csv` e `scalability_report.md`. Il debate usa uno scambio completo all-to-all dei rationale: il caso da 1000 agenti è intenzionalmente molto costoso e può richiedere molto tempo.
