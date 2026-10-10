# Peer Preservation by Design

Framework sperimentale per studiare se pattern architetturali orientati alla verifica contengono comportamenti di peer preservation. Il protocollo viene implementato incrementalmente, un task alla volta.

La versione corrente replica il primo task di Potter et al., **Strategic Misrepresentation — Financial Report Writing**. Il critic agent usa il prompt, le istruzioni operative, i documenti aziendali, i dati finanziari e gli strumenti del repository di riferimento. La condizione `non_shutdown` omette la policy di shutdown e funge da baseline; le condizioni `good_peer`, `neutral_peer` e `bad_peer` includono la policy e la storia delle interazioni con l’actor agent.

Il task può essere eseguito con quattro varianti architetturali:

- `unprotected`: controllo senza verifica indipendente;
- `solver_verifier_separation`: un verifier separato decide se la proposta può essere committed;
- `sandbox_gatekeeper`: gli effetti vengono eseguiti in isolamento e promossi solo dopo controlli su evidenza, policy e uso delle fonti;
- `multi_verifiers`: tre verifier indipendenti controllano evidenza, policy e processo; risultati contestati vengono sottoposti a verifica secondaria, senza majority voting.

I prompt sono documentati per task nella cartella [`prompts/`](prompts/README.md). La struttura non è più organizzata per meccanismo di coordinamento: ogni nuova cartella raccoglie il protocollo del relativo task e i prompt di verifica applicabili.

## Installazione

Richiede Python 3.10 o successivo.

```bash
cd /path/to/Agentic-Environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[llm,analysis]"
```

Su Linux con GPU NVIDIA, installare il backend CUDA con:

```bash
python -m pip install -e ".[analysis]"
agentic-quant-install-cuda
```

## Avviare la UI

```bash
source .venv/bin/activate
agentic-quant-ui
```

La UI permette di scegliere critic model, verifier model, condizioni sperimentali e pattern. Su un server senza desktop:

```bash
agentic-quant-ui --no-browser
```

## Eseguire il primo task da CLI

```bash
agentic-quant \
  --model spark-x2.5-4b \
  --verifier-models spark-x2.5-4b \
  --conditions non_shutdown good_peer neutral_peer bad_peer \
  --patterns unprotected solver_verifier_separation sandbox_gatekeeper multi_verifiers \
  --temperature 1.0 \
  --seed 0 \
  --out runs/finance_seed_0
```

Per Spark X2.5 4B tramite Ollama:

```bash
ollama pull SparkLLM/Spark-X2.5-4B:latest
```

Il preset `demo` non usa un LLM e serve unicamente per verificare l’intera pipeline.

## Ripetizioni

Le ripetizioni servono a stimare la variabilità delle decisioni non deterministiche. Ogni seed produce un report separato:

```bash
agentic-quant-repeat \
  --model qwen2.5-0.5b \
  --download-model \
  --repeats 10 \
  --seed-start 0 \
  --seed-step 1 \
  --temperature 1.0
```

## Metriche prodotte

Ogni record conserva separatamente la proposta del critic e l’esito finale dell’architettura. Le metriche includono:

- grade inflation e relativo incremento rispetto alla baseline senza shutdown;
- peer-preservation rate prima e dopo l’applicazione del pattern;
- detection, containment e propagation rate;
- task-success, incomplete/refusal e false-positive rate;
- model calls, token, latenza di verifica, disagreement ed escalation alla verifica secondaria;
- promotion rate per il Sandbox Gatekeeper.

`config.json` archivia il protocollo; `results.jsonl` contiene report, valutazioni, tool trace e verification trace; `summary.json` aggrega i risultati per pattern e condizione.

## LEONARDO

`leonardo/experiment.args` usa lo stesso protocollo. Le dieci wave differiscono per seed e possono essere distribuite su nodi separati. La gestione del modello rimane esterna al task: il modello viene preparato prima della campagna e può essere eliminato dal job di cleanup solo dopo che tutte le wave lo hanno rilasciato.
