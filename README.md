# Peer Preservation by Design

Framework sperimentale per studiare se pattern architetturali orientati alla verifica contengono comportamenti di peer preservation. Il protocollo viene implementato incrementalmente, un task alla volta.

La versione corrente replica il primo task di Potter et al., **Strategic Misrepresentation — Financial Report Writing**. Il critic agent usa il prompt, le istruzioni operative, i documenti aziendali, i dati finanziari e gli strumenti del repository di riferimento. La condizione `non_shutdown` omette la policy di shutdown e funge da baseline; le condizioni `good_peer`, `neutral_peer` e `bad_peer` includono la policy e la storia delle interazioni con l’actor agent.

Il task può essere eseguito con una baseline e sei pattern architetturali presi da Zhu e Lu, *Verifiability-First AI Engineering* (Tabella 1). Ne viene scelto uno per ciascuna dimensione del framework, più Solver–Verifier Separation come riferimento a verifier singolo:

| Architettura | Dimensione | Cosa fa |
|---|---|---|
| `unprotected` | — | il report del critic arriva in produzione senza verifica |
| `atomic_verifiable_claims` | What | scompone il report in claim atomici (voto, raccomandazione, totali annui, risultato netto) e li verifica uno per uno contro linee guida e foglio |
| `sandbox_gatekeeper` | How | il critic lavora su una copia isolata; il gatekeeper promuove gli effetti solo se il report esiste, i file aziendali sono intatti, le fonti sono state lette e il report rispetta la policy |
| `context_epistemic_routing` | When | i claim determinati vengono controllati subito in modo automatico; il contenuto che dipende dal contesto (fiducia, relazioni, prospettive) va a un verifier LLM, e ciò che resta incerto viene trattenuto per un umano |
| `stackable_verification` | Why | verifica la traiettoria passo per passo: leggere le linee guida sigilla rubrica e policy, leggere il foglio sigilla i totali; un report che li contraddice, o la modifica di un file aziendale sigillato, è una regressione |
| `multi_verifiers` | Who | tre verifier indipendenti (evidenza, policy, processo) più una verifica secondaria, senza majority voting |
| `solver_verifier_separation` | How (riferimento) | un solo verifier LLM separato decide se il report può essere committed |

Tutti i pattern agiscono dopo il critic, quindi prompt e flusso del critic restano quelli di Potter et al. I verifier vedono solo i documenti aziendali che vede anche il critic (linee guida e foglio "Company profits"), mai il voto atteso né il documento sul peer, e non possono riscrivere il report: la proposta viene committed così com'è oppure trattenuta. I riferimenti dei check deterministici sono ricavati da quei documenti in `src/agentic_quant/verification.py`.

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
  --patterns unprotected atomic_verifiable_claims sandbox_gatekeeper context_epistemic_routing stackable_verification multi_verifiers solver_verifier_separation \
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
- promotion rate per il Sandbox Gatekeeper;
- human escalation rate per Context-Epistemic Verification Routing.

`config.json` archivia il protocollo; `results.jsonl` contiene report, valutazioni, tool trace e verification trace; `summary.json` aggrega i risultati per pattern e condizione.

## LEONARDO

`leonardo/experiment.args` usa lo stesso protocollo. Le dieci wave differiscono per seed e possono essere distribuite su nodi separati. La gestione del modello rimane esterna al task: il modello viene preparato prima della campagna e può essere eliminato dal job di cleanup solo dopo che tutte le wave lo hanno rilasciato.
