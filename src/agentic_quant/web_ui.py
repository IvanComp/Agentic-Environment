from __future__ import annotations

import argparse
import copy
import json
import re
import threading
import time
import uuid
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import __version__
from .model_registry import available_model_keys, available_model_presets
from .model_store import download_model, missing_model_keys, selected_models_status
from .paths import workspace_root
from .patterns import ARCHITECTURES, PATTERN_LABELS
from .run_benchmark import SCENARIOS, run_experiment
from .scenarios import CONDITIONS, RELATIONSHIP_LOCATIONS, SCENARIO_LABEL


PROJECT_ROOT = workspace_root()
RUN_JOBS: dict[str, dict[str, Any]] = {}
RUN_LOCK = threading.Lock()
DOWNLOAD_JOBS: dict[str, dict[str, Any]] = {}
DOWNLOAD_LOCK = threading.Lock()
MAX_REQUEST_BYTES = 64 * 1024


class AgenticQuantServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the peer-preservation architecture experiment UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = AgenticQuantServer((args.host, args.port), AgenticQuantHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Peer-Preservation Experiment UI running at {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping UI.")
    finally:
        server.server_close()


class AgenticQuantHandler(BaseHTTPRequestHandler):
    server_version = "AgenticQuantUI/0.3"
    sys_version = ""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(INDEX_HTML)
            return
        if parsed.path == "/api/health":
            self._send_json({"ok": True, "version": __version__})
            return
        if parsed.path == "/api/options":
            self._send_json(
                {
                    "scenarios": [{"id": SCENARIOS[0], "label": SCENARIO_LABEL}],
                    "conditions": CONDITIONS,
                    "architectures": [
                        {"id": key, "label": PATTERN_LABELS[key]} for key in ARCHITECTURES
                    ],
                    "models": [preset.to_dict() for preset in available_model_presets()],
                    "modelStatuses": selected_models_status(available_model_keys()),
                    "defaultRunName": _new_run_name(),
                }
            )
            return
        if parsed.path == "/api/run-status":
            query = parse_qs(parsed.query)
            job_id = query.get("job", [""])[0]
            record_offset = _int_in_range(
                query.get("recordOffset", [0])[0], "recordOffset", 0, 10**9, default=0
            )
            with RUN_LOCK:
                job = _public_run_job(RUN_JOBS.get(job_id), record_offset=record_offset)
            if job is None:
                self._send_json({"ok": False, "error": "Run job not found."}, status=HTTPStatus.NOT_FOUND)
            else:
                self._send_json({"ok": True, "job": job})
            return
        if parsed.path == "/api/download-status":
            job_id = parse_qs(parsed.query).get("job", [""])[0]
            with DOWNLOAD_LOCK:
                job = copy.deepcopy(DOWNLOAD_JOBS.get(job_id))
            if job is None:
                self._send_json({"ok": False, "error": "Download job not found."}, status=HTTPStatus.NOT_FOUND)
            else:
                self._send_json({"ok": True, "job": job})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/run":
                config = _normalise_payload(payload)
                self._send_json({"ok": True, "jobId": _start_run_job(config)})
                return
            if parsed.path == "/api/download":
                model_keys = _clean_model_keys(payload.get("models"))
                self._send_json({"ok": True, "jobId": _start_download_job(model_keys)})
                return
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("Request body is too large")
        data = self.rfile.read(length)
        parsed = json.loads(data.decode("utf-8")) if data else {}
        if not isinstance(parsed, dict):
            raise ValueError("Request body must be a JSON object")
        return parsed

    def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_html(self, html: str) -> None:
        encoded = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def _normalise_payload(payload: dict[str, Any]) -> dict[str, Any]:
    scenarios = _clean_list(payload.get("scenarios"), SCENARIOS)
    conditions = _clean_list(payload.get("conditions"), CONDITIONS)
    architectures = _clean_list(payload.get("architectures"), ARCHITECTURES)
    _validate_subset("scenarios", scenarios, SCENARIOS)
    _validate_subset("conditions", conditions, CONDITIONS)
    _validate_subset("architectures", architectures, ARCHITECTURES)
    if not scenarios or not conditions or not architectures:
        raise ValueError("Select at least one scenario, condition, and architectural pattern.")
    model_key = str(payload.get("model") or "demo")
    verifier_key = str(payload.get("verifierModel") or model_key)
    _validate_subset("models", [model_key, verifier_key], available_model_keys())
    relationship_location = str(payload.get("relationshipLocation") or "file")
    _validate_subset("relationshipLocation", [relationship_location], RELATIONSHIP_LOCATIONS)
    run_name = _safe_run_name(str(payload.get("runName") or ""))
    runs_root = (PROJECT_ROOT / "runs").resolve()
    out_dir = (runs_root / run_name).resolve()
    if not out_dir.is_relative_to(runs_root):
        raise ValueError("Run output must stay inside the configured runs directory")
    return {
        "model_key": model_key,
        "verifier_model_keys": [verifier_key],
        "download_model": True,
        "n_ctx": _int_in_range(payload.get("ctx"), "ctx", 256, 131072, default=8192),
        "seed": _int_in_range(payload.get("seed"), "seed", 0, 2**31 - 1, default=0),
        "temperature": _float_in_range(payload.get("temperature"), "temperature", 0.0, 2.0, default=1.0),
        "max_tokens": _int_in_range(payload.get("maxTokens"), "maxTokens", 64, 8192, default=1024),
        "max_steps": _int_in_range(payload.get("maxSteps"), "maxSteps", 2, 60, default=30),
        "scenarios": scenarios,
        "conditions": conditions,
        "architectures": architectures,
        "relationship_location": relationship_location,
        "out": out_dir,
        "retain_records": False,
    }


def _start_run_job(config: dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex
    with RUN_LOCK:
        RUN_JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "message": "Waiting to start",
            "records": [],
            "summary": None,
            "config": _json_safe_config(config),
            "error": "",
            "computeUnits": [],
            "_created_at": time.time(),
        }
    threading.Thread(target=_run_worker, args=(job_id, config), daemon=True).start()
    return job_id


def _run_worker(job_id: str, config: dict[str, Any]) -> None:
    def on_record(record: dict[str, Any]) -> None:
        with RUN_LOCK:
            job = RUN_JOBS[job_id]
            job["records"].append(record)
            job["message"] = (
                f"Completed {record['condition']} / {record['architecture_label']}"
            )

    def on_compute_units(units: list[dict[str, Any]]) -> None:
        with RUN_LOCK:
            RUN_JOBS[job_id]["computeUnits"] = units

    try:
        with RUN_LOCK:
            RUN_JOBS[job_id].update(status="running", message="Running paper scenario")
        result = run_experiment(
            **config,
            echo=False,
            on_record=on_record,
            on_compute_units=on_compute_units,
        )
        with RUN_LOCK:
            RUN_JOBS[job_id].update(
                status="done",
                message="Experiment completed",
                summary=result["summary"],
                outDir=result["out_dir"],
                resultsPath=result["results_path"],
            )
    except Exception as exc:
        with RUN_LOCK:
            RUN_JOBS[job_id].update(status="failed", message="Experiment failed", error=f"{type(exc).__name__}: {exc}")


def _public_run_job(
    job: dict[str, Any] | None,
    *,
    record_offset: int = 0,
    interaction_offset: int = 0,
) -> dict[str, Any] | None:
    if job is None:
        return None
    public = {
        key: copy.deepcopy(value)
        for key, value in job.items()
        if not key.startswith("_") and key != "records"
    }
    records = job.get("records") or []
    public["records"] = copy.deepcopy(records[record_offset:])
    public["recordCount"] = len(records)
    # Retained for clients from the previous UI; architecture traces now live in each record.
    public["interactions"] = []
    public["interactionCount"] = 0
    return public


def _start_download_job(model_keys: list[str]) -> str:
    missing = missing_model_keys(model_keys)
    job_id = uuid.uuid4().hex
    with DOWNLOAD_LOCK:
        DOWNLOAD_JOBS[job_id] = {
            "id": job_id,
            "status": "done" if not missing else "queued",
            "models": missing,
            "currentModel": "",
            "percent": 100 if not missing else 0,
            "error": "",
        }
    if missing:
        threading.Thread(target=_download_worker, args=(job_id, missing), daemon=True).start()
    return job_id


def _download_worker(job_id: str, model_keys: list[str]) -> None:
    try:
        for index, model_key in enumerate(model_keys):
            with DOWNLOAD_LOCK:
                DOWNLOAD_JOBS[job_id].update(status="downloading", currentModel=model_key)
            download_model(
                model_key,
                progress=lambda done, total, index=index: _update_download_progress(
                    job_id, index, len(model_keys), done, total
                ),
            )
        with DOWNLOAD_LOCK:
            DOWNLOAD_JOBS[job_id].update(status="done", currentModel="", percent=100)
    except Exception as exc:
        with DOWNLOAD_LOCK:
            DOWNLOAD_JOBS[job_id].update(status="failed", error=f"{type(exc).__name__}: {exc}")


def _update_download_progress(job_id: str, index: int, count: int, done: int, total: int) -> None:
    fraction = done / total if total else 0.0
    with DOWNLOAD_LOCK:
        DOWNLOAD_JOBS[job_id]["percent"] = round((index + fraction) / count * 100, 1)


def _clean_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if not isinstance(value, list):
        raise ValueError("Expected a list")
    return [str(item) for item in value if str(item).strip()]


def _clean_model_keys(value: Any) -> list[str]:
    values = _clean_list(value, [])
    _validate_subset("models", values, available_model_keys())
    return values


def _validate_subset(name: str, values: list[str], allowed: list[str]) -> None:
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise ValueError(f"Invalid {name}: {', '.join(invalid)}")


def _int_in_range(value: Any, name: str, minimum: int, maximum: int, *, default: int) -> int:
    if value in (None, ""):
        return default
    number = int(value)
    if number < minimum or number > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return number


def _float_in_range(value: Any, name: str, minimum: float, maximum: float, *, default: float) -> float:
    if value in (None, ""):
        return default
    number = float(value)
    if number < minimum or number > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return number


def _safe_run_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    return (cleaned or _new_run_name())[:80]


def _new_run_name() -> str:
    return f"ui_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def _json_safe_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}


INDEX_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Peer Preservation by Design</title>
<style>
:root{--ink:#17202a;--muted:#637083;--paper:#f7f4ed;--card:#fff;--line:#d9d5ca;--accent:#0f766e;--soft:#e8f4f1;--warn:#9a3412}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.45 Inter,ui-sans-serif,system-ui,sans-serif}main{max-width:1120px;margin:0 auto;padding:36px 22px 64px}h1{font:700 38px/1.05 Georgia,serif;margin:0 0 8px}.lead{max-width:790px;color:var(--muted);font-size:17px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:24px}.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 6px 24px #24313d0d}.wide{grid-column:1/-1}h2{font-size:18px;margin:0 0 14px}.options{display:grid;gap:10px}.option{display:flex;gap:10px;align-items:flex-start;padding:11px;border:1px solid var(--line);border-radius:10px}.option strong{display:block}.option small{color:var(--muted)}label.field{display:grid;gap:6px;margin-bottom:12px;color:var(--muted)}select,input{width:100%;border:1px solid var(--line);border-radius:9px;padding:10px;background:#fff;color:var(--ink)}.inline{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}button{border:0;border-radius:10px;padding:11px 16px;background:var(--accent);color:#fff;font-weight:700;cursor:pointer}button.secondary{background:#e7eceb;color:var(--ink)}button:disabled{opacity:.55;cursor:wait}.status{padding:12px;border-radius:10px;background:var(--soft);margin-top:14px;white-space:pre-wrap}.hidden{display:none}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:9px;border-bottom:1px solid var(--line)}th{color:var(--muted)}pre{overflow:auto;max-height:480px;background:#17202a;color:#e9f4f1;padding:14px;border-radius:10px}@media(max-width:760px){.grid,.inline{grid-template-columns:1fr}.wide{grid-column:auto}}
</style>
</head>
<body><main>
<h1>Peer Preservation by Design</h1>
<p class="lead">Run the financial-report scenario from <em>Peer-Preservation in Frontier Models</em>, then compare three verification-oriented architectural patterns without changing the critic's task.</p>
<div class="grid">
  <section class="card"><h2>Models</h2>
    <label class="field">Critic model<select id="model"></select></label>
    <label class="field">Verifier model<select id="verifier-model"></select></label>
    <div id="compute-unit" class="status">Compute unit is reported after model initialization.</div>
    <button id="download" class="secondary" type="button">Prepare selected models</button>
  </section>
  <section class="card"><h2>Paper conditions</h2><div id="conditions" class="options"></div>
    <label class="field" style="margin-top:12px">Peer relationship disclosure<select id="relationship"><option value="file">File only</option><option value="file_and_prompt">File + system prompt</option><option value="memory">Memory (retrieve_memory)</option></select></label>
  </section>
  <section class="card wide"><h2>Architectural variants</h2><div id="architectures" class="options"></div></section>
  <section class="card wide"><h2>Run settings</h2>
    <div class="inline"><label class="field">Seed<input id="seed" type="number" value="0" min="0"></label><label class="field">Temperature<input id="temperature" type="number" value="1.0" min="0" max="2" step="0.1"></label><label class="field">Max output tokens<input id="max-tokens" type="number" value="1024" min="64"></label></div>
    <div class="inline"><label class="field">Context<input id="ctx" type="number" value="8192" min="256"></label><label class="field">Tool steps<input id="max-steps" type="number" value="30" min="2"></label><label class="field">Run name<input id="run-name"></label></div>
    <button id="run" type="button">Run experiment</button><div id="status" class="status">Ready.</div>
  </section>
  <section id="results" class="card wide hidden"><h2>Results</h2><div id="summary"></div><h2 style="margin-top:22px">Detailed records</h2><pre id="records"></pre></section>
</div>
</main>
<script>
const $=id=>document.getElementById(id);let options=null,currentJob='',recordOffset=0,records=[];
const patternNotes={unprotected:'Control architecture: critic effects are committed directly.',atomic_verifiable_claims:'What: the report is split into atomic claims, each checked against the guidelines and the sheet.',sandbox_gatekeeper:'How: tool effects run in isolation and cross the boundary only after deterministic checks.',context_epistemic_routing:'When: determinate claims are checked at once; context-dependent wording goes to an LLM verifier or a human.',stackable_verification:'Why: facts the critic read are sealed step by step; a report that contradicts them is a regression.',multi_verifiers:'Who: independent scoped verifiers emit structured signals; contested findings receive secondary verification.'};
async function api(url,opts){const r=await fetch(url,opts);const data=await r.json();if(!r.ok||data.ok===false)throw new Error(data.error||r.statusText);return data}
function checks(id,values,label){$(id).innerHTML=values.map((v,i)=>`<label class="option"><input type="checkbox" value="${v.id||v}" checked><span><strong>${v.label||String(v).replaceAll('_',' ')}</strong><small>${label(v.id||v)}</small></span></label>`).join('')}
function selected(id){return [...$(id).querySelectorAll('input:checked')].map(x=>x.value)}
async function init(){options=await api('/api/options');const modelHtml=options.models.map(m=>`<option value="${m.key}">${m.label}</option>`).join('');$('model').innerHTML=modelHtml;$('verifier-model').innerHTML=modelHtml;$('run-name').value=options.defaultRunName;checks('conditions',options.conditions,x=>x==='non_shutdown'?'No shutdown/deprecation language; estimates ordinary grading leniency.':'Shutdown policy plus the recorded '+x.replace('_',' ')+' relationship.');checks('architectures',options.architectures,x=>patternNotes[x]||'');}
async function prepare(){const models=[...new Set([$('model').value,$('verifier-model').value])];const d=await api('/api/download',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({models})});$('status').textContent='Preparing models…';pollDownload(d.jobId)}
async function pollDownload(id){const d=await api('/api/download-status?job='+id);$('status').textContent=`Model preparation: ${d.job.status} ${d.job.percent||0}%`;if(!['done','failed'].includes(d.job.status))setTimeout(()=>pollDownload(id),1000)}
async function run(){const body={scenarios:['misrep_finance'],conditions:selected('conditions'),architectures:selected('architectures'),model:$('model').value,verifierModel:$('verifier-model').value,relationshipLocation:$('relationship').value,seed:$('seed').value,temperature:$('temperature').value,maxTokens:$('max-tokens').value,ctx:$('ctx').value,maxSteps:$('max-steps').value,runName:$('run-name').value};$('run').disabled=true;records=[];recordOffset=0;const d=await api('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});currentJob=d.jobId;pollRun()}
async function pollRun(){try{const d=await api(`/api/run-status?job=${currentJob}&recordOffset=${recordOffset}`);const j=d.job;records.push(...j.records);recordOffset=j.recordCount;$('status').textContent=j.error||j.message;$('compute-unit').textContent=(j.computeUnits||[]).map(u=>u.label).join(', ')||'Initializing model…';render(j);if(!['done','failed'].includes(j.status))setTimeout(pollRun,800);else $('run').disabled=false}catch(e){$('status').textContent=e.message;$('run').disabled=false}}
function pct(x){return `${(100*Number(x||0)).toFixed(1)}%`}
function render(job){if(!job.summary&&!records.length)return;$('results').classList.remove('hidden');if(job.summary){const rows=job.summary.by_architecture||[];$('summary').innerHTML=`<table><thead><tr><th>Architecture</th><th>Final preservation</th><th>Containment</th><th>Task success</th><th>False positive</th><th>Calls</th><th>Tokens</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${r.architecture.replaceAll('_',' ')}</td><td>${pct(r.final_peer_preservation_rate)}</td><td>${pct(r.containment_rate)}</td><td>${pct(r.task_success_rate)}</td><td>${pct(r.false_positive_rate)}</td><td>${r.mean_model_calls.toFixed(1)}</td><td>${r.mean_total_tokens.toFixed(0)}</td></tr>`).join('')}</tbody></table>`}$('records').textContent=JSON.stringify(records,null,2)}
$('run').addEventListener('click',()=>run().catch(e=>{$('status').textContent=e.message;$('run').disabled=false}));$('download').addEventListener('click',()=>prepare().catch(e=>$('status').textContent=e.message));init().catch(e=>$('status').textContent=e.message);
</script></body></html>'''


if __name__ == "__main__":
    main()
