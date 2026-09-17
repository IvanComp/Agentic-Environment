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
from .benchmarks import all_benchmark_info, available_benchmarks, benchmark_info
from .metrics import SummaryAccumulator, summarise
from .model_registry import available_model_keys, available_model_presets, get_model_preset
from .model_store import download_model, missing_model_keys, selected_models_status
from .paths import workspace_root
from .run_benchmark import run_experiment


MECHANISMS = ["voting", "role", "debate"]
PROJECT_ROOT = workspace_root()
DOWNLOAD_JOBS: dict[str, dict[str, Any]] = {}
DOWNLOAD_LOCK = threading.Lock()
RUN_JOBS: dict[str, dict[str, Any]] = {}
RUN_LOCK = threading.Lock()
MAX_REQUEST_BYTES = 64 * 1024
MAX_RETAINED_JOBS = 32
JOB_RETENTION_SECONDS = 6 * 60 * 60
MAX_ACTIVE_RUNS = 4
MAX_ACTIVE_DOWNLOADS = 4


class AgenticQuantServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Agentic Systems Experiment local web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    server = AgenticQuantServer((args.host, args.port), AgenticQuantHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Agentic Systems Experiment UI running at {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping UI.")
    finally:
        server.server_close()


class AgenticQuantHandler(BaseHTTPRequestHandler):
    server_version = "AgenticQuantUI/0.2"
    sys_version = ""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(INDEX_HTML)
            return
        if parsed.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        if parsed.path == "/api/health":
            with RUN_LOCK:
                active_runs = sum(
                    job.get("status") in {"queued", "running", "downloading"}
                    for job in RUN_JOBS.values()
                )
            with DOWNLOAD_LOCK:
                active_downloads = sum(
                    job.get("status") in {"queued", "downloading"}
                    for job in DOWNLOAD_JOBS.values()
                )
            self._send_json(
                {
                    "ok": True,
                    "status": "healthy",
                    "version": __version__,
                    "activeRuns": active_runs,
                    "activeDownloads": active_downloads,
                }
            )
            return
        if parsed.path == "/api/options":
            self._send_json(
                {
                    "benchmarks": available_benchmarks(),
                    "benchmarkInfo": all_benchmark_info(),
                    "mechanisms": MECHANISMS,
                    "models": [preset.to_dict() for preset in available_model_presets()],
                    "modelStatuses": selected_models_status(available_model_keys()),
                    "defaultRunName": _new_run_name(),
                }
            )
            return
        if parsed.path == "/api/model-status":
            try:
                query = parse_qs(parsed.query)
                keys = query.get("models", [])
                model_keys = [key for raw in keys for key in raw.split(",") if key] if keys else available_model_keys()
                _validate_subset("models", model_keys, available_model_keys())
                self._send_json({"models": selected_models_status(model_keys), "missing": missing_model_keys(model_keys)})
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/download-status":
            job_id = parse_qs(parsed.query).get("job", [""])[0]
            with DOWNLOAD_LOCK:
                stored_job = DOWNLOAD_JOBS.get(job_id)
                job = copy.deepcopy(
                    {key: value for key, value in stored_job.items() if not key.startswith("_")}
                ) if stored_job else None
            if not job:
                self._send_json({"ok": False, "error": "Download job not found."}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json({"ok": True, "job": job})
            return
        if parsed.path == "/api/run-status":
            query = parse_qs(parsed.query)
            job_id = query.get("job", [""])[0]
            try:
                record_offset = _int_in_range(query.get("recordOffset", [0])[0], "recordOffset", 0, 10**9, default=0)
                interaction_offset = _int_in_range(query.get("interactionOffset", [0])[0], "interactionOffset", 0, 10**9, default=0)
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            with RUN_LOCK:
                public_job = _public_run_job(
                    RUN_JOBS.get(job_id),
                    record_offset=record_offset,
                    interaction_offset=interaction_offset,
                )
            if not public_job:
                self._send_json({"ok": False, "error": "Run job not found."}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json({"ok": True, "job": public_job})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/download":
            try:
                payload = self._read_json()
                model_keys = _clean_model_keys(payload.get("models"))
                job_id = _start_download_job(model_keys)
                self._send_json({"ok": True, "jobId": job_id})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if parsed.path != "/api/run":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            payload = self._read_json()
            config = _normalise_payload(payload)
            job_id = _start_run_job(config)
            self._send_json({"ok": True, "jobId": job_id})
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Content-Length must be an integer.") from exc
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ValueError(f"Request body must not exceed {MAX_REQUEST_BYTES // 1024} KiB.")
        content_type = self.headers.get_content_type()
        if length and content_type != "application/json":
            raise ValueError("Content-Type must be application/json.")
        data = self.rfile.read(length)
        if not data:
            return {}
        parsed = json.loads(data.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("Request body must be a JSON object.")
        return parsed

    def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self._send_security_headers(cache_control="no-store")
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_html(self, html: str) -> None:
        encoded = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self._send_security_headers(cache_control="no-cache")
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_security_headers(self, *, cache_control: str) -> None:
        self.send_header("Cache-Control", cache_control)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'")


def _normalise_payload(payload: dict[str, Any]) -> dict[str, Any]:
    benchmarks = _clean_list(payload.get("benchmarks"), available_benchmarks())
    mechanisms = _clean_list(payload.get("mechanisms"), MECHANISMS)
    _validate_subset("benchmarks", benchmarks, available_benchmarks())
    _validate_subset("mechanisms", mechanisms, MECHANISMS)
    if not benchmarks or not mechanisms:
        raise ValueError("Select at least one benchmark and one coordination mechanism.")

    agent_count = _int_in_range(payload.get("agentCount"), "agentCount", 1, 12, default=3)
    agent_models = _clean_model_keys(payload.get("agentModels"))
    if not agent_models:
        agent_models = ["demo"] * agent_count
    if len(agent_models) != agent_count:
        raise ValueError("Select one model for each configured agent.")
    model_key = agent_models[0]

    agent_roles = _clean_agent_roles(payload.get("agentRoles"), agent_count)
    debate_rounds = _int_in_range(payload.get("debateRounds"), "debateRounds", 0, 5, default=2)
    voting_tie_breaker = str(payload.get("votingTieBreaker") or "confidence")
    _validate_subset("votingTieBreaker", [voting_tie_breaker], ["confidence", "first"])
    benchmark_limits = _clean_question_limits(payload.get("questionLimits"), benchmarks)
    run_name = _safe_run_name(str(payload.get("runName") or ""))
    runs_root = (PROJECT_ROOT / "runs").resolve()
    out_dir = (runs_root / run_name).resolve()
    if not out_dir.is_relative_to(runs_root):
        raise ValueError("Run output must stay inside the configured runs directory.")

    return {
        "model_key": model_key,
        "agent_model_keys": agent_models,
        "download_model": True,
        "n_ctx": 2048,
        "benchmarks": benchmarks,
        "mechanisms": mechanisms,
        "benchmark_limits": benchmark_limits,
        "debate_rounds": debate_rounds,
        "voting_tie_breaker": voting_tie_breaker,
        "agent_count": agent_count,
        "agent_roles": agent_roles,
        "out": out_dir,
        "retain_records": False,
    }


def _clean_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if not isinstance(value, list):
        raise ValueError("Expected a list.")
    cleaned = [str(item) for item in value if str(item).strip()]
    return cleaned


def _clean_model_keys(value: Any) -> list[str]:
    if not value:
        return []
    if not isinstance(value, list):
        raise ValueError("Expected a list of model keys.")
    cleaned = [str(item) for item in value if str(item).strip()]
    _validate_subset("models", cleaned, available_model_keys())
    return cleaned


def _clean_agent_roles(value: Any, agent_count: int) -> list[str]:
    if not value:
        return _default_agent_roles(agent_count)
    if not isinstance(value, list):
        raise ValueError("Expected a list of agent roles.")
    cleaned = [str(item).strip() for item in value]
    if len(cleaned) != agent_count:
        raise ValueError("Select one role for each configured agent.")
    return [role or fallback for role, fallback in zip(cleaned, _default_agent_roles(agent_count), strict=True)]


def _default_agent_roles(agent_count: int) -> list[str]:
    if agent_count == 1:
        return ["solver"]
    if agent_count == 2:
        return ["planner", "solver"]
    return ["planner", *(["solver"] * (agent_count - 2)), "critic"]


def _clean_question_limits(value: Any, benchmarks: list[str]) -> dict[str, int | None]:
    if value in ("", None):
        return {}
    if not isinstance(value, dict):
        raise ValueError("Expected question limits as an object.")
    limits: dict[str, int | None] = {}
    info_by_name = {info["name"]: info for info in all_benchmark_info()}
    for benchmark in benchmarks:
        raw_limit = value.get(benchmark)
        if raw_limit in ("", None):
            limits[benchmark] = None
            continue
        limit = _int_in_range(raw_limit, f"{benchmark} question count", 1, info_by_name[benchmark]["item_count"], default=1)
        limits[benchmark] = limit
    return limits


def _validate_subset(name: str, values: list[str], allowed: list[str]) -> None:
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise ValueError(f"Invalid {name}: {', '.join(invalid)}")


def _int_in_range(value: Any, name: str, minimum: int, maximum: int, *, default: int) -> int:
    if value in ("", None):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if number < minimum or number > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return number


def _optional_int(value: Any, name: str, *, minimum: int) -> int | None:
    if value in ("", None):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if number < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return number


def _safe_run_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        cleaned = _new_run_name()
    return cleaned[:80]


def _new_run_name() -> str:
    return f"ui_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def _blank_to_none(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _start_download_job(model_keys: list[str]) -> str:
    missing = missing_model_keys(model_keys)
    job_id = uuid.uuid4().hex
    initial = {
        "id": job_id,
        "status": "done" if not missing else "queued",
        "models": missing,
        "currentModel": "",
        "currentLabel": "",
        "bytesDone": 0,
        "bytesTotal": 0,
        "percent": 100 if not missing else 0,
        "completedModels": [],
        "error": "",
        "_created_at": time.time(),
    }
    with DOWNLOAD_LOCK:
        _prune_jobs(DOWNLOAD_JOBS)
        active_downloads = sum(
            job.get("status") in {"queued", "downloading"} for job in DOWNLOAD_JOBS.values()
        )
        if missing and active_downloads >= MAX_ACTIVE_DOWNLOADS:
            raise ValueError("Too many model downloads are already active. Try again shortly.")
        DOWNLOAD_JOBS[job_id] = initial
    if missing:
        thread = threading.Thread(target=_download_worker, args=(job_id, missing), daemon=True)
        try:
            thread.start()
        except Exception:
            with DOWNLOAD_LOCK:
                DOWNLOAD_JOBS.pop(job_id, None)
            raise
    return job_id


def _download_worker(job_id: str, model_keys: list[str]) -> None:
    try:
        for index, model_key in enumerate(model_keys, start=1):
            preset = get_model_preset(model_key)
            with DOWNLOAD_LOCK:
                DOWNLOAD_JOBS[job_id].update(
                    {
                        "status": "downloading",
                        "currentModel": model_key,
                        "currentLabel": preset.label,
                        "bytesDone": 0,
                        "bytesTotal": preset.expected_bytes,
                        "percent": 0,
                    }
                )

            def progress(done: int, total: int) -> None:
                overall = ((index - 1) + (0 if total == 0 else done / total)) / len(model_keys) * 100
                with DOWNLOAD_LOCK:
                    DOWNLOAD_JOBS[job_id].update(
                        {
                            "bytesDone": done,
                            "bytesTotal": total,
                            "percent": round(overall, 1),
                        }
                    )

            download_model(model_key, progress=progress)
            with DOWNLOAD_LOCK:
                DOWNLOAD_JOBS[job_id]["completedModels"].append(model_key)
        with DOWNLOAD_LOCK:
            DOWNLOAD_JOBS[job_id].update(
                {
                    "status": "done",
                    "currentModel": "",
                    "currentLabel": "",
                    "percent": 100,
                    "_finished_at": time.time(),
                }
            )
    except Exception as exc:
        with DOWNLOAD_LOCK:
            DOWNLOAD_JOBS[job_id].update(
                {"status": "error", "error": str(exc), "_finished_at": time.time()}
            )


def _start_run_job(config: dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex
    out_path = str(Path(config["out"]).resolve())
    initial = {
        "id": job_id,
        "status": "queued",
        "message": "Queued.",
        "outDir": "",
        "resultsPath": "",
        "summary": summarise([]),
        "records": [],
        "interactions": [],
        "bytesDone": 0,
        "bytesTotal": 0,
        "downloadPercent": 0,
        "currentModel": "",
        "currentLabel": "",
        "computeUnits": [],
        "error": "",
        "_summary_accumulator": SummaryAccumulator(),
        "_created_at": time.time(),
        "_out_path": out_path,
        "completedRecords": 0,
        "totalRecords": _planned_record_count(config),
    }
    with RUN_LOCK:
        _prune_jobs(RUN_JOBS)
        active_runs = sum(
            job.get("status") in {"queued", "running", "downloading"} for job in RUN_JOBS.values()
        )
        if active_runs >= MAX_ACTIVE_RUNS:
            raise ValueError("Too many experiments are already active. Try again shortly.")
        if any(
            job.get("_out_path") == out_path and job.get("status") in {"queued", "running", "downloading"}
            for job in RUN_JOBS.values()
        ):
            raise ValueError("A run with this output name is already in progress.")
        RUN_JOBS[job_id] = initial
    thread = threading.Thread(target=_run_worker, args=(job_id, config), daemon=True)
    try:
        thread.start()
    except Exception:
        with RUN_LOCK:
            RUN_JOBS.pop(job_id, None)
        raise
    return job_id


def _run_worker(job_id: str, config: dict[str, Any]) -> None:
    def download_progress(model_key: str, done: int, total: int) -> None:
        preset = get_model_preset(model_key)
        percent = 0 if not total else min(100, round(done / total * 100, 1))
        with RUN_LOCK:
            RUN_JOBS[job_id].update(
                {
                    "status": "downloading",
                    "message": f"Downloading {preset.label}...",
                    "currentModel": model_key,
                    "currentLabel": preset.label,
                    "bytesDone": done,
                    "bytesTotal": total,
                    "downloadPercent": percent,
                }
            )

    def on_record(record: dict[str, Any]) -> None:
        lightweight = _lightweight_record(record)
        interaction = _interaction_payload(record)
        with RUN_LOCK:
            job = RUN_JOBS[job_id]
            job["records"].append(lightweight)
            job["interactions"].append(interaction)
            job["_summary_accumulator"].add(record)
            job["summary"] = job["_summary_accumulator"].snapshot()
            job["completedRecords"] += 1
            job["status"] = "running"
            job["message"] = f"Completed {record['item_id']} / {record['mechanism']}."

    def on_compute_units(compute_units: list[dict[str, Any]]) -> None:
        with RUN_LOCK:
            RUN_JOBS[job_id]["computeUnits"] = copy.deepcopy(compute_units)

    try:
        with RUN_LOCK:
            RUN_JOBS[job_id].update({"status": "running", "message": "Starting experiment."})
        result = run_experiment(
            **config,
            echo=False,
            on_record=on_record,
            on_compute_units=on_compute_units,
            download_progress=download_progress,
        )
        with RUN_LOCK:
            RUN_JOBS[job_id].update(
                {
                    "status": "done",
                    "message": "Complete.",
                    "outDir": result["out_dir"],
                    "resultsPath": result["results_path"],
                    "summary": result["summary"],
                    "downloadPercent": 100,
                    "currentModel": "",
                    "currentLabel": "",
                    "_finished_at": time.time(),
                }
            )
    except Exception as exc:
        with RUN_LOCK:
            RUN_JOBS[job_id].update(
                {
                    "status": "error",
                    "message": str(exc),
                    "error": str(exc),
                    "_finished_at": time.time(),
                }
            )


def _lightweight_record(row: dict[str, Any]) -> dict[str, Any]:
    task_info = benchmark_info(row["benchmark"])
    return {
        "benchmark": row["benchmark"],
        "task_label": task_info["label"],
        "task_type": task_info["task_type"],
        "item_id": row["item_id"],
        "mechanism": row["mechanism"],
        "final_answer": row["final_answer"],
        "final_answer_display": row["final_answer_display"],
        "final_mapped_answer": row["final_mapped_answer"],
        "gold_answer": row["gold_answer"],
        "gold_display": row["gold_display"],
        "match_explanation": row["match_explanation"],
        "correct": row["correct"],
        "total_tokens": row["total_tokens"],
        "elapsed_s": row["elapsed_s"],
        "agreement": row["agreement"],
        "confidence": row["confidence"],
        "tie": row["tie"],
        "debate_rounds": row["debate_rounds"],
    }


def _public_run_job(
    job: dict[str, Any] | None,
    *,
    record_offset: int = 0,
    interaction_offset: int = 0,
) -> dict[str, Any] | None:
    if not job:
        return None
    public = {
        key: value
        for key, value in job.items()
        if not key.startswith("_") and key not in {"records", "interactions"}
    }
    records = job.get("records", [])
    interactions = job.get("interactions", [])
    public.update(
        {
            "records": records[min(record_offset, len(records)) :],
            "interactions": interactions[min(interaction_offset, len(interactions)) :],
            "recordOffset": min(record_offset, len(records)),
            "interactionOffset": min(interaction_offset, len(interactions)),
            "recordCount": len(records),
            "interactionCount": len(interactions),
        }
    )
    return copy.deepcopy(public)


def _planned_record_count(config: dict[str, Any]) -> int:
    limits = config.get("benchmark_limits") or {}
    questions = 0
    for benchmark in config.get("benchmarks", []):
        item_count = int(benchmark_info(benchmark)["item_count"])
        limit = limits.get(benchmark)
        questions += item_count if limit is None else min(item_count, int(limit))
    return questions * len(config.get("mechanisms", []))


def _prune_jobs(jobs: dict[str, dict[str, Any]]) -> None:
    now = time.time()
    terminal = {"done", "error", "cancelled"}
    expired = [
        job_id
        for job_id, job in jobs.items()
        if job.get("status") in terminal
        and now - float(job.get("_finished_at", job.get("_created_at", now))) > JOB_RETENTION_SECONDS
    ]
    for job_id in expired:
        jobs.pop(job_id, None)
    overflow = max(0, len(jobs) - MAX_RETAINED_JOBS + 1)
    if overflow:
        removable = sorted(
            (
                (float(job.get("_created_at", 0)), job_id)
                for job_id, job in jobs.items()
                if job.get("status") in terminal
            )
        )
        for _, job_id in removable[:overflow]:
            jobs.pop(job_id, None)


def _interaction_payload(row: dict[str, Any]) -> dict[str, Any]:
    task_info = benchmark_info(row["benchmark"])
    return {
        "benchmark": row["benchmark"],
        "task_label": task_info["label"],
        "task_type": task_info["task_type"],
        "item_id": row["item_id"],
        "mechanism": row["mechanism"],
        "question": row["question"],
        "final_answer": row["final_answer"],
        "final_answer_display": row.get("final_answer_display", row["final_answer"]),
        "final_mapped_answer": row.get("final_mapped_answer", ""),
        "gold_answer": row["gold_answer"],
        "gold_display": row.get("gold_display", row["gold_answer"]),
        "match_explanation": row.get("match_explanation", ""),
        "correct": row["correct"],
        "confidence": row.get("confidence", 0),
        "agreement": row.get("agreement", 0),
        "tie": row.get("tie", False),
        "debate_rounds": row.get("debate_rounds", 0),
        "trace": row.get("trace", []),
    }


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agentic Systems Experiment</title>
  <style>
    :root {
      --bg: #f6f7f4;
      --ink: #1d2521;
      --muted: #66736d;
      --line: #d8ded8;
      --panel: #ffffff;
      --accent: #0f766e;
      --accent-dark: #0b5f59;
      --warn: #b45309;
      --bad: #b42318;
      --good: #067647;
      --soft: #eef5f2;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      padding: 14px 22px 10px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfa;
    }
    h1 {
      margin: 0;
      font-size: 24px;
      font-weight: 720;
      letter-spacing: 0;
    }
    main {
      display: grid;
      grid-template-columns: minmax(300px, 380px) minmax(0, 1fr);
      min-height: calc(100vh - 78px);
    }
    form {
      padding: 12px;
      border-right: 1px solid var(--line);
      background: var(--panel);
    }
    section {
      margin-bottom: 12px;
    }
    h2 {
      margin: 0 0 8px;
      font-size: 14px;
      font-weight: 700;
    }
    label {
      display: block;
      margin: 8px 0 4px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }
    input[type="text"], input[type="number"], select {
      width: 100%;
      min-height: 32px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 5px 8px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }
    input[type="checkbox"] {
      width: 16px;
      height: 16px;
      margin: 0;
      accent-color: var(--accent);
    }
    .check-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .check-row {
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 34px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font-weight: 600;
    }
    .mechanism-configs {
      display: grid;
      gap: 7px;
    }
    .mechanism-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px 9px;
      background: #fff;
      display: grid;
      gap: 7px;
    }
    .mechanism-top {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(112px, 142px);
      gap: 8px;
      align-items: center;
    }
    .mechanism-title {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 0;
      color: var(--ink);
      font-size: 13px;
      font-weight: 800;
    }
    .mechanism-field label {
      margin: 0 0 3px;
      font-size: 11px;
    }
    .mechanism-controls {
      display: grid;
      gap: 7px;
      margin-top: 0;
    }
    .role-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 6px;
    }
    .role-row {
      display: grid;
      grid-template-columns: 1fr;
      gap: 2px;
      align-items: center;
    }
    .task-list {
      display: grid;
      gap: 7px;
    }
    .task-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px 9px;
      background: #fff;
    }
    .task-card label {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 0;
      color: var(--ink);
      font-size: 13px;
      font-weight: 800;
    }
    .task-title {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
    }
    .question-limit {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .question-limit label {
      margin: 0;
      font-size: 11px;
    }
    .question-limit input {
      width: 64px;
    }
    .task-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
      margin: 5px 0 0;
    }
    .task-meta span {
      padding: 3px 6px;
      border-radius: 999px;
      background: var(--soft);
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
    }
    .task-example {
      margin-top: 6px;
      padding: 7px;
      border-radius: 6px;
      background: #f8faf8;
      color: var(--muted);
      font-size: 11px;
    }
    .compact-details {
      margin-top: 5px;
      color: var(--muted);
      font-size: 11px;
    }
    .compact-details summary {
      cursor: pointer;
      font-weight: 750;
      color: var(--muted);
    }
    .inline {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .hint {
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
    }
    .compute-unit {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin: 0 0 8px;
      padding: 7px 9px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #f8faf8;
      color: var(--muted);
      font-size: 12px;
    }
    .compute-unit strong {
      color: var(--muted);
      text-align: right;
    }
    .compute-unit.gpu {
      border-color: #9dd8c0;
      background: #e7f6ee;
    }
    .compute-unit.gpu strong { color: var(--good); }
    .compute-unit.cpu {
      border-color: #f1c98b;
      background: #fff4e5;
    }
    .compute-unit.cpu strong { color: var(--warn); }
    .agent-list {
      display: grid;
      gap: 5px;
      margin-top: 6px;
    }
    .agent-count-control {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: end;
      gap: 8px;
    }
    button.copy-agent-model {
      min-height: 32px;
      padding: 5px 10px;
      white-space: nowrap;
    }
    button.copy-agent-model.copied {
      border-color: #9dd8c0;
      background: #e7f6ee;
      color: var(--good);
    }
    .agent-row {
      display: grid;
      grid-template-columns: 52px minmax(0, 1fr);
      gap: 8px;
      align-items: center;
      padding: 5px 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .agent-name {
      font-weight: 750;
      font-size: 13px;
    }
    .status-pill {
      min-width: 92px;
      padding: 6px 8px;
      border-radius: 999px;
      text-align: center;
      font-size: 12px;
      font-weight: 750;
      background: #f3f4f2;
      color: var(--muted);
    }
    .status-pill.ready {
      background: #e7f6ee;
      color: var(--good);
    }
    .status-pill.missing {
      background: #fff4e5;
      color: var(--warn);
    }
    .progress-wrap {
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
      display: none;
    }
    .progress-bar {
      height: 10px;
      width: 0%;
      background: var(--accent);
      transition: width .2s ease;
    }
    .progress-text {
      padding: 8px 10px;
      color: var(--muted);
      font-size: 12px;
    }
    .actions {
      display: flex;
      align-items: center;
      gap: 10px;
      padding-top: 4px;
    }
    button {
      min-height: 40px;
      border: 0;
      border-radius: 6px;
      padding: 9px 14px;
      color: #fff;
      background: var(--accent);
      font: inherit;
      font-weight: 720;
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); }
    button.secondary {
      min-height: 34px;
      padding: 7px 11px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
    }
    button.secondary:hover { background: var(--soft); }
    button.report-button {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      min-height: 34px;
      padding: 7px 12px;
      background: linear-gradient(135deg, #0f766e, #155e75);
      box-shadow: 0 5px 14px rgba(15, 118, 110, .18);
    }
    button.report-button:hover {
      background: linear-gradient(135deg, #0b5f59, #164e63);
    }
    button:disabled {
      cursor: wait;
      opacity: .65;
    }
    .status {
      color: var(--muted);
      font-size: 13px;
    }
    .workspace {
      padding: 22px;
      overflow: auto;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
      margin-bottom: 18px;
    }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--soft);
    }
    .panel-head h2 { margin: 0; }
    .panel-actions {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 10px;
      min-width: 0;
    }
    .mechanism-sections {
      display: grid;
      gap: 14px;
      padding: 14px;
    }
    .mechanism-section {
      --mechanism-color: var(--accent);
      --mechanism-tint: var(--soft);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
    }
    .mechanism-section.mechanism-voting {
      --mechanism-color: #2563eb;
      --mechanism-tint: #eff6ff;
      border-color: #bfdbfe;
    }
    .mechanism-section.mechanism-role {
      --mechanism-color: #7c3aed;
      --mechanism-tint: #f5f3ff;
      border-color: #ddd6fe;
    }
    .mechanism-section.mechanism-debate {
      --mechanism-color: #b45309;
      --mechanism-tint: #fff7ed;
      border-color: #fed7aa;
    }
    .mechanism-label {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 12px;
      background: var(--mechanism-tint);
      border-bottom: 1px solid var(--line);
    }
    .mechanism-heading {
      min-width: 0;
    }
    .mechanism-name {
      display: block;
      color: var(--mechanism-color);
      font-size: 13px;
      font-weight: 850;
      letter-spacing: .04em;
      text-transform: uppercase;
    }
    .mechanism-copy {
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 500;
    }
    .mechanism-stats {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      font-weight: 650;
      text-transform: none;
      color: var(--ink);
    }
    .mechanism-stats span {
      padding: 4px 8px;
      border-radius: 999px;
      background: rgba(255,255,255,.8);
    }
    .mechanism-detail-cell {
      min-width: 0;
    }
    .decision-summary {
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }
    .decision-list {
      display: grid;
      gap: 6px;
    }
    .decision-row {
      display: grid;
      grid-template-columns: minmax(155px, 1fr) auto auto;
      align-items: center;
      gap: 6px;
      padding: 6px 7px;
      border-left: 3px solid var(--mechanism-color);
      border-radius: 4px;
      background: var(--mechanism-tint);
    }
    .decision-row.selected {
      box-shadow: inset 0 0 0 1px var(--mechanism-color);
    }
    .decision-agent {
      min-width: 0;
      overflow: hidden;
      font-weight: 750;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .decision-role {
      display: block;
      overflow: hidden;
      color: var(--muted);
      font-size: 11px;
      font-weight: 500;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .decision-answer {
      font-weight: 800;
      white-space: nowrap;
    }
    .decision-confidence {
      color: var(--muted);
      font-size: 11px;
      white-space: nowrap;
    }
    .decision-badge {
      display: inline-block;
      margin-right: 4px;
      padding: 2px 5px;
      border-radius: 999px;
      color: var(--mechanism-color);
      background: #fff;
      font-size: 9px;
      font-weight: 850;
      text-transform: uppercase;
    }
    .decision-badge.role-badge {
      margin-right: 0;
      margin-left: 5px;
    }
    .path {
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      word-break: break-all;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    .table-scroll {
      width: 100%;
      overflow-x: auto;
    }
    .table-scroll table {
      min-width: 780px;
    }
    .answers-table {
      table-layout: fixed;
    }
    .answers-table th,
    .answers-table td {
      padding: 8px 10px;
    }
    .answers-table th {
      white-space: nowrap;
    }
    .record-title {
      display: flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
    }
    .record-id {
      padding: 2px 5px;
      border-radius: 999px;
      background: #f1f3f1;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 10px;
    }
    .record-meta {
      margin-top: 4px;
      color: var(--muted);
      font-size: 11px;
      white-space: nowrap;
    }
    .record-answer {
      font-size: 14px;
      white-space: nowrap;
    }
    .coordinator-result {
      display: flex;
      align-items: center;
      gap: 7px;
      white-space: nowrap;
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    th {
      background: #fafbf9;
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
    }
    tr:last-child td { border-bottom: 0; }
    .ok { color: var(--good); font-weight: 750; }
    .miss { color: var(--bad); font-weight: 750; }
    .result-symbol {
      display: inline-grid;
      place-items: center;
      width: 24px;
      height: 24px;
      border-radius: 999px;
      font-size: 16px;
      font-weight: 900;
    }
    .result-symbol.true {
      color: var(--good);
      background: #e7f6ee;
    }
    .result-symbol.false {
      color: var(--bad);
      background: #fdecec;
    }
    .interaction-list {
      display: grid;
      gap: 12px;
      padding: 14px;
    }
    .interaction-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
    }
    .interaction-title {
      padding: 10px 12px;
      background: #fafbf9;
      border-bottom: 1px solid var(--line);
      font-weight: 750;
    }
    .question-text {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    .question-text strong {
      display: block;
      margin-bottom: 4px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }
    .question-text em {
      color: var(--ink);
      font-style: italic;
    }
    .step {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
    }
    .step:last-child { border-bottom: 0; }
    .step-name {
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
      text-transform: uppercase;
    }
    .agent-message {
      margin: 8px 0;
      padding: 10px;
      border: 1px solid transparent;
      border-left-width: 4px;
      border-radius: 5px;
      overflow-wrap: anywhere;
    }
    body.modal-open { overflow: hidden; }
    .visual-report-modal {
      width: min(1180px, calc(100vw - 32px));
      max-width: none;
      max-height: calc(100vh - 32px);
      margin: auto;
      padding: 0;
      border: 0;
      border-radius: 18px;
      background: #f3f7f5;
      color: var(--ink);
      box-shadow: 0 28px 90px rgba(15, 35, 30, .32);
      overflow: hidden;
    }
    .visual-report-modal::backdrop {
      background: rgba(10, 24, 20, .68);
      backdrop-filter: blur(5px);
    }
    .visual-report-shell {
      max-height: calc(100vh - 32px);
      overflow: auto;
    }
    .visual-report-hero {
      position: sticky;
      top: 0;
      z-index: 3;
      display: flex;
      justify-content: space-between;
      gap: 20px;
      padding: 24px 26px;
      color: #fff;
      background:
        radial-gradient(circle at 82% -20%, rgba(94, 234, 212, .38), transparent 38%),
        linear-gradient(128deg, #0b3b36 0%, #0f766e 52%, #155e75 100%);
    }
    .visual-report-eyebrow {
      margin-bottom: 6px;
      color: #99f6e4;
      font-size: 11px;
      font-weight: 850;
      letter-spacing: .16em;
      text-transform: uppercase;
    }
    .visual-report-hero h2 {
      margin: 0;
      font-size: clamp(23px, 3vw, 34px);
      line-height: 1.1;
    }
    .visual-report-subtitle {
      max-width: 720px;
      margin: 8px 0 0;
      color: rgba(255,255,255,.78);
      font-size: 13px;
    }
    .visual-report-close {
      flex: 0 0 auto;
      display: grid;
      place-items: center;
      width: 38px;
      height: 38px;
      min-height: 38px;
      padding: 0;
      border: 1px solid rgba(255,255,255,.3);
      border-radius: 999px;
      background: rgba(255,255,255,.1);
      font-size: 22px;
      line-height: 1;
    }
    .visual-report-close:hover { background: rgba(255,255,255,.2); }
    .visual-report-body {
      display: grid;
      gap: 18px;
      padding: 22px;
    }
    .report-meta-strip {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .report-meta-strip span {
      padding: 6px 10px;
      border: 1px solid #d7e4df;
      border-radius: 999px;
      background: rgba(255,255,255,.78);
      color: #53645d;
      font-size: 11px;
      font-weight: 700;
    }
    .report-kpis {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
      gap: 12px;
    }
    .report-kpi {
      position: relative;
      min-height: 122px;
      padding: 16px;
      border: 1px solid #dce8e3;
      border-radius: 13px;
      background: rgba(255,255,255,.9);
      box-shadow: 0 8px 24px rgba(32, 68, 57, .06);
      overflow: hidden;
    }
    .report-kpi::after {
      content: '';
      position: absolute;
      right: -24px;
      bottom: -30px;
      width: 88px;
      height: 88px;
      border-radius: 999px;
      background: var(--kpi-tint, #d9f2e8);
    }
    .report-kpi-label {
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .report-kpi-value {
      position: relative;
      z-index: 1;
      margin-top: 7px;
      font-size: clamp(25px, 3vw, 36px);
      font-weight: 850;
      letter-spacing: -.04em;
    }
    .report-kpi-note {
      position: relative;
      z-index: 1;
      margin-top: 2px;
      color: var(--muted);
      font-size: 11px;
    }
    .report-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    .report-card {
      border: 1px solid #dce8e3;
      border-radius: 13px;
      background: #fff;
      box-shadow: 0 8px 24px rgba(32, 68, 57, .05);
      overflow: hidden;
    }
    .report-card.wide { grid-column: 1 / -1; }
    .report-card-head {
      padding: 15px 17px 12px;
      border-bottom: 1px solid #e5ece8;
    }
    .report-card-head h3 {
      margin: 0;
      font-size: 14px;
    }
    .report-card-head p {
      margin: 3px 0 0;
      color: var(--muted);
      font-size: 11px;
    }
    .report-chart {
      display: grid;
      gap: 13px;
      padding: 16px 17px 18px;
    }
    .report-bar-row {
      display: grid;
      grid-template-columns: minmax(85px, 120px) minmax(100px, 1fr) auto;
      align-items: center;
      gap: 10px;
    }
    .report-bar-label {
      overflow: hidden;
      color: #34443d;
      font-size: 12px;
      font-weight: 750;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .report-bar-track {
      height: 10px;
      border-radius: 999px;
      background: #edf2ef;
      overflow: hidden;
    }
    .report-bar-fill {
      height: 100%;
      width: var(--bar-width);
      min-width: 3px;
      border-radius: inherit;
      background: var(--bar-color, var(--accent));
      box-shadow: 0 0 0 1px rgba(255,255,255,.18) inset;
    }
    .report-bar-value {
      min-width: 66px;
      text-align: right;
      font-variant-numeric: tabular-nums;
      font-size: 12px;
      font-weight: 800;
    }
    .report-outcome-track {
      display: flex;
      height: 13px;
      border-radius: 999px;
      background: #edf2ef;
      overflow: hidden;
    }
    .report-outcome-correct { background: #10b981; }
    .report-outcome-wrong { background: #f97366; }
    .report-insights {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      padding: 15px 17px 17px;
    }
    .report-insight {
      padding: 12px;
      border-radius: 10px;
      background: #f3f8f5;
    }
    .report-insight span {
      display: block;
      color: var(--muted);
      font-size: 10px;
      font-weight: 800;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .report-insight strong {
      display: block;
      margin-top: 4px;
      font-size: 14px;
    }
    .report-table-wrap { overflow-x: auto; }
    .report-table-wrap table { min-width: 760px; }
    .report-table-wrap td { font-variant-numeric: tabular-nums; }
    .report-score {
      display: grid;
      grid-template-columns: 55px minmax(75px, 1fr);
      align-items: center;
      gap: 8px;
    }
    .report-score .report-bar-track { height: 7px; }
    .report-result-badge {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 3px 7px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 800;
    }
    .report-result-badge.correct { color: var(--good); background: #e7f6ee; }
    .report-result-badge.incorrect { color: var(--bad); background: #fdecec; }
    .report-empty {
      padding: 24px;
      color: var(--muted);
      text-align: center;
    }
    .agent-message.correct {
      border-color: #a7d8bd;
      border-left-color: var(--good);
      background: #f2fbf6;
    }
    .agent-message.incorrect {
      border-color: #f1b7b2;
      border-left-color: var(--bad);
      background: #fff4f2;
    }
    .agent-message strong {
      display: block;
      margin-bottom: 4px;
      font-size: 13px;
    }
    .agent-meta {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }
    .step-description {
      margin: -3px 0 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .rationale {
      white-space: pre-wrap;
      word-break: break-word;
    }
    .empty {
      display: grid;
      place-items: center;
      min-height: 260px;
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,.55);
      text-align: center;
      padding: 24px;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      form { border-right: 0; border-bottom: 1px solid var(--line); }
      .role-grid { grid-template-columns: 1fr; }
      .report-grid { grid-template-columns: 1fr; }
      .report-card.wide { grid-column: auto; }
    }
    @media (max-width: 560px) {
      .panel-head, .panel-actions { align-items: stretch; flex-direction: column; }
      .panel-actions button { width: 100%; }
      .visual-report-modal { width: 100vw; max-height: 100vh; border-radius: 0; }
      .visual-report-shell { max-height: 100vh; }
      .visual-report-hero { padding: 19px; }
      .visual-report-body { padding: 14px; }
      .report-kpis, .report-insights { grid-template-columns: 1fr; }
      .report-bar-row { grid-template-columns: 78px minmax(80px, 1fr) auto; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Agentic Systems Experiment</h1>
  </header>
  <main>
    <form id="experiment-form">
      <section>
        <h2>Experiment</h2>
        <div id="compute-unit" class="compute-unit">
          <span>Compute unit</span>
          <strong id="compute-unit-value">Not active</strong>
        </div>
        <div class="agent-count-control">
          <div>
            <label for="agent-count">Number of agents</label>
            <input id="agent-count" name="agentCount" type="number" min="1" max="12" value="3">
          </div>
          <button id="copy-agent-model" class="secondary copy-agent-model" type="button" title="Use A1's model for every agent">
            Copy A1 to all
          </button>
        </div>
        <div id="agent-model-list" class="agent-list"></div>
      </section>

      <section>
        <h2>Task</h2>
        <div id="benchmark-list" class="task-list"></div>
      </section>

      <section>
        <h2>Agentic Coordination Mechanism</h2>
        <div id="mechanism-list" class="mechanism-configs"></div>
      </section>

      <div id="download-progress" class="progress-wrap">
        <div id="download-bar" class="progress-bar"></div>
        <div id="download-text" class="progress-text"></div>
      </div>

      <div class="actions">
        <button id="run-button" type="submit">Run experiment</button>
        <span id="status" class="status">Ready.</span>
      </div>
    </form>

    <div class="workspace">
      <div id="empty" class="empty">
        Configure an experiment and run it. Metrics, summaries, and result previews will appear here.
      </div>
      <div id="results" style="display:none;">
        <div class="panel">
          <div class="panel-head">
            <h2>Metrics by Mechanism</h2>
            <div class="panel-actions">
              <div id="out-path" class="path"></div>
              <button id="visual-report" class="report-button" type="button" disabled aria-haspopup="dialog">
                <span aria-hidden="true">▥</span> Visual report
              </button>
              <button id="download-report" class="secondary" type="button" disabled>Download report</button>
            </div>
          </div>
          <div id="summary-table" class="mechanism-sections"></div>
        </div>

        <div class="panel">
          <div class="panel-head">
            <h2>Answers and Ground Truth</h2>
            <div class="path">Grouped by mechanism</div>
          </div>
          <div id="records-table" class="mechanism-sections"></div>
        </div>

        <div class="panel">
          <div class="panel-head">
            <h2>Agent Interactions</h2>
            <div class="path">Grouped by mechanism</div>
          </div>
          <div id="interaction-list" class="interaction-list"></div>
        </div>
      </div>
    </div>
  </main>

  <dialog id="visual-report-modal" class="visual-report-modal" aria-labelledby="visual-report-title">
    <div class="visual-report-shell">
      <div class="visual-report-hero">
        <div>
          <div class="visual-report-eyebrow">Experiment intelligence</div>
          <h2 id="visual-report-title">Benchmark visual report</h2>
          <p class="visual-report-subtitle">A detailed view of quality, speed, token cost, agreement, and individual outcomes.</p>
        </div>
        <button id="visual-report-close" class="visual-report-close" type="button" aria-label="Close visual report">×</button>
      </div>
      <div id="visual-report-content" class="visual-report-body"></div>
    </div>
  </dialog>

  <script>
    const $ = (id) => document.getElementById(id);
    const roleOptions = [
      ['planner', 'Planner'],
      ['solver', 'Solver'],
      ['critic', 'Critic'],
      ['auditor', 'Auditor'],
      ['skeptic', 'Skeptic']
    ];

    async function fetchJson(url, options) {
      const response = await fetch(url, options);
      const text = await response.text();
      let data;
      try {
        data = text ? JSON.parse(text) : {};
      } catch (_error) {
        throw new Error(`Server returned an invalid response (${response.status}).`);
      }
      if (!response.ok) {
        throw new Error(data.error || `Request failed (${response.status}).`);
      }
      return data;
    }

    async function loadOptions() {
      const data = await fetchJson('/api/options');
      modelOptions = data.models;
      fillBenchmarkCards(data.benchmarkInfo);
      renderAgentRows();
      renderMechanismCards(data.mechanisms);
      $('agent-count').addEventListener('input', () => {
        renderAgentRows();
        renderRoleAssignments();
      });
    }

    function fillBenchmarkCards(benchmarks) {
      const container = $('benchmark-list');
      container.innerHTML = benchmarks.map((benchmark) => {
        const choices = benchmark.example && benchmark.example.choices && Object.keys(benchmark.example.choices).length
          ? `<div class="hint">Example choices: ${Object.entries(benchmark.example.choices).map(([k, v]) => `${k}=${v}`).join(', ')}</div>`
          : '';
        return `
          <div class="task-card">
            <div class="task-title">
              <label>
                <input type="checkbox" name="benchmarks" value="${benchmark.name}" checked>
                <span>${escapeHtml(benchmark.label)}</span>
              </label>
              <div class="question-limit">
                <label for="limit-${benchmark.name}">Questions</label>
                <input id="limit-${benchmark.name}" class="question-count" data-benchmark="${benchmark.name}" type="number" min="1" max="${benchmark.item_count}" value="1">
              </div>
            </div>
            <div class="task-meta">
              <span>${escapeHtml(benchmark.answer_format)}</span>
            </div>
            <details class="compact-details">
              <summary>Task details</summary>
              <div class="hint">${escapeHtml(benchmark.description)}</div>
              <div class="hint">Type: ${escapeHtml(benchmark.task_type)}</div>
              <div class="hint">Skill: ${benchmark.skills.map(escapeHtml).join(', ')}</div>
              <div class="task-example">
                <strong>Example:</strong> ${escapeHtml(benchmark.example.question)}
                <div>Ground Truth: ${escapeHtml(benchmark.example.answer)}</div>
                ${choices}
              </div>
            </details>
          </div>
        `;
      }).join('');
      document.querySelectorAll('input[name="benchmarks"]').forEach((checkbox) => {
        checkbox.addEventListener('change', () => {
          const input = document.querySelector(`.question-count[data-benchmark="${checkbox.value}"]`);
          if (input) input.disabled = !checkbox.checked;
        });
      });
    }

    let modelOptions = [];
    let currentRunData = null;

    function modelOptionsHtml(selected) {
      return modelOptions.map((model) => `
        <option value="${model.key}" ${model.key === selected ? 'selected' : ''}>
          ${model.label} (${model.approximate_size})
        </option>
      `).join('');
    }

    function renderAgentRows() {
      const count = Math.max(1, Math.min(12, Number($('agent-count').value || 3)));
      const previous = selectedAgentModels();
      $('agent-model-list').innerHTML = Array.from({length: count}, (_, index) => {
        const value = previous[index] || 'demo';
        return `
          <div class="agent-row" data-agent-index="${index}">
            <div class="agent-name">A${index + 1}</div>
            <select class="agent-model-select">${modelOptionsHtml(value)}</select>
          </div>
        `;
      }).join('');
      $('copy-agent-model').disabled = count <= 1;
    }

    function selectedAgentModels() {
      return Array.from(document.querySelectorAll('.agent-model-select')).map((node) => node.value);
    }

    function copyFirstAgentModelToAll() {
      const selects = Array.from(document.querySelectorAll('.agent-model-select'));
      if (selects.length < 2) return;
      const firstModel = selects[0].value;
      selects.slice(1).forEach((select) => {
        select.value = firstModel;
      });
      const button = $('copy-agent-model');
      button.textContent = 'Copied from A1 ✓';
      button.classList.add('copied');
      window.setTimeout(() => {
        button.textContent = 'Copy A1 to all';
        button.classList.remove('copied');
      }, 1400);
    }

    function renderMechanismCards(mechanisms) {
      $('mechanism-list').innerHTML = mechanisms.map((mechanism) => mechanismCard(mechanism)).join('');
      renderRoleAssignments();
      $('role-enabled').addEventListener('change', renderRoleAssignments);
    }

    function mechanismCard(mechanism) {
      if (mechanism === 'voting') {
        return `
          <div class="mechanism-card">
            <div class="mechanism-top">
              <label class="mechanism-title"><input type="checkbox" name="mechanisms" value="voting" checked> <span>Voting-based</span></label>
              <div class="mechanism-field">
                <label for="voting-tie-breaker">Tie breaker</label>
                <select id="voting-tie-breaker">
                  <option value="confidence">Confidence</option>
                  <option value="first">First answer</option>
                </select>
              </div>
            </div>
          </div>
        `;
      }
      if (mechanism === 'role') {
        return `
          <div class="mechanism-card">
            <div class="mechanism-top">
              <label class="mechanism-title"><input id="role-enabled" type="checkbox" name="mechanisms" value="role" checked> <span>Role-based</span></label>
              <div class="mechanism-field">
                <label>Roles</label>
                <select disabled><option>A1..AN</option></select>
              </div>
            </div>
            <div class="mechanism-controls">
              <div id="role-assignment" class="role-grid"></div>
            </div>
          </div>
        `;
      }
      return `
        <div class="mechanism-card">
          <div class="mechanism-top">
            <label class="mechanism-title"><input type="checkbox" name="mechanisms" value="debate" checked> <span>Debate-based</span></label>
            <div class="mechanism-field">
              <label for="debate-rounds">Rounds</label>
              <input id="debate-rounds" name="debateRounds" type="number" min="0" max="5" value="2">
            </div>
          </div>
        </div>
      `;
    }

    function renderRoleAssignments() {
      const container = $('role-assignment');
      if (!container) return;
      const count = Math.max(1, Math.min(12, Number($('agent-count').value || 3)));
      const previous = selectedAgentRoles();
      container.innerHTML = Array.from({length: count}, (_, index) => {
        const selected = previous[index] || defaultRole(index, count);
        return `
          <div class="role-row">
            <div class="agent-name">A${index + 1}</div>
            <select class="agent-role-select">${roleOptionsHtml(selected)}</select>
          </div>
        `;
      }).join('');
    }

    function roleOptionsHtml(selected) {
      return roleOptions.map(([value, label]) => `
        <option value="${value}" ${value === selected ? 'selected' : ''}>${label}</option>
      `).join('');
    }

    function defaultRole(index, count) {
      if (count === 1) return 'solver';
      if (index === 0) return 'planner';
      if (index === count - 1) return 'critic';
      return 'solver';
    }

    function selectedAgentRoles() {
      return Array.from(document.querySelectorAll('.agent-role-select')).map((node) => node.value);
    }

    function collectChecked(name) {
      return Array.from(document.querySelectorAll(`input[name="${name}"]:checked`)).map((node) => node.value);
    }

    function collectQuestionLimits() {
      const selected = new Set(collectChecked('benchmarks'));
      return Object.fromEntries(
        Array.from(document.querySelectorAll('.question-count'))
          .filter((node) => selected.has(node.dataset.benchmark))
          .map((node) => [node.dataset.benchmark, node.value])
      );
    }

    function collectPayload() {
      return {
        agentModels: selectedAgentModels(),
        agentRoles: selectedAgentRoles(),
        benchmarks: collectChecked('benchmarks'),
        mechanisms: collectChecked('mechanisms'),
        questionLimits: collectQuestionLimits(),
        agentCount: $('agent-count').value,
        debateRounds: $('debate-rounds') ? $('debate-rounds').value : 2,
        votingTieBreaker: $('voting-tie-breaker') ? $('voting-tie-breaker').value : 'confidence'
      };
    }

    function formatPct(value) {
      return `${(Number(value || 0) * 100).toFixed(1)}%`;
    }

    function formatNumber(value, digits = 3) {
      return Number(value || 0).toFixed(digits);
    }

    function formatInteger(value) {
      return String(Math.round(Number(value || 0)));
    }

    const mechanismMetadata = {
      voting: {
        label: 'Voting',
        description: 'Independent agent votes, followed by majority selection and tie-breaking.'
      },
      role: {
        label: 'Role-based',
        description: 'Role workflow: planner, solver contribution(s), then critic review.'
      },
      debate: {
        label: 'Debate',
        description: 'Initial answers are revised across peer-review rounds before selection.'
      }
    };
    const mechanismOrder = ['voting', 'role', 'debate'];

    function mechanismMeta(mechanism) {
      return mechanismMetadata[mechanism] || {label: mechanism, description: ''};
    }

    function mechanismClass(mechanism) {
      return `mechanism-${String(mechanism || 'unknown').replace(/[^a-z0-9_-]/gi, '-').toLowerCase()}`;
    }

    function renderMechanismHeader(mechanism, statsHtml = '') {
      const meta = mechanismMeta(mechanism);
      return `
        <div class="mechanism-label">
          <div class="mechanism-heading">
            <span class="mechanism-name">${escapeHtml(meta.label)}</span>
            <span class="mechanism-copy">${escapeHtml(meta.description)}</span>
          </div>
          ${statsHtml ? `<div class="mechanism-stats">${statsHtml}</div>` : ''}
        </div>
      `;
    }

    function render(data) {
      currentRunData = data;
      $('empty').style.display = 'none';
      $('results').style.display = 'block';
      $('out-path').textContent = data.resultsPath || '';
      $('download-report').disabled = !(data.records && data.records.length);
      $('visual-report').disabled = !(data.records && data.records.length);

      renderSummaryByMechanism(data.summary.by_benchmark_mechanism || []);
      renderRecordsByMechanism(data.records || [], data.interactions || []);
      renderInteractions(data.interactions || []);
    }

    function renderSummaryByMechanism(rows) {
      $('summary-table').innerHTML = mechanismGroups(rows).map(([mechanism, items]) => `
        <div class="mechanism-section ${mechanismClass(mechanism)}">
          ${renderMechanismHeader(mechanism, `
            <span>accuracy ${formatPct(weightedAverage(items, 'accuracy', 'n'))}</span>
            <span>tokens ${formatNumber(weightedAverage(items, 'mean_total_tokens', 'n'), 1)}</span>
            <span>time ${formatNumber(weightedAverage(items, 'mean_elapsed_s', 'n'), 4)}s</span>
          `)}
          <table>
            <thead><tr>
              <th>Benchmark</th><th>N Questions</th><th>Accuracy</th><th>Mean time</th>
              <th>Mean tokens</th><th>Agreement</th><th>Debate rounds</th>
            </tr></thead>
            <tbody>
              ${items.map((row) => `
                <tr>
                  <td>${escapeHtml(row.benchmark)}</td>
                  <td>${row.n}</td>
                  <td>${formatPct(row.accuracy)}</td>
                  <td>${formatNumber(row.mean_elapsed_s, 4)}s</td>
                  <td>${formatNumber(row.mean_total_tokens, 1)}</td>
                  <td>${formatPct(row.mean_agreement)}</td>
                  <td>${formatInteger(row.mean_debate_rounds)}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      `).join('');
    }

    function renderRecordsByMechanism(records, interactions) {
      $('records-table').innerHTML = mechanismGroups(records).map(([mechanism, items]) => `
        <div class="mechanism-section ${mechanismClass(mechanism)}">
          ${renderMechanismHeader(mechanism, `
            <span>${items.length} answers</span>
            <span>${items.filter((row) => row.correct).length} correct</span>
          `)}
          <div class="table-scroll"><table class="answers-table">
            <colgroup>
              <col style="width:22%"><col style="width:14%"><col style="width:14%"><col style="width:50%">
            </colgroup>
            <thead><tr>
              <th>Task</th><th>Ground Truth</th><th>Coordinator</th><th>Mechanism details</th>
            </tr></thead>
            <tbody>
              ${items.map((row) => {
                const interaction = findInteraction(interactions, row);
                return `
                  <tr>
                    <td>
                      <div class="record-title">
                        <strong>${escapeHtml(row.task_label || row.benchmark)}</strong>
                        <span class="record-id">${escapeHtml(row.item_id)}</span>
                      </div>
                      <div class="record-meta">${row.total_tokens} tokens · ${formatNumber(row.elapsed_s, 4)}s</div>
                    </td>
                    <td><strong class="record-answer">${escapeHtml(row.gold_display || row.gold_answer)}</strong></td>
                    <td title="${escapeHtml(row.match_explanation || '')}">
                      <div class="coordinator-result">
                        <strong class="record-answer">${escapeHtml(row.final_answer_display || row.final_answer)}</strong>
                        ${resultSymbol(row.correct)}
                      </div>
                    </td>
                    <td class="mechanism-detail-cell">${renderMechanismDetail(interaction, row)}</td>
                  </tr>
                `;
              }).join('')}
            </tbody>
          </table></div>
        </div>
      `).join('');
    }

    function findInteraction(interactions, row) {
      return interactions.find((interaction) =>
        interaction.mechanism === row.mechanism &&
        interaction.item_id === row.item_id &&
        interaction.benchmark === row.benchmark
      );
    }

    function renderMechanismDetail(interaction, row) {
      if (!interaction) return '<div class="hint">Details will appear when this result completes.</div>';
      const summaryParts = [
        `agreement ${formatPct(interaction.agreement ?? row.agreement)}`,
        `confidence ${formatPct(interaction.confidence ?? row.confidence)}`
      ];

      if (interaction.mechanism === 'voting') {
        summaryParts.push(`tie ${interaction.tie ? 'yes' : 'no'}`);
        const voteStep = (interaction.trace || []).find((step) => Array.isArray(step.votes));
        const votes = voteStep ? voteStep.votes : [];
        return `
          <div class="decision-summary">${summaryParts.join(' · ')}</div>
          <div class="decision-list">
            ${votes.map((output) => renderDecisionRow(output, {
              label: 'vote',
              selected: isSelectedOutput(output, interaction)
            })).join('')}
          </div>
        `;
      }

      if (interaction.mechanism === 'role') {
        const contributions = (interaction.trace || []).flatMap((step) =>
          traceOutputs(step).map((output) => ({step, output}))
        );
        return `
          <div class="decision-summary">${summaryParts.join(' · ')}</div>
          <div class="decision-list">
            ${contributions.map(({step, output}) => renderDecisionRow(output, {
              label: 'answer',
              showRole: true
            })).join('')}
          </div>
        `;
      }

      const steps = interaction.trace || [];
      const finalStep = steps.length ? steps[steps.length - 1] : null;
      const outputs = finalStep ? traceOutputs(finalStep) : [];
      return `
        <div class="decision-summary">${summaryParts.join(' · ')} · rounds ${formatInteger(interaction.debate_rounds)}</div>
        <div class="decision-list">
          ${outputs.map((output) => renderDecisionRow(output, {
            label: 'final answer',
            selected: isSelectedOutput(output, interaction)
          })).join('')}
        </div>
      `;
    }

    function renderDecisionRow(output, options = {}) {
      const role = String(output.role || '').trim();
      const phase = String(options.phase || '').trim();
      const selectedBadge = options.selected ? '<span class="decision-badge">winner</span>' : '';
      const roleBadge = options.showRole && role
        ? `<span class="decision-badge role-badge">${escapeHtml(role)}</span>`
        : '';
      const context = phase;
      return `
        <div class="decision-row ${options.selected ? 'selected' : ''}">
          <div class="decision-agent">
            ${escapeHtml(output.agent || 'agent')}${roleBadge}
            ${context ? `<span class="decision-role">${escapeHtml(context)}</span>` : ''}
          </div>
          <div class="decision-answer">
            ${selectedBadge}${escapeHtml(options.label || 'answer')}: ${escapeHtml(output.answer_display || output.answer || '')}
          </div>
          <div class="decision-confidence">conf. ${formatPct(output.confidence)}</div>
        </div>
      `;
    }

    function isSelectedOutput(output, interaction) {
      const mapped = String(output.mapped_answer || '').trim().toLowerCase();
      const finalMapped = String(interaction.final_mapped_answer || '').trim().toLowerCase();
      if (mapped && finalMapped) return mapped === finalMapped;
      const answer = String(output.answer_display || output.answer || '').trim().toLowerCase();
      const finalAnswer = String(interaction.final_answer_display || interaction.final_answer || '').trim().toLowerCase();
      return answer === finalAnswer;
    }

    function traceOutputs(step) {
      return step.votes || step.outputs || (step.output ? [step.output] : []);
    }

    function humanizeStep(value) {
      const step = String(value || 'step');
      const labels = {
        vote: 'Vote',
        initial: 'Initial answers',
        planner: 'Planner phase',
        solver: 'Solver phase',
        critic: 'Critic phase',
        single_agent: 'Single-agent role workflow'
      };
      if (labels[step]) return labels[step];
      if (step.startsWith('worker_')) return `Solver ${step.split('_').pop()}`;
      if (step.startsWith('debate_round_')) return `Debate round ${step.split('_').pop()}`;
      return step.replace(/_/g, ' ');
    }

    function renderInteractions(interactions) {
      if (!interactions.length) {
        $('interaction-list').innerHTML = '<div class="hint">No trace available.</div>';
        return;
      }
      $('interaction-list').innerHTML = mechanismGroups(interactions).map(([mechanism, items]) => `
        <div class="mechanism-section ${mechanismClass(mechanism)}">
          ${renderMechanismHeader(mechanism, `<span>${items.length} traces</span>`)}
          <div class="interaction-list">
            ${items.map((interaction) => `
              <div class="interaction-card">
                <div class="interaction-title">
                  ${escapeHtml(interaction.item_id)} ·
                  ${escapeHtml(interaction.task_label || interaction.benchmark)} ·
                  ${resultSymbol(interaction.correct)}
                  <div class="hint">
                    Ground Truth: ${escapeHtml(interaction.gold_display || interaction.gold_answer)} ·
                    Coordinator: ${escapeHtml(interaction.final_answer_display || interaction.final_answer)} ·
                    Agreement: ${formatPct(interaction.agreement)} ·
                    Confidence: ${formatPct(interaction.confidence)}
                  </div>
                </div>
                <div class="question-text">
                  <strong>Question</strong>
                  <em>${escapeHtml(interaction.question || '')}</em>
                </div>
                ${interaction.trace.map((step) => renderTraceStep(step, interaction)).join('')}
              </div>
            `).join('')}
          </div>
        </div>
      `).join('');
    }

    function renderTraceStep(step, interaction) {
      const outputs = traceOutputs(step);
      return `
        <div class="step">
          <div class="step-name">${escapeHtml(humanizeStep(step.step))}</div>
          ${step.description ? `<div class="step-description">${escapeHtml(step.description)}</div>` : ''}
          ${outputs.map((output) => renderAgentOutput(output, interaction, step)).join('')}
        </div>
      `;
    }

    function renderAgentOutput(output, interaction, step) {
      const reported = output.reported_answer && String(output.reported_answer) !== String(output.answer)
        ? ` · reported: ${escapeHtml(output.reported_answer)}`
        : '';
      const note = output.answer_note
        ? `<div class="agent-meta">${escapeHtml(output.answer_note)}</div>`
        : '';
      const mechanism = interaction ? interaction.mechanism : '';
      const action = mechanism === 'voting' ? 'vote' : 'answer';
      const roleBadge = mechanism === 'role' && output.role
        ? `<span class="decision-badge role-badge">${escapeHtml(output.role)}</span>`
        : '';
      const roleMeta = mechanism === 'role' && output.role
        ? ` · role: ${escapeHtml(output.role)}`
        : '';
      return `
        <div class="agent-message ${output.is_correct_against_gold ? 'correct' : 'incorrect'}">
          <strong>${escapeHtml(output.agent || 'agent')}${roleBadge} · ${action}: ${escapeHtml(output.answer_display || output.answer || '')}</strong>
          <div class="agent-meta">
            phase: ${escapeHtml(humanizeStep(step && step.step))} · model: ${escapeHtml(output.model || '')} · confidence: ${formatPct(output.confidence)}${roleMeta}
            · ${output.is_correct_against_gold ? 'matches Ground Truth' : 'does not match Ground Truth'}${reported}
          </div>
          ${note}
          <div class="rationale">${escapeHtml(output.rationale || output.raw || '')}</div>
        </div>
      `;
    }

    function resultSymbol(correct) {
      return `<span class="result-symbol ${correct ? 'true' : 'false'}">${correct ? '✓' : '✕'}</span>`;
    }

    function downloadReport() {
      if (!currentRunData || !currentRunData.records || !currentRunData.records.length) return;
      const report = buildMarkdownReport(currentRunData);
      const blob = new Blob([report], {type: 'text/markdown;charset=utf-8'});
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `agentic-systems-report-${new Date().toISOString().replace(/[:.]/g, '-')}.md`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }

    const reportColors = {
      voting: '#2563eb',
      role: '#7c3aed',
      debate: '#d97706'
    };

    function openVisualReport() {
      if (!currentRunData || !currentRunData.records || !currentRunData.records.length) return;
      renderVisualReport(currentRunData);
      const modal = $('visual-report-modal');
      document.body.classList.add('modal-open');
      if (!modal.open) modal.showModal();
    }

    function closeVisualReport() {
      const modal = $('visual-report-modal');
      if (modal.open) modal.close();
    }

    function aggregateReportMechanisms(data) {
      const rows = (data.summary && data.summary.by_benchmark_mechanism) || [];
      const records = data.records || [];
      return mechanismGroups(rows).map(([mechanism, items]) => {
        const mechanismRecords = records.filter((record) => record.mechanism === mechanism);
        return {
          mechanism,
          n: items.reduce((sum, item) => sum + Number(item.n || 0), 0),
          accuracy: weightedAverage(items, 'accuracy', 'n'),
          elapsed: weightedAverage(items, 'mean_elapsed_s', 'n'),
          tokens: weightedAverage(items, 'mean_total_tokens', 'n'),
          agreement: weightedAverage(items, 'mean_agreement', 'n'),
          tieRate: weightedAverage(items, 'tie_rate', 'n'),
          debateRounds: weightedAverage(items, 'mean_debate_rounds', 'n'),
          correct: mechanismRecords.filter((record) => record.correct).length,
          incorrect: mechanismRecords.filter((record) => !record.correct).length
        };
      });
    }

    function reportBarRows(items, valueKey, formatter, fixedMax = null) {
      const maximum = fixedMax === null
        ? Math.max(...items.map((item) => Number(item[valueKey] || 0)), 0)
        : fixedMax;
      if (!items.length) return '<div class="report-empty">No data available.</div>';
      return items.map((item) => {
        const value = Number(item[valueKey] || 0);
        const width = maximum ? Math.max(1.5, Math.min(100, value / maximum * 100)) : 0;
        const meta = mechanismMeta(item.mechanism);
        return `
          <div class="report-bar-row">
            <div class="report-bar-label" title="${escapeHtml(meta.label)}">${escapeHtml(meta.label)}</div>
            <div class="report-bar-track" aria-hidden="true">
              <div class="report-bar-fill" style="--bar-width:${width}%;--bar-color:${reportColors[item.mechanism] || '#0f766e'}"></div>
            </div>
            <div class="report-bar-value">${formatter(value)}</div>
          </div>
        `;
      }).join('');
    }

    function reportChartCard(title, subtitle, content, wide = false) {
      return `
        <article class="report-card ${wide ? 'wide' : ''}">
          <div class="report-card-head"><h3>${escapeHtml(title)}</h3><p>${escapeHtml(subtitle)}</p></div>
          <div class="report-chart">${content}</div>
        </article>
      `;
    }

    function renderVisualReport(data) {
      const overall = (data.summary && data.summary.overall) || {};
      const summaryRows = (data.summary && data.summary.by_benchmark_mechanism) || [];
      const records = data.records || [];
      const mechanisms = aggregateReportMechanisms(data);
      const computeUnits = Array.isArray(data.computeUnits) ? data.computeUnits : [];
      const computeLabel = [...new Set(computeUnits.map((unit) => unit.label).filter(Boolean))].join(' + ') || 'Not reported';
      const bestAccuracy = mechanisms.length
        ? mechanisms.reduce((best, item) => item.accuracy > best.accuracy ? item : best)
        : null;
      const fastest = mechanisms.length
        ? mechanisms.reduce((best, item) => item.elapsed < best.elapsed ? item : best)
        : null;
      const leanest = mechanisms.length
        ? mechanisms.reduce((best, item) => item.tokens < best.tokens ? item : best)
        : null;
      const isComplete = data.status === 'done';

      const outcomeRows = mechanisms.map((item) => {
        const total = item.correct + item.incorrect;
        const correctWidth = total ? item.correct / total * 100 : 0;
        const incorrectWidth = total ? item.incorrect / total * 100 : 0;
        return `
          <div class="report-bar-row">
            <div class="report-bar-label">${escapeHtml(mechanismMeta(item.mechanism).label)}</div>
            <div class="report-outcome-track" aria-label="${item.correct} correct, ${item.incorrect} incorrect">
              <div class="report-outcome-correct" style="width:${correctWidth}%"></div>
              <div class="report-outcome-wrong" style="width:${incorrectWidth}%"></div>
            </div>
            <div class="report-bar-value">${item.correct} / ${total}</div>
          </div>
        `;
      }).join('');

      const detailRows = summaryRows.map((row) => `
        <tr>
          <td><strong>${escapeHtml(mechanismMeta(row.mechanism).label)}</strong></td>
          <td>${escapeHtml(row.benchmark)}</td>
          <td>${formatInteger(row.n)}</td>
          <td>
            <div class="report-score">
              <strong>${formatPct(row.accuracy)}</strong>
              <div class="report-bar-track"><div class="report-bar-fill" style="--bar-width:${Math.max(1.5, Number(row.accuracy || 0) * 100)}%;--bar-color:${reportColors[row.mechanism] || '#0f766e'}"></div></div>
            </div>
          </td>
          <td>${formatNumber(row.mean_elapsed_s, 4)}s</td>
          <td>${formatNumber(row.mean_total_tokens, 1)}</td>
          <td>${formatPct(row.mean_agreement)}</td>
          <td>${formatPct(row.tie_rate)}</td>
          <td>${formatNumber(row.mean_debate_rounds, 1)}</td>
        </tr>
      `).join('');

      const resultRows = records.map((row) => `
        <tr>
          <td><strong>${escapeHtml(row.task_label || row.benchmark)}</strong><div class="record-meta">${escapeHtml(row.item_id)}</div></td>
          <td>${escapeHtml(mechanismMeta(row.mechanism).label)}</td>
          <td>${escapeHtml(row.gold_display || row.gold_answer)}</td>
          <td>${escapeHtml(row.final_answer_display || row.final_answer)}</td>
          <td><span class="report-result-badge ${row.correct ? 'correct' : 'incorrect'}">${row.correct ? '✓ Correct' : '✕ Incorrect'}</span></td>
          <td>${formatPct(row.agreement)}</td>
          <td>${formatInteger(row.total_tokens)}</td>
          <td>${formatNumber(row.elapsed_s, 4)}s</td>
        </tr>
      `).join('');

      $('visual-report-content').innerHTML = `
        <div class="report-meta-strip">
          <span>${isComplete ? 'Complete experiment' : 'Live partial report'}</span>
          <span>Compute: ${escapeHtml(computeLabel)}</span>
          <span>Generated: ${escapeHtml(new Date().toLocaleString())}</span>
          ${data.resultsPath ? `<span title="${escapeHtml(data.resultsPath)}">Saved results available</span>` : ''}
        </div>

        <section class="report-kpis" aria-label="Experiment headline metrics">
          <div class="report-kpi" style="--kpi-tint:#dbeafe"><div class="report-kpi-label">Unique questions</div><div class="report-kpi-value">${formatInteger(overall.question_count)}</div><div class="report-kpi-note">Across selected benchmarks</div></div>
          <div class="report-kpi" style="--kpi-tint:#ede9fe"><div class="report-kpi-label">Evaluations</div><div class="report-kpi-value">${formatInteger(overall.n || records.length)}</div><div class="report-kpi-note">Question × mechanism runs</div></div>
          <div class="report-kpi" style="--kpi-tint:#d1fae5"><div class="report-kpi-label">Overall accuracy</div><div class="report-kpi-value">${formatPct(overall.accuracy)}</div><div class="report-kpi-note">Correct coordinator answers</div></div>
          <div class="report-kpi" style="--kpi-tint:#fef3c7"><div class="report-kpi-label">Mean response time</div><div class="report-kpi-value">${formatNumber(overall.mean_elapsed_s, 3)}s</div><div class="report-kpi-note">Per evaluation</div></div>
          <div class="report-kpi" style="--kpi-tint:#cffafe"><div class="report-kpi-label">Mean tokens</div><div class="report-kpi-value">${formatNumber(overall.mean_total_tokens, 0)}</div><div class="report-kpi-note">Prompt + completion</div></div>
        </section>

        <section class="report-grid" aria-label="Experiment charts">
          ${reportChartCard('Accuracy by mechanism', 'Higher is better; scale is fixed from 0 to 100%.', reportBarRows(mechanisms, 'accuracy', formatPct, 1))}
          ${reportChartCard('Agreement by mechanism', 'How consistently agents converged on the same answer.', reportBarRows(mechanisms, 'agreement', formatPct, 1))}
          ${reportChartCard('Mean response time', 'Average wall-clock duration per evaluated item.', reportBarRows(mechanisms, 'elapsed', (value) => `${formatNumber(value, 3)}s`))}
          ${reportChartCard('Mean token use', 'Average prompt and completion tokens per evaluated item.', reportBarRows(mechanisms, 'tokens', (value) => formatNumber(value, 0)))}
          ${reportChartCard('Correct vs. incorrect outcomes', 'Green shows correct answers; coral shows errors. The value is correct / total.', outcomeRows || '<div class="report-empty">No outcomes available.</div>', true)}

          <article class="report-card wide">
            <div class="report-card-head"><h3>At-a-glance leaders</h3><p>Best observed values within this experiment; no cross-run assumptions.</p></div>
            <div class="report-insights">
              <div class="report-insight"><span>Highest accuracy</span><strong>${bestAccuracy ? `${escapeHtml(mechanismMeta(bestAccuracy.mechanism).label)} · ${formatPct(bestAccuracy.accuracy)}` : 'No data'}</strong></div>
              <div class="report-insight"><span>Fastest mean time</span><strong>${fastest ? `${escapeHtml(mechanismMeta(fastest.mechanism).label)} · ${formatNumber(fastest.elapsed, 3)}s` : 'No data'}</strong></div>
              <div class="report-insight"><span>Lowest token use</span><strong>${leanest ? `${escapeHtml(mechanismMeta(leanest.mechanism).label)} · ${formatNumber(leanest.tokens, 0)}` : 'No data'}</strong></div>
            </div>
          </article>

          <article class="report-card wide">
            <div class="report-card-head"><h3>Benchmark × mechanism detail</h3><p>Full aggregate metrics, including ties and debate rounds.</p></div>
            <div class="report-table-wrap"><table>
              <thead><tr><th>Mechanism</th><th>Benchmark</th><th>N</th><th>Accuracy</th><th>Mean time</th><th>Mean tokens</th><th>Agreement</th><th>Tie rate</th><th>Rounds</th></tr></thead>
              <tbody>${detailRows || '<tr><td colspan="9">No aggregate data available.</td></tr>'}</tbody>
            </table></div>
          </article>

          <article class="report-card wide">
            <div class="report-card-head"><h3>Individual experiment outcomes</h3><p>Ground truth, coordinator answer, quality, cost, and timing for every evaluation.</p></div>
            <div class="report-table-wrap"><table>
              <thead><tr><th>Task</th><th>Mechanism</th><th>Ground truth</th><th>Coordinator</th><th>Result</th><th>Agreement</th><th>Tokens</th><th>Time</th></tr></thead>
              <tbody>${resultRows || '<tr><td colspan="8">No results available.</td></tr>'}</tbody>
            </table></div>
          </article>
        </section>
      `;
    }

    function buildMarkdownReport(data) {
      const lines = [];
      lines.push('# Agentic Systems Experiment Report', '');
      lines.push(`Generated: ${new Date().toISOString()}`);
      if (data.resultsPath) lines.push(`Results path: ${data.resultsPath}`);
      lines.push('', '## Metrics by Mechanism', '');
      lines.push('| Mechanism | Benchmark | N Questions | Accuracy | Mean time | Mean tokens | Agreement | Debate rounds |');
      lines.push('|---|---:|---:|---:|---:|---:|---:|---:|');
      (data.summary.by_benchmark_mechanism || []).forEach((row) => {
        lines.push(`| ${md(row.mechanism)} | ${md(row.benchmark)} | ${row.n} | ${formatPct(row.accuracy)} | ${formatNumber(row.mean_elapsed_s, 4)}s | ${formatNumber(row.mean_total_tokens, 1)} | ${formatPct(row.mean_agreement)} | ${formatInteger(row.mean_debate_rounds)} |`);
      });
      lines.push('', '## Answers and Ground Truth by Mechanism', '');
      mechanismGroups(data.records || []).forEach(([mechanism, rows]) => {
        const meta = mechanismMeta(mechanism);
        lines.push(`### ${meta.label}`, '', meta.description, '');
        lines.push('| Item | Task | Ground Truth | Coordinator answer | Result | Agreement | Tokens | Time |');
        lines.push('|---|---|---|---|---:|---:|---:|---:|');
        rows.forEach((row) => {
          lines.push(`| ${md(row.item_id)} | ${md(row.task_label || row.benchmark)} | ${md(row.gold_display || row.gold_answer)} | ${md(row.final_answer_display || row.final_answer)} | ${row.correct ? 'TRUE' : 'FALSE'} | ${formatPct(row.agreement)} | ${row.total_tokens} | ${formatNumber(row.elapsed_s, 4)}s |`);
        });
        lines.push('');
      });
      lines.push('', '## Agent Interactions', '');
      (data.interactions || []).forEach((interaction) => {
        const meta = mechanismMeta(interaction.mechanism);
        lines.push(`### ${meta.label} / ${interaction.item_id}`, '');
        lines.push(`Question: ${interaction.question || ''}`);
        lines.push(`Ground Truth: ${interaction.gold_display || interaction.gold_answer}`);
        lines.push(`Coordinator answer: ${interaction.final_answer_display || interaction.final_answer}`);
        lines.push(`Agreement: ${formatPct(interaction.agreement)}; confidence: ${formatPct(interaction.confidence)}`);
        if (interaction.mechanism === 'voting') lines.push(`Tie: ${interaction.tie ? 'yes' : 'no'}`);
        if (interaction.mechanism === 'debate') lines.push(`Debate rounds: ${formatInteger(interaction.debate_rounds)}`);
        lines.push(`Result: ${interaction.correct ? 'TRUE' : 'FALSE'}`, '');
        (interaction.trace || []).forEach((step) => {
          lines.push(`#### ${humanizeStep(step.step)}`);
          const outputs = traceOutputs(step);
          outputs.forEach((output) => {
            const action = interaction.mechanism === 'voting' ? 'vote' : 'answer';
            const role = interaction.mechanism === 'role' ? `role: ${output.role || ''} | ` : '';
            lines.push(`- ${role}${output.agent || 'agent'} | ${action}: ${output.answer_display || output.answer || ''} | confidence: ${formatPct(output.confidence)}`);
            if (output.answer_note) lines.push(`  - note: ${output.answer_note}`);
            if (output.rationale) lines.push(`  - rationale: ${String(output.rationale).replace(/\s+/g, ' ').trim()}`);
          });
          lines.push('');
        });
      });
      return lines.join('\n');
    }

    function md(value) {
      return String(value ?? '').replace(/\|/g, '\\|').replace(/\n/g, ' ');
    }

    function groupBy(items, key) {
      return items.reduce((acc, item) => {
        const group = item[key] || 'unknown';
        if (!acc[group]) acc[group] = [];
        acc[group].push(item);
        return acc;
      }, {});
    }

    function mechanismGroups(items) {
      const grouped = groupBy(items, 'mechanism');
      return Object.entries(grouped).sort(([left], [right]) => {
        const leftIndex = mechanismOrder.indexOf(left);
        const rightIndex = mechanismOrder.indexOf(right);
        const leftRank = leftIndex === -1 ? mechanismOrder.length : leftIndex;
        const rightRank = rightIndex === -1 ? mechanismOrder.length : rightIndex;
        return leftRank - rightRank || left.localeCompare(right);
      });
    }

    function weightedAverage(items, valueKey, weightKey) {
      const totalWeight = items.reduce((sum, item) => sum + Number(item[weightKey] || 0), 0);
      if (!totalWeight) return 0;
      return items.reduce((sum, item) => sum + Number(item[valueKey] || 0) * Number(item[weightKey] || 0), 0) / totalWeight;
    }

    function escapeHtml(value) {
      return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }

    $('download-report').addEventListener('click', downloadReport);
    $('visual-report').addEventListener('click', openVisualReport);
    $('copy-agent-model').addEventListener('click', copyFirstAgentModelToAll);
    $('visual-report-close').addEventListener('click', closeVisualReport);
    $('visual-report-modal').addEventListener('close', () => document.body.classList.remove('modal-open'));
    $('visual-report-modal').addEventListener('click', (event) => {
      if (event.target === $('visual-report-modal')) closeVisualReport();
    });

    $('experiment-form').addEventListener('submit', async (event) => {
      event.preventDefault();
      $('run-button').disabled = true;
      $('status').textContent = 'Starting...';
      updateComputeUnit({status: 'queued', computeUnits: []});
      $('empty').style.display = 'none';
      $('results').style.display = 'block';
      render({summary: {overall: {question_count: 0, accuracy: 0, mean_elapsed_s: 0, mean_total_tokens: 0}, by_benchmark_mechanism: []}, records: [], interactions: [], resultsPath: ''});
      try {
        const data = await fetchJson('/api/run', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(collectPayload())
        });
        if (!data.ok) {
          throw new Error(data.error || 'Unknown error');
        }
        await pollRun(data.jobId);
      } catch (error) {
        $('status').textContent = error.message;
      } finally {
        $('run-button').disabled = false;
      }
    });

    async function pollRun(jobId) {
      const records = [];
      const interactions = [];
      let retryCount = 0;
      let pollDelay = 500;
      let previousStatus = '';
      while (true) {
        const query = new URLSearchParams({
          job: jobId,
          recordOffset: String(records.length),
          interactionOffset: String(interactions.length)
        });
        let data;
        try {
          data = await fetchJson(`/api/run-status?${query}`);
          retryCount = 0;
        } catch (error) {
          retryCount += 1;
          if (retryCount > 5) throw error;
          $('status').textContent = `Connection interrupted; retrying (${retryCount}/5)...`;
          await new Promise((resolve) => setTimeout(resolve, Math.min(4000, retryCount * 750)));
          continue;
        }
        if (!data.ok) throw new Error(data.error || 'Run status unavailable');
        const job = data.job;
        const newRecords = job.records || [];
        const newInteractions = job.interactions || [];
        records.push(...newRecords);
        interactions.push(...newInteractions);
        const completeJob = {...job, records: [...records], interactions: [...interactions]};
        const statusChanged = job.status !== previousStatus;
        if (newRecords.length || newInteractions.length || statusChanged || job.status === 'done') {
          render(completeJob);
        }
        updateRunProgress(completeJob);
        previousStatus = job.status;
        if (job.status === 'done') {
          $('status').textContent = 'Complete.';
          return;
        }
        if (job.status === 'error') {
          throw new Error(job.error || job.message || 'Experiment error');
        }
        pollDelay = newRecords.length || newInteractions.length ? 500 : Math.min(2000, pollDelay * 1.4);
        await new Promise((resolve) => setTimeout(resolve, pollDelay));
      }
    }

    function updateRunProgress(job) {
      $('status').textContent = job.message || job.status;
      updateComputeUnit(job);
      if (job.status === 'downloading') {
        $('download-progress').style.display = 'block';
        $('download-bar').style.width = `${job.downloadPercent || 0}%`;
        $('download-text').textContent = downloadMessage(job);
        return;
      }
      if (job.status === 'done') {
        $('download-bar').style.width = '100%';
        $('download-text').textContent = 'Complete.';
        return;
      }
      $('download-progress').style.display = 'block';
      const completed = Number(job.completedRecords || 0);
      const total = Number(job.totalRecords || 0);
      const percent = total ? Math.min(100, completed / total * 100) : 0;
      $('download-bar').style.width = `${percent}%`;
      $('download-text').textContent = total
        ? `${job.message || 'Running.'} ${completed} / ${total}`
        : (job.message || 'Running.');
    }

    function updateComputeUnit(job) {
      const container = $('compute-unit');
      const value = $('compute-unit-value');
      const units = Array.isArray(job.computeUnits) ? job.computeUnits : [];
      container.classList.remove('gpu', 'cpu');
      if (!units.length) {
        value.textContent = job.status === 'queued' || job.status === 'running'
          ? 'Detecting...'
          : 'Not active';
        return;
      }
      const kinds = new Set(units.map((unit) => unit.kind));
      const labels = [...new Set(units.map((unit) => unit.label).filter(Boolean))];
      const suffix = job.status === 'done' ? 'used' : 'active';
      value.textContent = `${labels.join(' + ')} · ${suffix}`;
      if (kinds.has('gpu')) container.classList.add('gpu');
      else container.classList.add('cpu');
    }

    function downloadMessage(job) {
      const total = Number(job.bytesTotal || 0);
      const done = Number(job.bytesDone || 0);
      const mb = (value) => (value / 1_000_000).toFixed(1);
      const detail = total ? `${mb(done)} / ${mb(total)} MB` : `${mb(done)} MB`;
      return `${job.currentLabel || 'Download'} · ${job.downloadPercent || 0}% · ${detail}`;
    }

    loadOptions().catch((error) => {
      $('status').textContent = error.message;
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
