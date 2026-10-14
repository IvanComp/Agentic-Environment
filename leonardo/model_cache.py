#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time

from agentic_quant.model_registry import get_model_preset
from agentic_quant.model_store import delete_model, download_model, model_status
from agentic_quant.ollama_client import ollama_models


_printed: dict[str, float] = {"percent": -1.0}


def _progress(done: int, total: int) -> None:
    percent = 100.0 if total <= 0 else min(100.0, done / total * 100)
    if percent < 100.0 and percent - _printed["percent"] < 1.0:
        return
    _printed["percent"] = percent
    print(f"model-download bytes={done}/{total} percent={percent:.1f}", flush=True)


def _wait_until_ready(model_key: str, timeout: int) -> dict:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        status = model_status(model_key)
        last_error = str(status.get("error") or "")
        if status.get("downloaded"):
            return status
        time.sleep(2)
    raise RuntimeError(f"Model {model_key!r} did not become ready within {timeout}s. {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage one Agentic Quant model in the shared HPC cache.")
    parser.add_argument("command", choices=["backend", "prepare", "delete", "status", "wait", "wait-server"])
    parser.add_argument("model")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    preset = get_model_preset(args.model)
    if args.command == "backend":
        print(preset.backend)
        return
    if args.command == "wait-server":
        deadline = time.monotonic() + args.timeout
        while True:
            try:
                ollama_models(timeout=3)
                print("Ollama server ready")
                return
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(2)
    if args.command == "prepare":
        path = download_model(args.model, progress=_progress)
        print(json.dumps({"model": args.model, "path": str(path), "status": model_status(args.model)}, indent=2))
        return
    if args.command == "delete" and preset.keep_cached:
        print(json.dumps({"model": args.model, "removed": [], "kept": True}, indent=2))
        return
    if args.command == "delete":
        print(json.dumps({"model": args.model, "removed": delete_model(args.model)}, indent=2))
        return
    if args.command == "wait":
        print(json.dumps(_wait_until_ready(args.model, args.timeout), indent=2))
        return
    print(json.dumps(model_status(args.model), indent=2))


if __name__ == "__main__":
    main()
