from __future__ import annotations

import os
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

from .model_registry import get_model_preset
from .ollama_client import ollama_model_info, pull_ollama_model
from .paths import workspace_root


PROJECT_ROOT = workspace_root()
MODELS_DIR = PROJECT_ROOT / "models"

ProgressCallback = Callable[[int, int], None]
_MODEL_LOCKS: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)


def model_file_paths(model_key: str) -> list[Path]:
    preset = get_model_preset(model_key)
    if not preset.filenames:
        return [MODELS_DIR / model_key]
    return [MODELS_DIR / model_key / filename for filename in preset.filenames]


def model_path(model_key: str) -> Path:
    return model_file_paths(model_key)[0]


def model_status(model_key: str) -> dict[str, Any]:
    preset = get_model_preset(model_key)
    if model_key == "demo":
        return {
            **preset.to_dict(),
            "downloaded": True,
            "path": "",
            "file_count": 0,
            "files": [],
            "bytes_downloaded": 0,
            "bytes_total": 0,
            "percent": 100,
        }
    if getattr(preset, "backend", "gguf") == "ollama":
        return _ollama_model_status(model_key)

    paths = model_file_paths(model_key)
    file_statuses = []
    for path in paths:
        exists = path.is_file()
        partial_path = path.with_suffix(path.suffix + ".part")
        partial_bytes = partial_path.stat().st_size if not exists and partial_path.is_file() else 0
        file_statuses.append(
            {
                "filename": path.name,
                "path": str(path),
                "exists": exists,
                "bytes": path.stat().st_size if exists else 0,
                "partial_bytes": partial_bytes,
            }
        )
    downloaded_bytes = sum(file_status["bytes"] or file_status["partial_bytes"] for file_status in file_statuses)
    total = preset.expected_bytes or downloaded_bytes
    downloaded = all(file_status["exists"] for file_status in file_statuses) and (
        preset.expected_bytes == 0 or downloaded_bytes == preset.expected_bytes
    )
    percent = 0 if not total else min(100, round(downloaded_bytes / total * 100, 1))
    return {
        **preset.to_dict(),
        "downloaded": downloaded,
        "path": str(paths[0]),
        "file_count": len(paths),
        "files": file_statuses,
        "bytes_downloaded": downloaded_bytes,
        "bytes_total": total,
        "percent": percent,
    }


def selected_models_status(model_keys: list[str]) -> list[dict]:
    seen: set[str] = set()
    statuses = []
    for key in model_keys:
        if key in seen:
            continue
        seen.add(key)
        statuses.append(model_status(key))
    return statuses


def missing_model_keys(model_keys: list[str]) -> list[str]:
    return [status["key"] for status in selected_models_status(model_keys) if not status["downloaded"]]


def ensure_models(model_keys: list[str], *, download: bool, progress: Callable[[str, int, int], None] | None = None) -> None:
    missing = missing_model_keys(model_keys)
    if missing and not download:
        labels = ", ".join(get_model_preset(key).label for key in missing)
        if any(getattr(get_model_preset(key), "backend", "gguf") == "ollama" for key in missing):
            raise RuntimeError(
                f"Ollama models unavailable: {labels}. Start Ollama, then use --download-model "
                "or pull the requested tags with `ollama pull`."
            )
        raise RuntimeError(
            "Models not downloaded: "
            f"{labels}. Download them from the UI or rerun with --download-model."
        )
    for key in missing:
        download_model(key, progress=None if progress is None else lambda done, total, model_key=key: progress(model_key, done, total))


def download_model(model_key: str, *, progress: ProgressCallback | None = None) -> Path:
    with _MODEL_LOCKS[model_key]:
        return _download_model_locked(model_key, progress=progress)


def _download_model_locked(model_key: str, *, progress: ProgressCallback | None = None) -> Path:
    preset = get_model_preset(model_key)
    if model_key == "demo":
        return model_path(model_key)
    if getattr(preset, "backend", "gguf") == "ollama":
        pull_ollama_model(preset.ollama_model or model_key, progress=progress)
        return model_path(model_key)
    if not preset.download_urls:
        raise ValueError(f"Model {model_key} does not define a download URL.")

    final_paths = model_file_paths(model_key)
    status = model_status(model_key)
    if status["downloaded"]:
        if progress:
            progress(status["bytes_total"], status["bytes_total"])
        return final_paths[0]

    final_paths[0].parent.mkdir(parents=True, exist_ok=True)
    expected_total = preset.expected_bytes

    # A previous version may have left complete-looking files whose aggregate
    # size is invalid. Redownload that set instead of permanently failing on it.
    if expected_total and all(path.is_file() for path in final_paths):
        current_total = sum(path.stat().st_size for path in final_paths)
        if current_total != expected_total:
            for path in final_paths:
                path.unlink()

    completed_bytes = sum(path.stat().st_size for path in final_paths if path.is_file())
    if progress and completed_bytes:
        progress(completed_bytes, expected_total or completed_bytes)

    for url, final_path in zip(preset.download_urls, final_paths, strict=True):
        if final_path.is_file():
            continue
        tmp_path = final_path.with_suffix(final_path.suffix + ".part")
        resume_bytes = tmp_path.stat().st_size if tmp_path.is_file() else 0
        headers = {"User-Agent": "agentic-quant/0.2"}
        if resume_bytes:
            headers["Range"] = f"bytes={resume_bytes}-"

        request = Request(url, headers=headers)
        with urlopen(request, timeout=60) as response:
            status_code = getattr(response, "status", response.getcode())
            range_start = _content_range_start(response)
            if resume_bytes and status_code == 206 and range_start != resume_bytes:
                raise RuntimeError(
                    f"Server resumed {final_path.name} at byte {range_start}, expected {resume_bytes}."
                )
            is_resumed = resume_bytes > 0 and status_code == 206
            write_mode = "ab" if is_resumed else "wb"
            starting_bytes = resume_bytes if is_resumed else 0
            file_total = _response_file_size(response, starting_bytes)
            total_for_progress = expected_total or completed_bytes + file_total
            downloaded = starting_bytes
            with tmp_path.open(write_mode) as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if progress:
                        progress(completed_bytes + downloaded, total_for_progress)

        actual_file_size = tmp_path.stat().st_size
        if file_total and actual_file_size != file_total:
            raise RuntimeError(
                f"Incomplete file {final_path.name}: {actual_file_size} bytes instead of {file_total}."
            )
        os.replace(tmp_path, final_path)
        completed_bytes += final_path.stat().st_size

    actual_size = sum(path.stat().st_size for path in final_paths if path.exists())
    if expected_total and actual_size != expected_total:
        raise RuntimeError(
            f"Incomplete download for {preset.label}: {actual_size} bytes instead of {expected_total}."
        )
    if progress:
        progress(actual_size, expected_total or actual_size)
    return final_paths[0]


def _ollama_model_status(model_key: str) -> dict[str, Any]:
    preset = get_model_preset(model_key)
    ollama_model = preset.ollama_model or model_key
    error = ""
    try:
        info = ollama_model_info(ollama_model)
    except RuntimeError as exc:
        info = None
        error = str(exc)
    size = int((info or {}).get("size") or 0)
    downloaded = info is not None
    path = f"ollama://{ollama_model}"
    return {
        **preset.to_dict(),
        "downloaded": downloaded,
        "available": downloaded,
        "path": path,
        "file_count": 1 if downloaded else 0,
        "files": [
            {
                "filename": ollama_model,
                "path": path,
                "exists": downloaded,
                "bytes": size,
                "partial_bytes": 0,
            }
        ],
        "bytes_downloaded": size,
        "bytes_total": size,
        "percent": 100 if downloaded else 0,
        "error": error,
    }


def _content_length(response) -> int:
    header_value = response.headers.get("Content-Length")
    if header_value and header_value.isdigit():
        return int(header_value)
    return 0


def _response_file_size(response: Any, starting_bytes: int) -> int:
    content_range = response.headers.get("Content-Range", "")
    range_match = re.search(r"/(\d+)$", content_range)
    if range_match:
        return int(range_match.group(1))
    length = _content_length(response)
    return starting_bytes + length if length else 0


def _content_range_start(response: Any) -> int | None:
    content_range = response.headers.get("Content-Range", "")
    match = re.match(r"bytes\s+(\d+)-", content_range, flags=re.I)
    return int(match.group(1)) if match else None
