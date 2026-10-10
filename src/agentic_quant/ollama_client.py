from __future__ import annotations

import json
import os
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ProgressCallback = Callable[[int, int], None]


def ollama_base_url() -> str:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    if "://" not in host:
        host = f"http://{host}"
    return host.rstrip("/")


def ollama_models(*, timeout: float = 3) -> list[dict[str, Any]]:
    response = _json_request("/api/tags", timeout=timeout)
    models = response.get("models", [])
    return models if isinstance(models, list) else []


def ollama_model_info(model: str, *, timeout: float = 3) -> dict[str, Any] | None:
    for item in ollama_models(timeout=timeout):
        if item.get("model") == model or item.get("name") == model:
            return item
    return None


def ollama_chat(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    seed: int,
    tools: list[dict[str, Any]] | None = None,
    num_ctx: int | None = None,
    timeout: float = 600,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "temperature": temperature,
        "seed": seed,
        "num_predict": max_tokens,
    }
    if num_ctx:
        options["num_ctx"] = num_ctx
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    if tools:
        payload["tools"] = tools
    else:
        payload["format"] = "json"
    if model.startswith("gpt-oss:"):
        payload["think"] = "low"
    return _json_request(
        "/api/chat",
        payload=payload,
        timeout=timeout,
    )


def pull_ollama_model(
    model: str,
    *,
    progress: ProgressCallback | None = None,
    timeout: float = 600,
) -> None:
    request = _request("/api/pull", {"model": model, "stream": True})
    try:
        with urlopen(request, timeout=timeout) as response:
            last_done = 0
            last_total = 0
            for raw_line in response:
                if not raw_line.strip():
                    continue
                event = json.loads(raw_line)
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                done = int(event.get("completed") or last_done)
                total = int(event.get("total") or last_total)
                last_done, last_total = done, total
                if progress and total:
                    progress(done, total)
            if progress:
                final_total = last_total or 1
                progress(final_total, final_total)
    except HTTPError as exc:
        raise _http_error(exc) from exc
    except URLError as exc:
        raise _connection_error(exc) from exc


def delete_ollama_model(model: str, *, timeout: float = 600) -> None:
    request = _request("/api/delete", {"model": model}, method="DELETE")
    try:
        with urlopen(request, timeout=timeout):
            return
    except HTTPError as exc:
        raise _http_error(exc) from exc
    except URLError as exc:
        raise _connection_error(exc) from exc


def _json_request(
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> dict[str, Any]:
    request = _request(path, payload)
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except HTTPError as exc:
        raise _http_error(exc) from exc
    except URLError as exc:
        raise _connection_error(exc) from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"Ollama returned an invalid response for {path}.")
    if result.get("error"):
        raise RuntimeError(str(result["error"]))
    return result


def _request(path: str, payload: dict[str, Any] | None, *, method: str | None = None) -> Request:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    return Request(
        f"{ollama_base_url()}{path}",
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "agentic-quant/0.2"},
        method=method or ("GET" if payload is None else "POST"),
    )


def _http_error(exc: HTTPError) -> RuntimeError:
    try:
        body = json.loads(exc.read())
        detail = body.get("error") if isinstance(body, dict) else None
    except (json.JSONDecodeError, OSError):
        detail = None
    return RuntimeError(detail or f"Ollama returned HTTP {exc.code}.")


def _connection_error(exc: URLError) -> RuntimeError:
    return RuntimeError(
        f"Ollama is not reachable at {ollama_base_url()}. Start it with `ollama serve`. ({exc.reason})"
    )
