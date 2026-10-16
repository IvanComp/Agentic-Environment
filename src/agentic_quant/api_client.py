from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class QuotaExhausted(RuntimeError):
    pass


def api_base_url() -> str:
    return os.environ.get("AGENTIC_API_BASE", "https://openrouter.ai/api/v1").strip().rstrip("/")


def api_key() -> str:
    return (os.environ.get("AGENTIC_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or "").strip()


def to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    pending: list[str] = []
    for index, message in enumerate(messages):
        role = message["role"]
        if role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
            calls = message.get("tool_calls") or []
            if calls:
                pending = []
                entry["tool_calls"] = []
                for position, call in enumerate(calls):
                    function = call.get("function") or {}
                    call_id = call.get("id") or f"call_{index}_{position}"
                    pending.append(call_id)
                    arguments = function.get("arguments")
                    entry["tool_calls"].append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": function.get("name") or "",
                                "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments or {}),
                            },
                        }
                    )
            converted.append(entry)
        elif role == "tool":
            call_id = pending.pop(0) if pending else f"call_{index}"
            converted.append({"role": "tool", "tool_call_id": call_id, "content": str(message.get("content") or "")})
        else:
            converted.append({"role": role, "content": str(message.get("content") or "")})
    return converted


def api_chat(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float | None,
    seed: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    timeout: float = 900,
    attempts: int = 8,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": to_openai_messages(messages),
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if seed is not None:
        payload["seed"] = seed
    if tools:
        payload["tools"] = tools
    delay = 5.0
    for attempt in range(attempts):
        try:
            result = _post("/chat/completions", payload, timeout=timeout)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 400 and "temperature" in body and "temperature" in payload:
                payload.pop("temperature")
                continue
            if exc.code == 400 and "seed" in body and "seed" in payload:
                payload.pop("seed")
                continue
            if exc.code in RETRY_STATUS and attempt < attempts - 1:
                time.sleep(_retry_after(exc) or delay)
                delay = min(delay * 2, 300)
                continue
            if exc.code == 429:
                raise QuotaExhausted(f"{model}: rate limit or daily quota reached: {body[:300]}") from exc
            raise RuntimeError(f"{model}: HTTP {exc.code}: {body[:500]}") from exc
        except (URLError, TimeoutError, ConnectionError) as exc:
            if attempt < attempts - 1:
                time.sleep(delay)
                delay = min(delay * 2, 300)
                continue
            raise RuntimeError(f"{model}: {api_base_url()} unreachable: {exc}") from exc
        if result.get("error"):
            error = result["error"]
            code = error.get("code") if isinstance(error, dict) else None
            if code in RETRY_STATUS and attempt < attempts - 1:
                time.sleep(delay)
                delay = min(delay * 2, 300)
                continue
            raise RuntimeError(f"{model}: {error}")
        if not result.get("choices"):
            if attempt < attempts - 1:
                time.sleep(delay)
                continue
            raise RuntimeError(f"{model}: empty response")
        _check_served_model(model, str(result.get("model") or ""))
        return result
    raise RuntimeError(f"{model}: no answer after {attempts} attempts")


def response_message(result: dict[str, Any]) -> dict[str, Any]:
    return (result.get("choices") or [{}])[0].get("message") or {}


def response_thinking(message: dict[str, Any]) -> str:
    return str(message.get("reasoning") or message.get("reasoning_content") or "").strip()


def _check_served_model(requested: str, served: str) -> None:
    if not served:
        return
    wanted = requested.split(":")[0].split("/")[-1].lower()
    if wanted not in served.lower():
        raise RuntimeError(f"Requested {requested} but the endpoint answered with {served}")


def _retry_after(exc: HTTPError) -> float:
    value = exc.headers.get("Retry-After") if exc.headers else None
    try:
        return min(float(value), 600) if value else 0
    except ValueError:
        return 0


def _post(path: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "User-Agent": "agentic-quant/0.2"}
    key = api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = Request(f"{api_base_url()}{path}", data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise RuntimeError(f"Invalid response from {api_base_url()}{path}")
    return result
