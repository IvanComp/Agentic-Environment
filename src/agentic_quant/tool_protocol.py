from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .tools import try_parse_json


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    text: str
    tool_calls: list[ToolCall]
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float
    model: str
    native_tools: bool
    malformed: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


def tool_schema(name: str, description: str, parameters: dict[str, str] | None = None) -> dict[str, Any]:
    properties = {
        key: {"type": "string", "description": value}
        for key, value in (parameters or {}).items()
    }
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
            },
        },
    }


def assistant_message(text: str, tool_calls: list[ToolCall]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = [
            {"type": "function", "function": {"name": call.name, "arguments": call.arguments}}
            for call in tool_calls
        ]
    return message


def tool_message(name: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_name": name, "content": content}


def text_protocol_instructions(tools: list[dict[str, Any]]) -> str:
    lines = ["You can call the following tools:"]
    for tool in tools:
        function = tool["function"]
        params = function["parameters"]["properties"]
        signature = ", ".join(params)
        lines.append(f"- {function['name']}({signature}): {function['description']}")
        for key, spec in params.items():
            lines.append(f"    {key}: {spec['description']}")
    lines.append("")
    lines.append(
        "To call a tool, reply with one JSON object and nothing else: "
        '{"message": "optional text for this step", "tool": "tool_name", "arguments": {...}}'
    )
    return "\n".join(lines)


def to_text_protocol(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[dict[str, str]]:
    instructions = text_protocol_instructions(tools)
    rendered: list[dict[str, str]] = []
    for message in messages:
        role = message["role"]
        if role == "system":
            rendered.append({"role": "system", "content": f"{message['content']}\n\n{instructions}"})
        elif role == "assistant":
            calls = message.get("tool_calls") or []
            if calls:
                call = calls[0]["function"]
                content = json.dumps(
                    {"message": message.get("content") or "", "tool": call["name"], "arguments": call["arguments"]},
                    ensure_ascii=False,
                )
            else:
                content = str(message.get("content") or "")
            rendered.append({"role": "assistant", "content": content})
        elif role == "tool":
            rendered.append(
                {"role": "user", "content": f"Tool output ({message['tool_name']}):\n{message['content']}"}
            )
        else:
            rendered.append({"role": role, "content": str(message.get("content") or "")})
    if not rendered or rendered[0]["role"] != "system":
        rendered.insert(0, {"role": "system", "content": instructions})
    return rendered


def parse_text_tool_call(text: str) -> tuple[str, dict[str, Any], str] | None:
    parsed = try_parse_json(text)
    if parsed and parsed.get("tool"):
        arguments = parsed.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if not isinstance(arguments, dict):
            return None
        return str(parsed["tool"]), arguments, str(parsed.get("message") or "")

    function_call = re.search(
        r"\b([a-z][a-z0-9_]*)\s*\(\s*(\{.*?\})?\s*\)",
        text,
        flags=re.I | re.S,
    )
    if not function_call:
        return None
    raw_arguments = function_call.group(2) or "{}"
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return None
    message_match = re.search(r"\bmessage\s*:\s*[\"'](.*?)[\"']", text, flags=re.I | re.S)
    message = message_match.group(1) if message_match else ""
    return function_call.group(1), arguments, message


def text_response_to_calls(text: str) -> tuple[str, list[ToolCall], bool]:
    stripped = text.strip()
    action = parse_text_tool_call(stripped)
    if action is None:
        return stripped, [], True
    name, arguments, message = action
    return message, [ToolCall(name, arguments)], False
