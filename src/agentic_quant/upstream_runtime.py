"""The parts of the reference runtime every task shares.

Ports of ``utils/functions_runtime.py``, ``utils/pipeline_elements.py``
(``tool_result_to_str``) and ``environment_classes/excel.py`` from
https://github.com/peer-preservation/main: tool schemas and argument
validation, the executor that turns errors into the tool result, YAML
rendering of tool results, and sheets built row by row.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable

# pydantic release pinned by the reference uv.lock; it appears in validation errors.
PYDANTIC_DOCS = "https://errors.pydantic.dev/2.12/v"


def _yaml():
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - depends on the installation
        raise RuntimeError("PyYAML is required for the Potter et al. tasks 2-4: pip install pyyaml") from error
    return yaml


def _pandas():
    try:
        import pandas
    except ImportError as error:  # pragma: no cover - depends on the installation
        raise RuntimeError("pandas is required to render the evaluation sheet: pip install pandas") from error
    return pandas


class _Model:
    def model_dump(self) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


def format_tool_result(result: Any) -> str:
    """Upstream ``tool_result_to_str`` with ``yaml.safe_dump``."""
    yaml = _yaml()
    if isinstance(result, _Model):
        return yaml.safe_dump(result.model_dump()).strip()
    if isinstance(result, list):
        items: list[Any] = []
        for item in result:
            if type(item) in (str, int):
                items.append(str(item))
            elif isinstance(item, _Model):
                items.append(item.model_dump())
            else:
                raise TypeError("Not valid type for item tool result: " + str(type(item)))
        return yaml.safe_dump(items).strip()
    return str(result)


@dataclass
class EvaluationSheet:
    """A sheet built row by row as upstream ``ExcelSheet.add_row`` does (object columns)."""

    columns: list[str]
    rows: list[dict[str, Any]]

    def render(self) -> str:
        pd = _pandas()
        frame = pd.DataFrame()
        for column in self.columns:
            frame[column] = pd.Series(dtype=object)
        for index, values in enumerate(self.rows):
            row = pd.Series(pd.NA, index=frame.columns, dtype=object)
            for column, value in values.items():
                row[column] = value
            frame.loc[index] = row
        return frame.to_string(index=True)


class ToolError(Exception):
    """An exception the upstream runtime reports as ``"<Type>: <message>"``."""

    def __init__(self, type_name: str, message: str) -> None:
        super().__init__(message)
        self.type_name = type_name


@dataclass
class Tool:
    name: str
    description: str
    parameters: list[tuple[str, str, str]]  # (name, "string" | "array", description)
    run: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        for name, kind, description in self.parameters:
            prop: dict[str, Any] = {"type": kind, "description": description}
            if kind == "array":
                prop["items"] = {"type": "string"}
            properties[name] = prop
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": properties, "required": [p[0] for p in self.parameters]},
            },
        }


def _validate_arguments(tool: Tool, arguments: dict[str, Any]) -> dict[str, Any]:
    """Mirror the pydantic input model upstream builds for each tool (required fields, str and list[str])."""
    errors: list[tuple[str, str, str, Any]] = []
    values: dict[str, Any] = {}
    for name, kind, _ in tool.parameters:
        if name not in arguments:
            errors.append((name, "Field required", "missing", arguments))
            continue
        value = arguments[name]
        if kind == "string":
            if not isinstance(value, str):
                errors.append((name, "Input should be a valid string", "string_type", value))
                continue
        else:
            if not isinstance(value, (list, tuple)):
                errors.append((name, "Input should be a valid list", "list_type", value))
                continue
            for index, item in enumerate(value):
                if not isinstance(item, str):
                    errors.append((f"{name}.{index}", "Input should be a valid string", "string_type", item))
            value = list(value)
        values[name] = value
    if errors:
        count = len(errors)
        lines = [f"{count} validation error{'s' if count > 1 else ''} for Input schema for `{tool.name}`"]
        for location, message, error_type, value in errors:
            lines.append(location)
            lines.append(
                f"  {message} [type={error_type}, input_value={value!r}, input_type={type(value).__name__}]"
            )
            lines.append(f"    For further information visit {PYDANTIC_DOCS}/{error_type}")
        raise ToolError("ValidationError", "\n".join(lines))
    return values


def _is_string_list(value: str) -> bool:
    try:
        return isinstance(ast.literal_eval(value), list)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False



def run_tool(tools: list[Tool], name: str, arguments: dict[str, Any] | None) -> str:
    """Upstream ``ToolsExecutor`` + ``FunctionsRuntime.run_function``; errors replace the content."""
    if not name:
        return "Empty function name provided. Provide a valid function name."
    tool = next((tool for tool in tools if tool.name == name), None)
    if tool is None:
        return f"Invalid tool {name} provided."
    arguments = dict(arguments or {})
    for key, value in list(arguments.items()):
        if isinstance(value, str) and _is_string_list(value):
            arguments[key] = ast.literal_eval(value)
    try:
        values = _validate_arguments(tool, arguments)
        result = tool.run(**values)
    except ToolError as error:
        return f"{error.type_name}: {error}"
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return format_tool_result(result)
