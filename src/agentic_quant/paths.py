from __future__ import annotations

import os
from pathlib import Path


def workspace_root() -> Path:
    configured = os.environ.get("AGENTIC_QUANT_HOME", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()

    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").is_file():
        return source_root
    return Path.cwd().resolve()
