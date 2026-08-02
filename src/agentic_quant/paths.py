from __future__ import annotations

import os
from pathlib import Path


def workspace_root() -> Path:
    """Return the writable root used for models and experiment runs.

    Editable/source installs keep the historical repository-local layout. A
    regular wheel install falls back to the caller's working directory, while
    ``AGENTIC_QUANT_HOME`` offers an explicit stable location for services.
    """
    configured = os.environ.get("AGENTIC_QUANT_HOME", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()

    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").is_file():
        return source_root
    return Path.cwd().resolve()
