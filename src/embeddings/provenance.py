"""What produced an output: written next to every table so runs can be traced and repeated."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


def git_commit() -> str | None:
    """Commit of the checkout this code runs from, or ``None`` outside a Git work tree."""
    root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def run_provenance(arguments: Mapping[str, Any], **sections: Any) -> dict[str, Any]:
    """Describe this run: code version, environment, command line, plus caller-supplied sections."""
    from .. import __version__

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mpn_al_version": __version__,
        "git_commit": git_commit(),
        "command": sys.argv,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "arguments": {key: _plain(value) for key, value in arguments.items()},
        **sections,
    }


def write_json(path: str | Path, document: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _plain(value: Any) -> Any:
    """JSON-friendly rendering of argparse values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value
