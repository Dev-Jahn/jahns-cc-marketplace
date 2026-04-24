"""Wandb summary extraction with pointer-file + log-scan fallback.

Contract:
  - `read_via_pointer(run_dir)` is the authoritative path. It reads
    `runs/{id}/wandb_pointer.json` (written by train.py's AR-SAVE block
    *before* the final metric log), then reads
    `{wandb_run_dir}/files/wandb-summary.json`. Returns the parsed summary
    dict or `None` on any failure.
  - `read_from_log(log_path, wanted_keys)` is the fallback. Scans run.log for
    lines of the form `^key=<number>$` (anchored, newline-separated) and
    returns {key: float} for any matches in `wanted_keys`. Returns `None` if
    no wanted key was found at all (caller decides status=invalid).

Design note: both functions swallow IO/JSON errors and return None. The
caller (ar.cli) is responsible for translating None into status=invalid.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


_POINTER_FILENAME = "wandb_pointer.json"
_SUMMARY_RELATIVE = Path("files") / "wandb-summary.json"

# Matches: "key=123.456" or "key=1e-4" at start of line, whitespace trimmed.
# We accept only pure numeric RHS to avoid picking up sentence fragments.
_LOG_METRIC_RE = re.compile(
    r"^\s*([A-Za-z0-9_./-]+)\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)


def read_via_pointer(run_dir: Path) -> dict | None:
    """Load wandb-summary.json via the pointer file written by AR-SAVE.

    Returns the parsed summary dict, or None if any step fails.
    """
    try:
        run_dir = Path(run_dir)
        pointer_path = run_dir / _POINTER_FILENAME
        if not pointer_path.is_file():
            return None
        with open(pointer_path, "r", encoding="utf-8") as f:
            pointer = json.load(f)
        wandb_run_dir = pointer.get("wandb_run_dir")
        if not wandb_run_dir:
            return None
        summary_path = Path(wandb_run_dir) / _SUMMARY_RELATIVE
        if not summary_path.is_file():
            return None
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        if not isinstance(summary, dict):
            return None
        return summary
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def read_pointer(run_dir: Path) -> dict | None:
    """Return the parsed pointer file (wandb_run_id + wandb_run_dir) or None."""
    try:
        path = Path(run_dir) / _POINTER_FILENAME
        if not path.is_file():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def read_from_log(log_path: Path, wanted_keys: list[str]) -> dict | None:
    """Scan run.log for `^key=<number>$` matches, return last-seen per key.

    If no wanted key was found at all, returns None (caller signals invalid).
    If at least one wanted key was found, returns a dict with that subset.
    """
    try:
        log_path = Path(log_path)
        if not log_path.is_file():
            return None
        wanted = set(wanted_keys)
        found: dict[str, float] = {}
        # Scan line-by-line; last occurrence wins (metric may be logged many times).
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _LOG_METRIC_RE.match(line)
                if not m:
                    continue
                key, val = m.group(1), m.group(2)
                if key in wanted:
                    try:
                        found[key] = float(val)
                    except ValueError:
                        continue
        return found if found else None
    except OSError:
        return None
