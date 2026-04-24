"""Metric reader — dispatches to the correct backend based on session config.

This module replaces v0.2's `wandb_reader.py` as the authoritative entry point
for `ar run`'s metric-extraction step. `wandb_reader.py` now re-exports the
wandb-specific helpers here for backward compatibility with any external caller
that imported them by their historical names.

Public API (used by `cli.cmd_run`):
  - `read_via_pointer(run_dir)` — wandb pointer → wandb-summary.json
  - `read_pointer(run_dir)` — parsed pointer file (for wandb_run_id extraction)
  - `read_from_log(log_path, wanted_keys)` — log-scan fallback
  - `read_tensorboard(run_dir, events_glob=None)` — tbparse / EventAccumulator
    fallback, returns dict[str, float] of last-seen values per tag, or None.
  - `tail_file(path, max_bytes)` — last N bytes of a file (for custom backend
    snippet inputs).

Design notes:
  - wandb + log + tb readers all swallow IO errors and return None so the
    caller (ar.cli) can translate absence into `status=invalid` uniformly.
  - The tb backend imports tbparse / tensorboard lazily so projects not using
    tensorboard pay nothing for the extra dep.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_POINTER_FILENAME = "wandb_pointer.json"
_SUMMARY_RELATIVE = Path("files") / "wandb-summary.json"

# Matches: "key=123.456" or "key=1e-4" at start of line, whitespace trimmed.
# We accept only pure numeric RHS to avoid picking up sentence fragments.
_LOG_METRIC_RE = re.compile(
    r"^\s*([A-Za-z0-9_./-]+)\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)


# ------------------------------- wandb -----------------------------------


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


# --------------------------- tensorboard ---------------------------------


def read_tensorboard(
    run_dir: Path,
    events_glob: str | None = None,
) -> dict[str, float] | None:
    """Read tensorboard events files under `run_dir` (or an explicit glob).

    Returns a dict of `{tag: last_scalar_value}` across all events files, or
    None if no scalars could be read. The glob defaults to
    `{run_dir}/events.out.tfevents.*` and supports `**` recursion.

    Resolution order:
      1. tbparse (preferred; handles multi-run dirs cleanly, pandas dep)
      2. tensorboard.backend.event_processing.event_accumulator (fallback)
      3. None if neither is installed or no events found.
    """
    run_dir = Path(run_dir)
    pattern = events_glob or "events.out.tfevents.*"

    # Expand glob: accept absolute / relative-to-run_dir patterns uniformly.
    candidates: list[Path]
    if Path(pattern).is_absolute():
        # absolute pattern — use glob against filesystem root
        import glob as _glob

        candidates = [Path(p) for p in _glob.glob(pattern, recursive=True) if Path(p).is_file()]
    else:
        # relative glob: tried against run_dir first, then recursively.
        candidates = [p for p in run_dir.glob(pattern) if p.is_file()]
        if not candidates:
            # recursive fallback — handles `lightning_logs/*/events.out.tfevents.*`
            recursive = pattern if "**" in pattern else f"**/{pattern}"
            candidates = [p for p in run_dir.glob(recursive) if p.is_file()]

    if not candidates:
        return None

    # Attempt tbparse first.
    try:
        from tbparse import SummaryReader  # type: ignore

        # SummaryReader handles both single-file and directory inputs. Passing
        # the run_dir (parent) is the most forgiving form.
        try:
            reader = SummaryReader(str(run_dir), extra_columns={"dir_name"})
            df = reader.scalars
            if df is None or df.empty:
                return None
            # Group by tag, take last value chronologically (by step).
            out: dict[str, float] = {}
            for tag, group in df.groupby("tag"):
                try:
                    last = group.sort_values("step").iloc[-1]
                    out[str(tag)] = float(last["value"])
                except Exception:
                    continue
            return out or None
        except Exception:
            # Fall through to EventAccumulator path.
            pass
    except ImportError:
        pass

    # Fallback: EventAccumulator.
    try:
        from tensorboard.backend.event_processing.event_accumulator import (  # type: ignore
            EventAccumulator,
        )

        out: dict[str, float] = {}
        for ev_path in candidates:
            try:
                ea = EventAccumulator(str(ev_path))
                ea.Reload()
                for tag in ea.Tags().get("scalars", []):
                    events = ea.Scalars(tag)
                    if not events:
                        continue
                    # EventAccumulator yields events in wall-order; take last.
                    out[tag] = float(events[-1].value)
            except Exception:
                continue
        return out or None
    except ImportError:
        return None


# ----------------------------- custom ------------------------------------


def tail_file(path: Path, max_bytes: int = 64 * 1024) -> str:
    """Return the last `max_bytes` of a text file. Empty string on failure."""
    try:
        path = Path(path)
        if not path.is_file():
            return ""
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


# ------------------------- auto-detection --------------------------------


def detect_backend(project_root: Path) -> str:
    """Probe `project_root` for metric-backend artifacts. Returns one of
    `wandb` | `tensorboard` | `custom` | `log`.

    Heuristic order (first hit wins):
      - `wandb/run-*/` directory        -> wandb
      - `events.out.tfevents.*` anywhere (non-recursive at root, then
        common subdirs like `lightning_logs/`, `runs/`, `logs/`) -> tensorboard
      - `mlruns/` directory             -> custom (MLflow; user fills snippet)
      - default                         -> log
    """
    project_root = Path(project_root)

    # wandb
    wandb_root = project_root / "wandb"
    if wandb_root.is_dir():
        for child in wandb_root.iterdir():
            if child.is_dir() and child.name.startswith("run-"):
                return "wandb"
        # Also accept `wandb/latest-run` symlink variant.
        if (wandb_root / "latest-run").exists():
            return "wandb"

    # tensorboard — check root + common log directories
    tb_dirs = [project_root, project_root / "lightning_logs", project_root / "runs", project_root / "logs", project_root / "tb_logs"]
    for d in tb_dirs:
        if not d.is_dir():
            continue
        # shallow check + one-level-deep scan (lightning_logs/version_0/events.*)
        for p in d.glob("events.out.tfevents.*"):
            if p.is_file():
                return "tensorboard"
        for p in d.glob("*/events.out.tfevents.*"):
            if p.is_file():
                return "tensorboard"

    # mlflow — punted to v0.4, mark as custom with TODO
    if (project_root / "mlruns").is_dir():
        return "custom"

    return "log"


def detect_distributed_framework(
    project_root: Path,
    entry_main_module: str | None = None,
) -> str:
    """Probe pyproject.toml dependencies + entry script head for framework.

    Returns one of `accelerate | deepspeed | fsdp | ddp | lightning | none`.
    """
    project_root = Path(project_root)

    # 1. pyproject.toml scan (string match, we don't need real toml parsing
    # for this cheap heuristic and want to avoid pulling tomllib on older py).
    pyproject = project_root / "pyproject.toml"
    deps_text = ""
    if pyproject.is_file():
        try:
            deps_text = pyproject.read_text(encoding="utf-8", errors="replace")
        except OSError:
            deps_text = ""

    # Also consult requirements.txt and setup.cfg for older projects.
    for extra in ("requirements.txt", "requirements-dev.txt", "setup.cfg"):
        p = project_root / extra
        if p.is_file():
            try:
                deps_text += "\n" + p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass

    deps_lower = deps_text.lower()

    # 2. entry script head probe — look for explicit imports.
    entry_head = ""
    if entry_main_module:
        # Translate dotted module -> path candidates.
        dotted = entry_main_module.replace(".", "/")
        for candidate in (
            project_root / f"{dotted}.py",
            project_root / dotted / "__init__.py",
            project_root / "src" / f"{dotted}.py",
            project_root / "src" / dotted / "__init__.py",
        ):
            if candidate.is_file():
                try:
                    with open(candidate, "r", encoding="utf-8", errors="replace") as f:
                        # Read only the first ~200 lines; imports live at top.
                        lines: list[str] = []
                        for i, ln in enumerate(f):
                            if i >= 200:
                                break
                            lines.append(ln)
                        entry_head = "".join(lines)
                    break
                except OSError:
                    continue

    head_lower = entry_head.lower()

    # Decision cascade: prefer the most specific framework when multiple are
    # present (deepspeed / lightning win over generic ddp).
    if "pytorch_lightning" in deps_lower or "lightning" in deps_lower:
        if "import pytorch_lightning" in head_lower or "import lightning" in head_lower or "from pytorch_lightning" in head_lower or "from lightning" in head_lower:
            return "lightning"
    if "deepspeed" in deps_lower and ("import deepspeed" in head_lower or "from deepspeed" in head_lower or not entry_head):
        return "deepspeed"
    if "accelerate" in deps_lower and ("from accelerate" in head_lower or "import accelerate" in head_lower or not entry_head):
        return "accelerate"
    # FSDP is part of torch.distributed.fsdp
    if "torch.distributed.fsdp" in head_lower or "fully_shard" in head_lower or "fullyshardeddataparallel" in head_lower:
        return "fsdp"
    if "torch.distributed" in head_lower or "distributeddataparallel" in head_lower:
        return "ddp"

    # Fallback cascade using dep presence only when entry head was empty.
    if not entry_head:
        if "pytorch_lightning" in deps_lower or ("lightning" in deps_lower and "pytorch" not in deps_lower):
            return "lightning"
        if "accelerate" in deps_lower:
            return "accelerate"
        if "deepspeed" in deps_lower:
            return "deepspeed"

    return "none"
