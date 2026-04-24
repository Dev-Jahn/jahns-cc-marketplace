"""Per-expr history I/O: run ids, results.tsv, snapshots, restores.

Contract:
  - `next_run_id(expr_dir)` returns the next monotonic id (r0001, r0002, ...)
    based on existing `runs/r*` directories. Purely name-based; never reads
    result.json.
  - `append_tsv(expr_dir, row)` appends one row to `results.tsv`, creating
    the file + header if missing. Free-text fields are sanitized per spec §F:
    tab/CR/LF -> single space, truncate at 200 chars, prefix `"~ "` if the
    value was modified. Structured columns are asserted to be clean and
    never rewritten.
  - `snapshot_train_py(expr_dir, run_id)` copies the current working
    `train.py` into `runs/{run_id}/train.py` *before* the run starts.
  - `restore_train_py_from_best(expr_dir)` copies
    `runs/{best_id}/train.py` back onto the working `train.py`. If no
    best exists yet, falls back to `.baseline/train.py` (captured by
    `ar init`), re-asserting the spec's invariant "revert restores to last
    best".

Spec references: §results.tsv header, §F (sanitization), §D (unified revert).
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any


RESULTS_HEADER = [
    "run_id",
    "ts",
    "wall_s",
    "status",
    "verdict",
    "primary_name",
    "primary_value",
    "valid",
    "improved",
    "note",
]

# Columns that are free-text and must be sanitized before write. Anything
# else is asserted to have no tab/newline; if it does we refuse to write
# rather than corrupt the table.
_FREE_TEXT_COLUMNS = {"note"}

_RUN_DIR_RE = re.compile(r"^r(\d{4,})$")


# --------------------------- run-id allocation ---------------------------


def next_run_id(expr_dir: Path) -> str:
    """Return next monotonic run id — r0001, r0002, ..., r9999, r10000."""
    runs_dir = Path(expr_dir) / "runs"
    if not runs_dir.is_dir():
        return "r0001"
    highest = 0
    for entry in runs_dir.iterdir():
        if not entry.is_dir():
            continue
        m = _RUN_DIR_RE.match(entry.name)
        if not m:
            continue
        try:
            n = int(m.group(1))
        except ValueError:
            continue
        highest = max(highest, n)
    return f"r{highest + 1:04d}"


def list_run_ids(expr_dir: Path) -> list[str]:
    """Return all run ids in natural order."""
    runs_dir = Path(expr_dir) / "runs"
    if not runs_dir.is_dir():
        return []
    ids = []
    for entry in runs_dir.iterdir():
        if entry.is_dir() and _RUN_DIR_RE.match(entry.name):
            ids.append(entry.name)
    ids.sort(key=lambda s: int(s[1:]))
    return ids


# --------------------------- TSV I/O + sanitize --------------------------


def _sanitize_free_text(value: Any) -> tuple[str, bool]:
    """Replace tab/CR/LF with space, truncate to 200 chars.

    Returns (sanitized_value, was_touched).
    """
    s = "" if value is None else str(value)
    replaced = s.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    truncated = replaced[:200]
    touched = (truncated != s)
    return truncated, touched


def _assert_structured_clean(column: str, value: Any) -> str:
    """Structured column must be free of tab/CR/LF. Raise if not."""
    s = "" if value is None else str(value)
    if any(ch in s for ch in ("\t", "\r", "\n")):
        raise ValueError(
            f"structured TSV column {column!r} contains whitespace control "
            f"char; refusing to write corrupt TSV. value={s!r}"
        )
    return s


def append_tsv(expr_dir: Path, row: dict[str, Any]) -> None:
    """Append one row to `results.tsv`, creating file + header if missing."""
    expr_dir = Path(expr_dir)
    tsv_path = expr_dir / "results.tsv"
    need_header = not tsv_path.is_file() or tsv_path.stat().st_size == 0

    cells: list[str] = []
    for col in RESULTS_HEADER:
        raw = row.get(col, "")
        if col in _FREE_TEXT_COLUMNS:
            cleaned, touched = _sanitize_free_text(raw)
            if touched:
                cleaned = "~ " + cleaned
            cells.append(cleaned)
        else:
            cells.append(_assert_structured_clean(col, raw))

    line = "\t".join(cells) + "\n"
    with open(tsv_path, "a", encoding="utf-8") as f:
        if need_header:
            f.write("\t".join(RESULTS_HEADER) + "\n")
        f.write(line)


def read_results_tsv(expr_dir: Path) -> list[dict[str, str]]:
    """Return results.tsv as a list of dicts (header-indexed)."""
    expr_dir = Path(expr_dir)
    tsv_path = expr_dir / "results.tsv"
    if not tsv_path.is_file():
        return []
    rows: list[dict[str, str]] = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        header: list[str] | None = None
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            cells = line.split("\t")
            if header is None:
                header = cells
                continue
            if len(cells) < len(header):
                cells = cells + [""] * (len(header) - len(cells))
            rows.append({k: v for k, v in zip(header, cells)})
    return rows


# ------------------------ train.py snapshot/restore ----------------------


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy src -> dst atomically via same-dir tempfile + rename."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=dst.name + ".", suffix=".tmp", dir=str(dst.parent)
    )
    os.close(fd)
    try:
        shutil.copyfile(src, tmp)
        # fsync for durability before rename.
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.rename(tmp, dst)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def snapshot_train_py(expr_dir: Path, run_id: str) -> Path:
    """Copy current working train.py into runs/{run_id}/train.py."""
    expr_dir = Path(expr_dir)
    src = expr_dir / "train.py"
    dst = expr_dir / "runs" / run_id / "train.py"
    if not src.is_file():
        raise FileNotFoundError(f"train.py not found at {src}")
    _atomic_copy(src, dst)
    return dst


def snapshot_baseline(expr_dir: Path) -> Path:
    """Snapshot initial train.py into .baseline/train.py.

    Called exactly once by `ar init` so we have a fallback restore target
    when no run has yet advanced.
    """
    expr_dir = Path(expr_dir)
    src = expr_dir / "train.py"
    dst = expr_dir / ".baseline" / "train.py"
    if not src.is_file():
        raise FileNotFoundError(f"train.py not found at {src}")
    _atomic_copy(src, dst)
    return dst


def restore_train_py_from_best(expr_dir: Path) -> str:
    """Restore working train.py from runs/{best_id}/train.py, else baseline.

    Returns a short description of what happened ("best:r0037" or "baseline"
    or "noop" if neither source exists).
    """
    from .schemas import best_record_from_dict, load_json

    expr_dir = Path(expr_dir)
    best_path = expr_dir / "best.json"
    dst = expr_dir / "train.py"

    if best_path.is_file():
        try:
            best = best_record_from_dict(load_json(best_path))
            src = expr_dir / "runs" / best.run_id / "train.py"
            if src.is_file():
                _atomic_copy(src, dst)
                return f"best:{best.run_id}"
        except (OSError, ValueError, KeyError):
            pass

    baseline = expr_dir / ".baseline" / "train.py"
    if baseline.is_file():
        _atomic_copy(baseline, dst)
        return "baseline"

    return "noop"
