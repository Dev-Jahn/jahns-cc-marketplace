"""Data models for result/best/session files.

Contract:
  - Every file the agent or helper reads has a typed model here.
  - Serialization goes through `dump_json` (atomic: tempfile + fsync + rename).
  - `RunResult` enforces the status x verdict legal-combination table from the
    design spec.

Dependencies: stdlib only — we deliberately avoid pydantic to keep the plugin
dependency surface minimal. If we later want strict validation we can swap in
pydantic v2 without changing callers (the public API is load_json/dump_json +
dataclass-style fields).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


# --- Legal status x verdict combinations (spec §Status x verdict table) ----

_LEGAL_STATUS_VERDICT: dict[str, set[str]] = {
    "ok": {"advance", "revert"},
    "invalid": {"revert"},
    "crash": {"crash"},
    "timeout": {"timeout"},
    "interrupted": {"interrupted"},
    "unclean": {"crash"},
}


def validate_status_verdict(status: str, verdict: str) -> None:
    allowed = _LEGAL_STATUS_VERDICT.get(status)
    if allowed is None:
        raise ValueError(f"unknown status: {status!r}")
    if verdict not in allowed:
        raise ValueError(
            f"illegal verdict {verdict!r} for status {status!r}; "
            f"allowed = {sorted(allowed)}"
        )


# ----------------------------- Core models --------------------------------


@dataclass
class Primary:
    """Primary metric descriptor + observed value."""

    name: str
    direction: str  # "min" | "max"
    value: float | None = None

    def __post_init__(self) -> None:
        if self.direction not in ("min", "max"):
            raise ValueError(f"direction must be 'min' or 'max', got {self.direction!r}")


@dataclass
class Constraint:
    """Hard constraint specification."""

    name: str
    op: str  # "<=", "<", ">=", ">"
    threshold: float

    def __post_init__(self) -> None:
        if self.op not in ("<=", "<", ">=", ">"):
            raise ValueError(f"invalid constraint op: {self.op!r}")


@dataclass
class ConstraintResult:
    name: str
    op: str
    threshold: float
    value: float | None
    passed: bool


@dataclass
class Metrics:
    """Thin wrapper around a dict of metric_name -> float.

    Exists to keep type hints clean; we serialize as a flat dict.
    """

    values: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, float]:
        return dict(self.values)

    @classmethod
    def from_json(cls, obj: dict[str, float] | None) -> "Metrics":
        return cls(values=dict(obj or {}))


@dataclass
class RunResult:
    """Structured summary persisted to runs/{run_id}/result.json."""

    run_id: str
    started_at: str
    wall_seconds: float
    status: str  # ok | crash | timeout | invalid | interrupted | unclean
    verdict: str  # advance | revert | invalid | crash | timeout | interrupted
    metrics: dict[str, float]
    primary: dict[str, Any]  # serialized Primary
    constraints: list[dict[str, Any]]  # serialized ConstraintResult list
    valid: bool
    improved_over_best: bool
    previous_best_run_id: str | None
    wandb_run_id: str | None
    log_path: str
    note: str
    should_terminate: bool
    terminated_by: str | None  # null | primary_threshold | plateau | max_runs | manual | chain_budget
    exit_code: int | None = None
    unclean: bool = False

    def __post_init__(self) -> None:
        validate_status_verdict(self.status, self.verdict)


@dataclass
class BestRecord:
    """Contents of {expr}/best.json."""

    run_id: str
    primary: dict[str, Any]
    created_at: str
    ckpt_path: str = "best_ckpt/state.pt"


@dataclass
class Session:
    """Per-expr session settings — `.autoresearch/{expr}/.ar-session.json`."""

    expr: str
    duration_seconds: int
    termination_conditions: dict[str, Any]  # {primary_threshold: float|None, plateau_n: int|None, max_runs: int|None, unlimited: bool}
    hard_constraints: list[dict[str, Any]]
    runner: dict[str, Any]  # {kind: "accelerate"|"torchrun"|"python"|"custom", config_file: str|None, num_processes: int|None, extra_args: list[str]}
    chain_mode: str  # "disabled" | "max_n" | "unlimited"
    created_at: str
    baseline_acknowledged: bool = False
    resume_mode: str | None = None  # "continue" | "fresh" | None
    # v0.3.0: metric backend + distributed framework metadata. Kept separate
    # from runner.kind (launcher shape) so the same torchrun runner can pair
    # with any of accelerate/deepspeed/fsdp/ddp/lightning/none semantics.
    metric_backend: str = "wandb"  # wandb | tensorboard | log | custom
    distributed_framework: str = "accelerate"  # accelerate | deepspeed | fsdp | ddp | lightning | none


@dataclass
class ChainSession:
    """Top-level chain scope state — `.autoresearch/.chain-session.json`."""

    chain_mode: str  # "disabled" | "max_n" | "unlimited"
    chain_remaining: int | None  # null for unlimited
    chain_trail: list[str]  # ordered slugs
    total_wall_time_budget_s: int | None
    current_slug: str | None
    created_at: str


@dataclass
class ChainDecision:
    """Read-only audit artifact per chain transition."""

    parent_expr: str
    parent_primary: dict[str, Any]
    child_primary: dict[str, Any]
    child_goal: str
    child_mutation_scope: list[str]
    rationale: str
    parent_report_sha256: str
    created_at: str
    chain_position: int


@dataclass
class ProgramMeta:
    """Parsed view of program.md front-matter / key fields.

    We render program.md as markdown with a small YAML-ish header so we can
    recover structured fields for decisions (parent_ckpt, primary metric,
    runner spec). This model captures those.
    """

    expr_slug: str
    goal: str
    mutation_scope: list[str]
    primary_name: str
    primary_direction: str
    auxiliary: list[str]
    hard_constraints: list[dict[str, Any]]
    runner: dict[str, Any]
    per_run_seconds: int
    parent_expr: str | None = None
    parent_ckpt: str | None = None
    metric_backend: str = "wandb"
    distributed_framework: str = "accelerate"
    checkpoint_glob: str | None = None


# ------------------------- JSON I/O helpers -------------------------------


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def dump_json(path: Path, data: Any, *, indent: int = 2) -> None:
    """Atomic JSON write: tempfile in same dir + fsync + rename.

    Same-directory tempfile guarantees `os.rename` is atomic on POSIX.
    fsync ensures data is on disk before the rename commits the new name.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_to_jsonable(data), indent=indent, ensure_ascii=False)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_name, path)
        # Best-effort directory fsync so the rename is durable.
        try:
            dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------- Constructors from dicts --------------------------


def run_result_from_dict(d: dict[str, Any]) -> RunResult:
    return RunResult(
        run_id=d["run_id"],
        started_at=d["started_at"],
        wall_seconds=float(d["wall_seconds"]),
        status=d["status"],
        verdict=d["verdict"],
        metrics=dict(d.get("metrics") or {}),
        primary=dict(d["primary"]),
        constraints=list(d.get("constraints") or []),
        valid=bool(d.get("valid", False)),
        improved_over_best=bool(d.get("improved_over_best", False)),
        previous_best_run_id=d.get("previous_best_run_id"),
        wandb_run_id=d.get("wandb_run_id"),
        log_path=d.get("log_path", ""),
        note=d.get("note", ""),
        should_terminate=bool(d.get("should_terminate", False)),
        terminated_by=d.get("terminated_by"),
        exit_code=d.get("exit_code"),
        unclean=bool(d.get("unclean", False)),
    )


def best_record_from_dict(d: dict[str, Any]) -> BestRecord:
    return BestRecord(
        run_id=d["run_id"],
        primary=dict(d["primary"]),
        created_at=d["created_at"],
        ckpt_path=d.get("ckpt_path", "best_ckpt/state.pt"),
    )


def session_from_dict(d: dict[str, Any]) -> Session:
    # v0.3.0: backward compat — pre-0.3 sessions have no metric_backend /
    # distributed_framework fields, so default them on load.
    return Session(
        expr=d["expr"],
        duration_seconds=int(d["duration_seconds"]),
        termination_conditions=dict(d.get("termination_conditions") or {}),
        hard_constraints=list(d.get("hard_constraints") or []),
        runner=dict(d["runner"]),
        chain_mode=d.get("chain_mode", "disabled"),
        created_at=d["created_at"],
        baseline_acknowledged=bool(d.get("baseline_acknowledged", False)),
        resume_mode=d.get("resume_mode"),
        metric_backend=d.get("metric_backend") or "wandb",
        distributed_framework=d.get("distributed_framework") or "accelerate",
    )


def chain_session_from_dict(d: dict[str, Any]) -> ChainSession:
    return ChainSession(
        chain_mode=d.get("chain_mode", "disabled"),
        chain_remaining=d.get("chain_remaining"),
        chain_trail=list(d.get("chain_trail") or []),
        total_wall_time_budget_s=d.get("total_wall_time_budget_s"),
        current_slug=d.get("current_slug"),
        created_at=d["created_at"],
    )


def chain_decision_from_dict(d: dict[str, Any]) -> ChainDecision:
    return ChainDecision(
        parent_expr=d["parent_expr"],
        parent_primary=dict(d["parent_primary"]),
        child_primary=dict(d["child_primary"]),
        child_goal=d["child_goal"],
        child_mutation_scope=list(d["child_mutation_scope"]),
        rationale=d["rationale"],
        parent_report_sha256=d["parent_report_sha256"],
        created_at=d["created_at"],
        chain_position=int(d["chain_position"]),
    )
