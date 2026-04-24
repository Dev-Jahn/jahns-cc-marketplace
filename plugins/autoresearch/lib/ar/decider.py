"""Improvement / constraint / termination decisions.

Contract:
  - `is_improved(primary, best)` returns True iff `primary.value` is strictly
    better than `best.primary.value` under the primary's direction. A
    candidate is also considered "improved" when there is no prior best.
  - `check_constraints(metrics, constraints)` evaluates every constraint
    against the metrics dict; returns (all_passed, per-constraint results).
    Missing metric values count as constraint failures.
  - `check_termination(session, history)` evaluates the session's
    termination conditions against the run history; returns
    (should_terminate, terminated_by).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .schemas import BestRecord, ConstraintResult, Primary, Session


def is_improved(primary: Primary, best: BestRecord | None) -> bool:
    """True iff `primary.value` strictly improves over `best.primary.value`.

    Rules:
      - `primary.value is None` (no metric extracted) -> False.
      - `best is None` (no prior best) -> True.
      - direction=min: improved iff primary.value < best.value.
      - direction=max: improved iff primary.value > best.value.
    """
    if primary.value is None:
        return False
    if best is None:
        return True
    best_val = best.primary.get("value")
    if best_val is None:
        return True
    if primary.direction == "min":
        return float(primary.value) < float(best_val)
    if primary.direction == "max":
        return float(primary.value) > float(best_val)
    raise ValueError(f"unknown direction: {primary.direction!r}")


def _cmp(op: str, value: float, threshold: float) -> bool:
    if op == "<=":
        return value <= threshold
    if op == "<":
        return value < threshold
    if op == ">=":
        return value >= threshold
    if op == ">":
        return value > threshold
    raise ValueError(f"invalid op {op!r}")


def check_constraints(
    metrics: dict[str, float],
    constraints: list[dict[str, Any]],
) -> tuple[bool, list[ConstraintResult]]:
    """Evaluate every constraint. Missing metric -> fail."""
    results: list[ConstraintResult] = []
    all_passed = True
    for spec in constraints:
        name = spec["name"]
        op = spec["op"]
        threshold = float(spec["threshold"])
        raw = metrics.get(name)
        if raw is None:
            all_passed = False
            results.append(
                ConstraintResult(name=name, op=op, threshold=threshold, value=None, passed=False)
            )
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            all_passed = False
            results.append(
                ConstraintResult(name=name, op=op, threshold=threshold, value=None, passed=False)
            )
            continue
        passed = _cmp(op, value, threshold)
        if not passed:
            all_passed = False
        results.append(
            ConstraintResult(name=name, op=op, threshold=threshold, value=value, passed=passed)
        )
    return all_passed, results


def check_termination(
    session: Session,
    history: list[dict[str, Any]],
    primary_direction: str = "min",
) -> tuple[bool, str | None]:
    """Evaluate session termination conditions against run history.

    history: rows of results.tsv (list of dicts with string values).
    primary_direction: "min" or "max", sourced from prepare.primary_spec at the
      caller (cmd_run). We take it explicitly because session.runner is the
      runner spec and never holds direction — plumbing from prepare.py keeps
      the contract single-sourced.
    Returns (should_terminate, terminated_by) where terminated_by is one of:
      None | "primary_threshold" | "plateau" | "max_runs" | "manual" | "chain_budget"
    """
    conds = session.termination_conditions or {}
    if conds.get("unlimited"):
        return False, None

    direction = primary_direction or "min"

    # max_runs
    max_runs = conds.get("max_runs")
    if isinstance(max_runs, int) and max_runs > 0 and len(history) >= max_runs:
        return True, "max_runs"

    # primary_threshold
    threshold = conds.get("primary_threshold")
    if threshold is not None:
        for row in history:
            v = _parse_float(row.get("primary_value"))
            if v is None:
                continue
            if (direction == "min" and v <= float(threshold)) or (
                direction == "max" and v >= float(threshold)
            ):
                return True, "primary_threshold"

    # plateau(N): last N runs that actually produced a primary value (status=ok)
    # all failed to improve the best. We deliberately exclude status=invalid
    # (constraint violations or extraction failures — structurally failed
    # attempts, not exhausted search) and status in {crash,timeout,interrupted}
    # so a streak of broken edits does not falsely trigger plateau termination.
    plateau_n = conds.get("plateau_n")
    if isinstance(plateau_n, int) and plateau_n > 0:
        scored = [row for row in history if row.get("status") == "ok"]
        if len(scored) >= plateau_n:
            window = scored[-plateau_n:]
            if all(row.get("improved", "false").lower() != "true" for row in window):
                return True, "plateau"

    return False, None


def _parse_float(x: Any) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
