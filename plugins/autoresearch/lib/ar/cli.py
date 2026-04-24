"""argparse entry for `ar` — subcommands: init, run, status, tail, report, resume, chain-init.

Responsibilities:
  - `init`: scaffold {expr}/ from jinja templates (used by setup skill and
    by chain-init).
  - `run`: execute one run per spec §ar run flow. Implements startup
    reconciliation (step 0), two-stage atomic best_ckpt swap (step 10),
    TSV append with sanitization (step 11), termination evaluation
    (step 12), result.json write (step 13), one-line stderr summary
    (step 14), and sys.exit(0) unless an unresolved `.ar-unclean` sentinel
    was detected at startup.
  - `status --compact`: <= 10 lines. `status --last N`: most recent N rows.
  - `tail --run {id} --lines N`: traceback region of run.log.
  - `report --expr {slug}`: markdown summary including chain lineage.
  - `resume [--expr]`: reload `.ar-session.json` pointer (primarily for
    operators; idempotent no-op if session is intact).
  - `chain-init`: materialize a child expr with parent linkage; write
    read-only `chain_decision.json`; signal AR_NONINTERACTIVE via the
    chain-session env_overrides file (see design note on parent-shell
    export below).

Design note — AR_NONINTERACTIVE parent-shell export:
  Python subprocesses cannot mutate the parent shell's environment. The
  spec asks for AR_NONINTERACTIVE=1 to reach the next /autoresearch:run
  invocation, so chain-init writes the desired environment overrides to
  `.autoresearch/.chain-session.json.env_overrides` (a single-line `.env`
  file with `AR_NONINTERACTIVE=1`) — that file is the authoritative source.
  stdout carries the new slug (single line, machine-readable); stderr
  carries a human-readable hint pointing at the overrides file. The run
  SKILL.md sources the overrides file before re-entering the loop. This
  keeps the Python side purely functional and makes the env override
  survivable across shell invocations without relying on process-tree
  inheritance tricks.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import history as hist
from . import launcher as _launcher
from . import wandb_reader
from .decider import check_constraints, check_termination, is_improved
from .schemas import (
    BestRecord,
    ChainDecision,
    ChainSession,
    ConstraintResult,
    Primary,
    RunResult,
    Session,
    best_record_from_dict,
    chain_decision_from_dict,
    chain_session_from_dict,
    dump_json,
    load_json,
    run_result_from_dict,
    session_from_dict,
    validate_status_verdict,
)


# --------------------------- project discovery ---------------------------


def _find_autoresearch_root() -> Path:
    """Ascend from CWD until we find `.autoresearch/`. Fallback: CWD/.autoresearch."""
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        cand = p / ".autoresearch"
        if cand.is_dir():
            return cand
    return cwd / ".autoresearch"


def _pick_default_expr(ar_root: Path) -> str | None:
    """Most-recent expr by mtime. None if no expr exists."""
    if not ar_root.is_dir():
        return None
    candidates = []
    for entry in ar_root.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            candidates.append(entry)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].name


def _expr_dir(ar_root: Path, slug: str) -> Path:
    return ar_root / slug


# ----------------------------- time helpers ------------------------------


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------- reconciliation (ar run step 0) ---------------------


def _reconcile_startup(expr_dir: Path) -> None:
    """Spec §ar run flow step 0. Raises RuntimeError on `.ar-unclean`."""
    sentinel = expr_dir / ".ar-unclean"
    if sentinel.is_file():
        raise RuntimeError(
            f".ar-unclean sentinel present at {sentinel}. Inspect, kill any "
            "stray processes, and delete the sentinel before running again."
        )

    best_json = expr_dir / "best.json"
    best_ckpt = expr_dir / "best_ckpt"
    best_ckpt_new = expr_dir / "best_ckpt.new"
    best_ckpt_old = expr_dir / "best_ckpt.old"
    best_json_new = expr_dir / "best.json.new"

    # best_ckpt.new/ is always stage-1 debris. best.json never points at
    # best_ckpt.new — it always references best_ckpt/, so any leftover .new
    # directory means stage-2 hadn't started when the prior ar run was
    # interrupted.
    if best_ckpt_new.is_dir():
        shutil.rmtree(best_ckpt_new, ignore_errors=True)

    # best_ckpt.old/ leftover -> commit got past first rename; live ckpt is
    # already the successor.
    if best_ckpt_old.is_dir():
        shutil.rmtree(best_ckpt_old, ignore_errors=True)

    # best.json.new orphan -> stage-2 never started with it.
    if best_json_new.is_file():
        try:
            best_json_new.unlink()
        except OSError:
            pass

    # Silence unused-var lint.
    _ = best_ckpt


# ------------------------- prepare.py importer ---------------------------


def _import_prepare(expr_dir: Path):
    """Import `{expr}/prepare.py` with `{expr}` prepended to sys.path."""
    import importlib.util

    prepare_path = expr_dir / "prepare.py"
    if not prepare_path.is_file():
        raise FileNotFoundError(f"prepare.py not found at {prepare_path}")

    # Ensure the expr dir is on sys.path so prepare.py's imports (e.g., of
    # the target project's modules) resolve relative to the project root —
    # the cwd is already the project root per setup contract.
    expr_str = str(expr_dir)
    if expr_str not in sys.path:
        sys.path.insert(0, expr_str)

    spec = importlib.util.spec_from_file_location(
        f"ar_prepare_{expr_dir.name.replace('-', '_')}", prepare_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load spec for {prepare_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _verify_prepare_contract(prepare_mod) -> None:
    # Required on every expr regardless of entry_pattern — this is the
    # metric/resume contract ar itself needs.
    required = [
        "primary_spec",
        "extract_metrics",
        "extract_metrics_from_log",
        "resume_ckpt_path",
    ]
    # build_train_loader / build_val_loader are only required in function /
    # custom mode. In argparse-cli wrapper mode the host script builds its own
    # loaders via its CLI, so prepare.py doesn't need to expose them.
    missing = [n for n in required if not hasattr(prepare_mod, n)]
    if missing:
        raise RuntimeError(
            f"prepare.py missing required attributes: {missing}"
        )


# --------------------- two-stage atomic best-ckpt swap -------------------


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _fsync_file(path: Path) -> None:
    try:
        with open(path, "rb") as f:
            os.fsync(f.fileno())
    except OSError:
        pass


def _commit_best_swap(
    expr_dir: Path,
    run_id: str,
    primary: Primary,
    parent_lineage: str | None,
) -> None:
    """Spec §ar run step 10.

    Stage 1: fully prepare best_ckpt.new/ + best.json.new (reversible).
    Stage 2: three ordered renames + rmtree.
    """
    expr_dir = Path(expr_dir)
    run_dir = expr_dir / "runs" / run_id
    src_state = run_dir / "state.pt"
    if not src_state.is_file() or src_state.stat().st_size == 0:
        raise RuntimeError(f"state.pt missing or empty at {src_state}")
    save_failed = run_dir / ".ar-save-failed"
    if save_failed.exists():
        raise RuntimeError(f".ar-save-failed sentinel present at {save_failed}")

    best_ckpt = expr_dir / "best_ckpt"
    best_ckpt_new = expr_dir / "best_ckpt.new"
    best_ckpt_old = expr_dir / "best_ckpt.old"
    best_json = expr_dir / "best.json"
    best_json_new = expr_dir / "best.json.new"

    # Stage 1: prepare.
    if best_ckpt_new.is_dir():
        shutil.rmtree(best_ckpt_new)
    best_ckpt_new.mkdir(parents=True, exist_ok=False)

    dst_state = best_ckpt_new / "state.pt"
    shutil.copyfile(src_state, dst_state)
    _fsync_file(dst_state)

    meta = {
        "run_id": run_id,
        "primary": {
            "name": primary.name,
            "direction": primary.direction,
            "value": primary.value,
        },
        "created_at": _iso_now(),
        "parent_lineage": parent_lineage,
    }
    meta_path = best_ckpt_new / "meta.json"
    dump_json(meta_path, meta)
    _fsync_file(meta_path)
    _fsync_dir(best_ckpt_new)

    record = BestRecord(
        run_id=run_id,
        primary={
            "name": primary.name,
            "direction": primary.direction,
            "value": primary.value,
        },
        created_at=meta["created_at"],
        ckpt_path="best_ckpt/state.pt",
    )
    dump_json(best_json_new, asdict(record))
    _fsync_file(best_json_new)

    # Stage 2: commit.
    os.rename(best_json_new, best_json)
    _fsync_dir(expr_dir)

    if best_ckpt.is_dir():
        os.rename(best_ckpt, best_ckpt_old)
        _fsync_dir(expr_dir)

    os.rename(best_ckpt_new, best_ckpt)
    _fsync_dir(expr_dir)

    if best_ckpt_old.is_dir():
        shutil.rmtree(best_ckpt_old, ignore_errors=True)


# ---------------------- subcommand: init --------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    """Scaffold `.autoresearch/{slug}/` from the jinja templates."""
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError:
        print("error: jinja2 not installed (uv add jinja2)", file=sys.stderr)
        return 1

    ar_root = _find_autoresearch_root()
    ar_root.mkdir(parents=True, exist_ok=True)
    expr_dir = ar_root / args.expr
    if expr_dir.exists():
        print(f"error: {expr_dir} already exists", file=sys.stderr)
        return 2

    # Locate templates inside the plugin package.
    # __file__ = plugin/lib/ar/cli.py -> plugin/assets
    assets_dir = Path(__file__).resolve().parent.parent.parent / "assets"
    env = Environment(
        loader=FileSystemLoader(str(assets_dir)),
        autoescape=False,
        keep_trailing_newline=True,
    )

    expr_dir.mkdir(parents=True)
    (expr_dir / "runs").mkdir()

    runner_spec = _launcher.parse_runner_string(args.runner)
    # Preserve the original verbatim string for display / debugging / program.md.
    runner_spec["verbatim"] = args.runner

    # P1: entry-pattern + project-introspection flags. Defaults preserve
    # legacy `function` behavior when the new flags are not supplied.
    entry_pattern = getattr(args, "entry_pattern", None) or "custom"
    if entry_pattern not in {"argparse-cli", "function", "custom"}:
        print(
            f"error: --entry-pattern must be one of argparse-cli/function/custom, got {entry_pattern!r}",
            file=sys.stderr,
        )
        return 2

    cli_args_dict: dict[str, Any] = {}
    cli_args_raw = getattr(args, "cli_args_json", None)
    if cli_args_raw:
        import json as _json

        if isinstance(cli_args_raw, str):
            try:
                cli_args_dict = _json.loads(cli_args_raw)
            except Exception as e:
                print(f"error: --cli-args-json not valid JSON: {e}", file=sys.stderr)
                return 2
        elif isinstance(cli_args_raw, dict):
            cli_args_dict = cli_args_raw
    if not isinstance(cli_args_dict, dict):
        print("error: --cli-args-json must decode to a JSON object (dict)", file=sys.stderr)
        return 2

    entry_main_module = getattr(args, "entry_main_module", None)
    wandb_project = getattr(args, "wandb_project", None)
    distributed_framework = getattr(args, "distributed_framework", None) or "accelerate"
    resume_flag_name = getattr(args, "resume_flag_name", None)

    # Serialize CLI overrides as a Python-literal dict for direct paste into
    # train_wrapper.py.jinja. `pprint.pformat` preserves Python syntax for
    # True/False/None (unlike json.dumps which emits true/false/null and
    # breaks at import time).
    import pprint as _pprint
    cli_args_literal = _pprint.pformat(cli_args_dict, indent=4, sort_dicts=True, width=80)

    ctx = {
        "expr_slug": args.expr,
        "goal": args.goal,
        "mutation_scope": list(args.mutation_scope or []),
        "primary_name": args.primary_metric,
        "primary_direction": args.primary_direction,
        "auxiliary": list(args.auxiliary or []),
        "hard_constraints": list(args.constraints or []),
        "runner": runner_spec,
        "per_run_seconds": args.seconds or 300,
        "parent_expr": args.parent_expr,
        "parent_ckpt": args.parent_ckpt,
        "training_entrypoint": args.training_entrypoint
        or "training.train.main  # TODO: set to project's training entrypoint",
        "loader_module": args.loader_module
        or "training.data  # TODO: set to project's dataloader module",
        "initial_params": dict(args.initial_params or {}),
        "run_id": "{{ run_id }}",  # left literal so train.py.jinja can self-expand
        # P1: new entry-point introspection fields.
        "entry_pattern": entry_pattern,
        "entry_main_module": entry_main_module,
        "cli_args_json": cli_args_literal,
        "cli_args_dict": cli_args_dict,
        "wandb_project": wandb_project,
        "distributed_framework": distributed_framework,
        "resume_flag_name": resume_flag_name,
    }

    _render(env, "program.md.jinja", ctx, expr_dir / "program.md")

    prepare_template = "prepare_full.py.jinja" if args.full_prep else "prepare.py.jinja"
    _render(env, prepare_template, ctx, expr_dir / "prepare.py")

    # P2: select train.py template by entry_pattern.
    #   - argparse-cli -> wrapper mode (runpy.run_module on host entry)
    #   - function|custom -> legacy function-mode template
    # See program.md "Entry point" section for the recorded choice.
    if entry_pattern == "argparse-cli":
        if not entry_main_module:
            print(
                "error: --entry-main-module is required when --entry-pattern=argparse-cli",
                file=sys.stderr,
            )
            return 2
        _render(env, "train_wrapper.py.jinja", ctx, expr_dir / "train.py")
    else:
        _render(env, "train.py.jinja", ctx, expr_dir / "train.py")

    # Capture baseline snapshot for revert-to-baseline fallback.
    hist.snapshot_baseline(expr_dir)

    # Empty results.tsv with just the header.
    tsv = expr_dir / "results.tsv"
    with open(tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(hist.RESULTS_HEADER) + "\n")

    print(args.expr)  # slug on stdout for chain-init consumers
    return 0


def _render(env, template_name: str, ctx: dict[str, Any], dest: Path) -> None:
    tpl = env.get_template(template_name)
    rendered = tpl.render(**ctx)
    dump_text_atomic(dest, rendered)


def dump_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ----------------------- subcommand: run --------------------------------


class _SigIntState:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.count = 0


def cmd_run(args: argparse.Namespace) -> int:
    ar_root = _find_autoresearch_root()
    slug = args.expr or _pick_default_expr(ar_root)
    if slug is None:
        print("error: no expr found under .autoresearch/", file=sys.stderr)
        return 1
    expr_dir = _expr_dir(ar_root, slug)
    if not expr_dir.is_dir():
        print(f"error: expr not found: {expr_dir}", file=sys.stderr)
        return 1

    # Step 0: startup reconciliation.
    try:
        _reconcile_startup(expr_dir)
    except RuntimeError as e:
        print(f"ar run: {e}", file=sys.stderr)
        return 3  # unresolved sentinel -> non-zero exit per spec step 15

    # Step 1: ensure session exists.
    session_path = expr_dir / ".ar-session.json"
    if not session_path.is_file():
        print(
            f"error: {session_path} missing — run /autoresearch:run to initialize session.",
            file=sys.stderr,
        )
        return 1
    try:
        session = session_from_dict(load_json(session_path))
    except (OSError, ValueError, KeyError) as e:
        print(f"error: failed to load session: {e}", file=sys.stderr)
        return 1

    duration_s = args.seconds or session.duration_seconds

    # Step 2: allocate run id.
    run_id = hist.next_run_id(expr_dir)
    run_dir = expr_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "run.log"
    result_path = run_dir / "result.json"
    started_at = _iso_now()

    # SIGINT handler: write status=interrupted to in-flight result.json.
    sigint = _SigIntState()
    original_sigint = signal.getsignal(signal.SIGINT)

    def _on_sigint(signum, frame):  # noqa: ARG001
        sigint.count += 1
        sigint.event.set()
        if sigint.count >= 2:
            # second Ctrl+C: actually die
            signal.signal(signal.SIGINT, original_sigint)
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _on_sigint)

    # Read existing best (if any) for improvement comparison + revert target.
    best_path = expr_dir / "best.json"
    best: BestRecord | None = None
    if best_path.is_file():
        try:
            best = best_record_from_dict(load_json(best_path))
        except (OSError, ValueError, KeyError):
            best = None

    # Step 3: snapshot train.py BEFORE any mutation by the run itself.
    try:
        hist.snapshot_train_py(expr_dir, run_id)
    except FileNotFoundError as e:
        _write_and_exit_invalid(
            run_id=run_id,
            started_at=started_at,
            wall_s=0.0,
            log_path=log_path,
            result_path=result_path,
            reason=str(e),
            expr_dir=expr_dir,
            session=session,
            best=best,
            note="",
        )
        signal.signal(signal.SIGINT, original_sigint)
        return 0

    # Step 4: import prepare.py, verify contract. The thin-wrapper template's
    # `_self_test()` runs at module import and does a one-batch sanity check on
    # CPU context, which is enough — we deliberately avoid calling
    # `build_train_loader()` again from the ar process so that the real GPU /
    # DDP init happens exclusively inside the training subprocess.
    try:
        prepare_mod = _import_prepare(expr_dir)
        _verify_prepare_contract(prepare_mod)
    except Exception as e:  # broad by design: user code
        msg = f"prepare contract violation: {e}"
        print(msg, file=sys.stderr)
        _write_and_exit_invalid(
            run_id=run_id,
            started_at=started_at,
            wall_s=0.0,
            log_path=log_path,
            result_path=result_path,
            reason=msg,
            expr_dir=expr_dir,
            session=session,
            best=best,
            note=_extract_run_note(expr_dir),
        )
        signal.signal(signal.SIGINT, original_sigint)
        return 0

    primary_spec = getattr(prepare_mod, "primary_spec", None) or {
        "name": "val/loss",
        "direction": "min",
    }
    hard_constraints = list(getattr(prepare_mod, "hard_constraints", []) or [])

    # Step 5-6: build cmd + launch with watchdog.
    cmd = _launcher.build_command(session.runner, expr_dir / "train.py")
    env = os.environ.copy()
    env.setdefault("AR_RUN_ID", run_id)
    env.setdefault("AR_EXPR_DIR", str(expr_dir))

    launch = _launcher.run_with_watchdog(
        cmd=cmd,
        seconds=duration_s,
        log_path=log_path,
        runner_kind=session.runner.get("kind", "python"),
        env=env,
        interrupt_event=sigint.event,
    )

    # Translate launch status -> run status.
    status: str
    verdict: str
    valid = False
    improved = False
    primary_value: float | None = None
    metrics_dict: dict[str, float] = {}
    constraint_results: list[ConstraintResult] = []
    wandb_run_id: str | None = None
    unclean = False

    if launch.status == "unclean":
        status, verdict, unclean = "unclean", "crash", True
    elif launch.status == "timeout":
        status, verdict = "timeout", "timeout"
    elif launch.status == "interrupted" or sigint.event.is_set():
        status, verdict = "interrupted", "interrupted"
    elif launch.status == "crash":
        status, verdict = "crash", "crash"
    else:
        # launch.status == "ok"; proceed with metric extraction.
        summary = wandb_reader.read_via_pointer(run_dir)
        pointer = wandb_reader.read_pointer(run_dir)
        if pointer:
            wandb_run_id = pointer.get("wandb_run_id")

        metrics_obj = None
        if summary is not None:
            try:
                metrics_obj = prepare_mod.extract_metrics(summary)
            except Exception as e:
                print(f"extract_metrics failed: {e}", file=sys.stderr)
                metrics_obj = None

        if metrics_obj is None:
            # log fallback
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    log_text = f.read()
                metrics_obj = prepare_mod.extract_metrics_from_log(log_text)
            except Exception as e:
                print(f"extract_metrics_from_log failed: {e}", file=sys.stderr)
                metrics_obj = None

        if metrics_obj is None:
            status, verdict = "invalid", "revert"
        else:
            metrics_dict = _metrics_to_dict(metrics_obj)
            primary_value = metrics_dict.get(primary_spec["name"])

            # Constraint evaluation (step 9).
            passed, constraint_results = check_constraints(metrics_dict, hard_constraints)
            valid = passed
            if not passed:
                status, verdict = "invalid", "revert"
            else:
                candidate_primary = Primary(
                    name=primary_spec["name"],
                    direction=primary_spec["direction"],
                    value=primary_value,
                )
                improved = is_improved(candidate_primary, best)
                if improved:
                    status, verdict = "ok", "advance"
                else:
                    status, verdict = "ok", "revert"

    # Step 10: commit two-stage swap iff advance.
    if verdict == "advance":
        try:
            parent_lineage = best.run_id if best else None
            _commit_best_swap(
                expr_dir=expr_dir,
                run_id=run_id,
                primary=Primary(
                    name=primary_spec["name"],
                    direction=primary_spec["direction"],
                    value=primary_value,
                ),
                parent_lineage=parent_lineage,
            )
        except Exception as e:
            # state.pt missing / save failed -> force revert.
            print(f"best-swap aborted: {e}", file=sys.stderr)
            status, verdict = "invalid", "revert"
            valid = False
            improved = False

    # Revert train.py on any non-advance disposition.
    if verdict != "advance":
        hist.restore_train_py_from_best(expr_dir)
        # Also drop the failed candidate's state.pt (best_ckpt untouched).
        bad_state = run_dir / "state.pt"
        if bad_state.is_file():
            try:
                bad_state.unlink()
            except OSError:
                pass

    # Step 11: append to results.tsv.
    note = _extract_run_note(expr_dir) if verdict == "advance" else _extract_run_note_from_snapshot(
        expr_dir, run_id
    )
    tsv_row = {
        "run_id": run_id,
        "ts": started_at,
        "wall_s": f"{launch.wall_seconds:.1f}",
        "status": status,
        "verdict": verdict,
        "primary_name": primary_spec["name"],
        "primary_value": "" if primary_value is None else f"{primary_value:.6g}",
        "valid": "true" if valid else "false",
        "improved": "true" if improved else "false",
        "note": note,
    }
    try:
        hist.append_tsv(expr_dir, tsv_row)
    except Exception as e:
        print(f"warn: failed to append results.tsv: {e}", file=sys.stderr)

    # Step 12: termination check.
    history_rows = hist.read_results_tsv(expr_dir)
    should_terminate, terminated_by = check_termination(
        session, history_rows, primary_direction=primary_spec["direction"]
    )
    if sigint.event.is_set():
        should_terminate, terminated_by = True, "manual"

    # Step 13: write result.json.
    try:
        validate_status_verdict(status, verdict)
    except ValueError as e:
        # If we produced an illegal combo, coerce to closest legal pair
        # rather than crash. This should be a defensive no-op.
        print(f"warn: coercing illegal status/verdict: {e}", file=sys.stderr)
        if status == "ok":
            verdict = "revert"

    result = RunResult(
        run_id=run_id,
        started_at=started_at,
        wall_seconds=round(launch.wall_seconds, 2),
        status=status,
        verdict=verdict,
        metrics=metrics_dict,
        primary={
            "name": primary_spec["name"],
            "direction": primary_spec["direction"],
            "value": primary_value,
        },
        constraints=[asdict(c) for c in constraint_results],
        valid=valid,
        improved_over_best=improved,
        previous_best_run_id=best.run_id if best else None,
        wandb_run_id=wandb_run_id,
        log_path=str(log_path.relative_to(expr_dir)),
        note=note,
        should_terminate=should_terminate,
        terminated_by=terminated_by,
        exit_code=launch.exit_code,
        unclean=unclean,
    )
    dump_json(result_path, asdict(result))

    # Step 14: one-line compact summary to stderr.
    _print_run_oneliner(result, file=sys.stderr)

    # Restore original SIGINT handler before exit.
    signal.signal(signal.SIGINT, original_sigint)

    # Step 15: sys.exit(0) unless unresolved unclean was detected earlier
    # (we already returned 3 in that case).
    return 0


def _write_and_exit_invalid(
    *,
    run_id: str,
    started_at: str,
    wall_s: float,
    log_path: Path,
    result_path: Path,
    reason: str,
    expr_dir: Path,
    session: Session,
    best: BestRecord | None,
    note: str,
) -> None:
    # Write a minimal invalid result.json + TSV row; revert train.py.
    # Make sure log_path file exists so result.json can reference it.
    if not log_path.is_file():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(reason + "\n")
    hist.restore_train_py_from_best(expr_dir)
    tsv_row = {
        "run_id": run_id,
        "ts": started_at,
        "wall_s": f"{wall_s:.1f}",
        "status": "invalid",
        "verdict": "revert",
        "primary_name": "",
        "primary_value": "",
        "valid": "false",
        "improved": "false",
        "note": note,
    }
    try:
        hist.append_tsv(expr_dir, tsv_row)
    except Exception:
        pass

    result = RunResult(
        run_id=run_id,
        started_at=started_at,
        wall_seconds=wall_s,
        status="invalid",
        verdict="revert",
        metrics={},
        primary={"name": "", "direction": "min", "value": None},
        constraints=[],
        valid=False,
        improved_over_best=False,
        previous_best_run_id=best.run_id if best else None,
        wandb_run_id=None,
        log_path=str(log_path.relative_to(expr_dir)),
        note=note,
        should_terminate=False,
        terminated_by=None,
        exit_code=None,
    )
    dump_json(result_path, asdict(result))
    _print_run_oneliner(result, file=sys.stderr)


def _metrics_to_dict(obj: Any) -> dict[str, float]:
    """Accept either a plain dict or our `Metrics` dataclass."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return {k: _to_float(v) for k, v in obj.items() if _to_float(v) is not None}  # type: ignore
    if hasattr(obj, "values") and isinstance(getattr(obj, "values"), dict):
        return {k: _to_float(v) for k, v in obj.values.items() if _to_float(v) is not None}  # type: ignore
    if hasattr(obj, "to_json"):
        return _metrics_to_dict(obj.to_json())
    return {}


def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


_RUN_NOTE_RE = None


def _extract_run_note(expr_dir: Path) -> str:
    """Parse `RUN_NOTE = "..."` out of the working train.py."""
    return _extract_run_note_from_path(expr_dir / "train.py")


def _extract_run_note_from_snapshot(expr_dir: Path, run_id: str) -> str:
    return _extract_run_note_from_path(expr_dir / "runs" / run_id / "train.py")


def _extract_run_note_from_path(path: Path) -> str:
    import re

    global _RUN_NOTE_RE
    if _RUN_NOTE_RE is None:
        _RUN_NOTE_RE = re.compile(
            r"^\s*RUN_NOTE\s*=\s*([\"'])(.*?)\1\s*$", re.MULTILINE
        )
    if not path.is_file():
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = f.read(4096)
    except OSError:
        return ""
    m = _RUN_NOTE_RE.search(head)
    if not m:
        return ""
    return m.group(2)


def _print_run_oneliner(result: RunResult, file) -> None:
    pv = result.primary.get("value")
    pv_str = "-" if pv is None else f"{pv:.6g}"
    line = (
        f"[ar] {result.run_id} status={result.status} verdict={result.verdict} "
        f"{result.primary.get('name', '')}={pv_str} "
        f"wall={result.wall_seconds:.1f}s "
        f"improved={'y' if result.improved_over_best else 'n'} "
        f"terminate={'y' if result.should_terminate else 'n'}"
        f"{' by=' + result.terminated_by if result.terminated_by else ''}"
    )
    print(line, file=file)


# ----------------------- subcommand: status ------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    ar_root = _find_autoresearch_root()
    slug = args.expr or _pick_default_expr(ar_root)
    if slug is None:
        print("no expr found")
        return 0
    expr_dir = _expr_dir(ar_root, slug)
    rows = hist.read_results_tsv(expr_dir)

    if args.last:
        n = args.last
        recent = rows[-n:] if rows else []
        _print_rows_table(recent)
        return 0

    # compact: <= 10 lines.
    lines: list[str] = []
    lines.append(f"expr: {slug}")
    best_path = expr_dir / "best.json"
    if best_path.is_file():
        try:
            best = best_record_from_dict(load_json(best_path))
            pv = best.primary.get("value")
            lines.append(
                f"primary: {best.primary.get('name')} "
                f"best={pv if pv is not None else '-'} "
                f"({best.primary.get('direction')}) "
                f"at {best.run_id}"
            )
        except Exception as e:
            lines.append(f"primary: (best.json parse error: {e})")
    else:
        lines.append("primary: no best yet (baseline not established)")

    total = len(rows)
    advances = sum(1 for r in rows if r.get("verdict") == "advance")
    reverts = sum(1 for r in rows if r.get("verdict") == "revert")
    crashes = sum(1 for r in rows if r.get("status") == "crash")
    timeouts = sum(1 for r in rows if r.get("status") == "timeout")
    lines.append(
        f"runs: total={total} advance={advances} revert={reverts} "
        f"crash={crashes} timeout={timeouts}"
    )

    session_path = expr_dir / ".ar-session.json"
    if session_path.is_file():
        try:
            session = session_from_dict(load_json(session_path))
            conds = session.termination_conditions or {}
            lines.append(
                f"session: chain={session.chain_mode} dur={session.duration_seconds}s "
                f"cond={dict(conds)}"
            )
        except Exception:
            pass

    # Divergence hint: is working train.py == runs/{best_id}/train.py ?
    if best_path.is_file():
        try:
            best = best_record_from_dict(load_json(best_path))
            working = expr_dir / "train.py"
            snapshot = expr_dir / "runs" / best.run_id / "train.py"
            if working.is_file() and snapshot.is_file():
                same = _sha256_file(working) == _sha256_file(snapshot)
                lines.append(f"train.py: {'clean' if same else 'edited-since-best'}")
        except Exception:
            pass

    for line in lines[:10]:
        print(line)
    return 0


def _print_rows_table(rows: list[dict[str, str]]) -> None:
    if not rows:
        print("(no runs)")
        return
    cols = ["run_id", "status", "verdict", "primary_value", "improved", "note"]
    widths = {c: max(len(c), *(len(r.get(c, "")) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join((r.get(c, "") or "").ljust(widths[c]) for c in cols))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------- subcommand: tail ------------------------------


def cmd_tail(args: argparse.Namespace) -> int:
    ar_root = _find_autoresearch_root()
    slug = args.expr or _pick_default_expr(ar_root)
    if slug is None:
        print("error: no expr", file=sys.stderr)
        return 1
    expr_dir = _expr_dir(ar_root, slug)
    log_path = expr_dir / "runs" / args.run / "run.log"
    if not log_path.is_file():
        print(f"error: {log_path} not found", file=sys.stderr)
        return 1
    lines_cap = args.lines or 80

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    out_lines = _extract_traceback_region(text, lines_cap)
    sys.stdout.write("\n".join(out_lines) + "\n")
    return 0


def _extract_traceback_region(text: str, lines_cap: int) -> list[str]:
    lines = text.splitlines()
    # Find last occurrence of a line starting with "Traceback".
    last_tb = -1
    for i, line in enumerate(lines):
        if line.startswith("Traceback"):
            last_tb = i
    if last_tb >= 0:
        region = lines[last_tb:]
        return region[:lines_cap]
    return lines[-lines_cap:] if len(lines) > lines_cap else lines


# ------------------------ subcommand: report -----------------------------


def cmd_report(args: argparse.Namespace) -> int:
    ar_root = _find_autoresearch_root()
    slug = args.expr or _pick_default_expr(ar_root)
    if slug is None:
        print("error: no expr", file=sys.stderr)
        return 1
    expr_dir = _expr_dir(ar_root, slug)

    md = _render_report_md(expr_dir, slug)
    sys.stdout.write(md)
    return 0


def _render_report_md(expr_dir: Path, slug: str) -> str:
    lines = [f"# Report — `{slug}`\n"]

    # Program.md summary (goal + scope).
    prog = expr_dir / "program.md"
    if prog.is_file():
        try:
            lines.append("## Program\n")
            with open(prog, "r", encoding="utf-8") as f:
                body = f.read()
            lines.append(body)
            lines.append("")
        except OSError:
            pass

    # Best summary.
    best_path = expr_dir / "best.json"
    if best_path.is_file():
        try:
            best = best_record_from_dict(load_json(best_path))
            lines.append("## Best\n")
            lines.append(
                f"- run_id: `{best.run_id}`\n"
                f"- primary: `{best.primary.get('name')}` "
                f"({best.primary.get('direction')}) = "
                f"`{best.primary.get('value')}`\n"
                f"- created_at: `{best.created_at}`\n"
            )
        except Exception as e:
            lines.append(f"## Best\n(parse error: {e})\n")
    else:
        lines.append("## Best\n(none — baseline not yet established)\n")

    # Results table.
    rows = hist.read_results_tsv(expr_dir)
    lines.append("## Results\n")
    if not rows:
        lines.append("(no runs)\n")
    else:
        header = "| " + " | ".join(hist.RESULTS_HEADER) + " |"
        sep = "| " + " | ".join("---" for _ in hist.RESULTS_HEADER) + " |"
        lines.append(header)
        lines.append(sep)
        for r in rows:
            cells = [r.get(c, "") for c in hist.RESULTS_HEADER]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # Chain lineage.
    cd_path = expr_dir / "chain_decision.json"
    if cd_path.is_file():
        try:
            decision = chain_decision_from_dict(load_json(cd_path))
            lines.append("## Chain lineage\n")
            lines.append(
                f"- chain_position: `{decision.chain_position}`\n"
                f"- parent_expr: `{decision.parent_expr}`\n"
                f"- parent_primary: "
                f"`{decision.parent_primary.get('name')}` = "
                f"`{decision.parent_primary.get('best_value')}` "
                f"({decision.parent_primary.get('direction')})\n"
                f"- child_primary: "
                f"`{decision.child_primary.get('name')}` "
                f"({decision.child_primary.get('direction')})\n"
                f"- parent_report_sha256: `{decision.parent_report_sha256}`\n\n"
                f"### Rationale\n\n{decision.rationale}\n"
            )
        except Exception as e:
            lines.append(f"## Chain lineage\n(parse error: {e})\n")

    return "\n".join(lines) + "\n"


# ----------------------- subcommand: resume ------------------------------


def cmd_resume(args: argparse.Namespace) -> int:
    ar_root = _find_autoresearch_root()
    slug = args.expr or _pick_default_expr(ar_root)
    if slug is None:
        print("error: no expr found", file=sys.stderr)
        return 1
    expr_dir = _expr_dir(ar_root, slug)
    session_path = expr_dir / ".ar-session.json"
    if not session_path.is_file():
        print(f"error: {session_path} missing", file=sys.stderr)
        return 1
    try:
        _reconcile_startup(expr_dir)
    except RuntimeError as e:
        print(f"ar resume: {e}", file=sys.stderr)
        return 3
    session = session_from_dict(load_json(session_path))
    print(f"resumed expr={slug} chain={session.chain_mode} dur={session.duration_seconds}s")
    return 0


# --------------------- subcommand: chain-init ----------------------------


def cmd_chain_init(args: argparse.Namespace) -> int:
    ar_root = _find_autoresearch_root()
    parent_dir = _expr_dir(ar_root, args.from_expr)
    if not parent_dir.is_dir():
        print(f"error: parent expr {parent_dir} not found", file=sys.stderr)
        return 1

    rationale = (args.rationale or "").strip()
    if not rationale:
        print("error: --rationale is required and must be non-empty", file=sys.stderr)
        return 2

    # Refuse if parent has outstanding .ar-unclean.
    if (parent_dir / ".ar-unclean").is_file():
        print(
            f"error: parent has .ar-unclean sentinel at {parent_dir / '.ar-unclean'}",
            file=sys.stderr,
        )
        return 2

    # Verify parent ckpt exists.
    parent_ckpt = Path(args.parent_ckpt).resolve() if args.parent_ckpt else (
        parent_dir / "best_ckpt"
    ).resolve()
    if not parent_ckpt.is_dir():
        print(f"error: parent ckpt dir missing: {parent_ckpt}", file=sys.stderr)
        return 2
    if not (parent_ckpt / "state.pt").is_file():
        print(f"error: parent ckpt state.pt missing in {parent_ckpt}", file=sys.stderr)
        return 2

    # Disk-space check: >= 2x ckpt size free.
    ckpt_size = _dir_size_bytes(parent_ckpt)
    usage = shutil.disk_usage(str(ar_root))
    if usage.free < 2 * ckpt_size:
        print(
            f"error: insufficient disk space (need >= {2 * ckpt_size} bytes, have {usage.free})",
            file=sys.stderr,
        )
        return 2

    # Load parent session to inherit.
    parent_session_path = parent_dir / ".ar-session.json"
    if not parent_session_path.is_file():
        print(f"error: parent session missing at {parent_session_path}", file=sys.stderr)
        return 2
    parent_session = session_from_dict(load_json(parent_session_path))

    # Load parent primary best.
    parent_best_path = parent_dir / "best.json"
    if parent_best_path.is_file():
        parent_best = best_record_from_dict(load_json(parent_best_path))
        parent_primary = {
            "name": parent_best.primary.get("name"),
            "direction": parent_best.primary.get("direction"),
            "best_value": parent_best.primary.get("value"),
        }
    else:
        parent_primary = {"name": None, "direction": None, "best_value": None}

    # Compute parent report sha256 BEFORE materializing the child so the
    # hash reflects the transition point.
    parent_report_md = _render_report_md(parent_dir, args.from_expr)
    parent_report_sha = hashlib.sha256(parent_report_md.encode("utf-8")).hexdigest()

    # Materialize child expr via init (subprocess-style: call cmd_init in-proc).
    init_args = argparse.Namespace(
        expr=args.new_slug,
        goal=args.goal,
        mutation_scope=list(args.mutation_scope or []),
        primary_metric=args.primary_metric,
        primary_direction=args.primary_direction,
        auxiliary=list(args.auxiliary or []),
        constraints=(
            parent_session.hard_constraints
            if args.constraints == "inherit"
            else _parse_constraints(args.constraints)
        ),
        runner=(
            parent_session.runner.get("kind", "python")
            if args.runner == "inherit"
            else args.runner
        ),
        runner_config=parent_session.runner.get("config_file"),
        num_processes=parent_session.runner.get("num_processes"),
        runner_extra=list(parent_session.runner.get("extra_args") or []),
        seconds=(
            parent_session.duration_seconds
            if (args.seconds is None or args.seconds == "inherit")
            else int(args.seconds)
        ),
        parent_expr=args.from_expr,
        parent_ckpt=str(parent_ckpt),
        training_entrypoint=args.training_entrypoint,
        loader_module=args.loader_module,
        initial_params=None,
        full_prep=False,
        # v0.2.0: forward entry-point introspection so chain children inherit
        # the parent's wrapper-mode config. chain-init exposes these as
        # optional flags; all None means legacy function-mode rendering, which
        # preserves 0.1.x chain behavior.
        entry_pattern=getattr(args, "entry_pattern", None),
        entry_main_module=getattr(args, "entry_main_module", None),
        cli_args_json=getattr(args, "cli_args_json", None),
        wandb_project=getattr(args, "wandb_project", None),
        distributed_framework=getattr(args, "distributed_framework", None),
        resume_flag_name=getattr(args, "resume_flag_name", None),
    )
    rc = cmd_init(init_args)
    if rc != 0:
        print(f"error: init for chain child failed (rc={rc})", file=sys.stderr)
        return rc

    child_dir = _expr_dir(ar_root, args.new_slug)

    # Copy parent .ar-session.json into child (inherited deterministically).
    child_session = Session(
        expr=args.new_slug,
        duration_seconds=parent_session.duration_seconds
        if (args.seconds is None or args.seconds == "inherit")
        else int(args.seconds),
        termination_conditions=dict(parent_session.termination_conditions or {}),
        hard_constraints=list(parent_session.hard_constraints),
        runner=dict(parent_session.runner),
        chain_mode=parent_session.chain_mode,
        created_at=_iso_now(),
        baseline_acknowledged=True,
        resume_mode="continue",
    )
    dump_json(child_dir / ".ar-session.json", asdict(child_session))

    # Update chain-session state.
    chain_session_path = ar_root / ".chain-session.json"
    if chain_session_path.is_file():
        cs = chain_session_from_dict(load_json(chain_session_path))
    else:
        cs = ChainSession(
            chain_mode=parent_session.chain_mode,
            chain_remaining=None,
            chain_trail=[args.from_expr],
            total_wall_time_budget_s=None,
            current_slug=args.from_expr,
            created_at=_iso_now(),
        )

    chain_position = len(cs.chain_trail)  # 0-indexed; parent at 0
    cs.chain_trail.append(args.new_slug)
    cs.current_slug = args.new_slug
    if cs.chain_remaining is not None:
        cs.chain_remaining = max(0, cs.chain_remaining - 1)
    dump_json(chain_session_path, asdict(cs))

    # Write read-only chain_decision.json.
    decision = ChainDecision(
        parent_expr=args.from_expr,
        parent_primary=parent_primary,
        child_primary={
            "name": args.primary_metric,
            "direction": args.primary_direction,
        },
        child_goal=args.goal,
        child_mutation_scope=list(args.mutation_scope or []),
        rationale=rationale,
        parent_report_sha256=parent_report_sha,
        created_at=_iso_now(),
        chain_position=chain_position,
    )
    cd_path = child_dir / "chain_decision.json"
    # If a stale partial exists (from a prior failed attempt), delete it
    # first so the retry produces a single authoritative record.
    if cd_path.is_file():
        try:
            os.chmod(cd_path, stat.S_IWUSR | stat.S_IRUSR)
            cd_path.unlink()
        except OSError:
            pass
    dump_json(cd_path, asdict(decision))
    try:
        os.chmod(cd_path, 0o444)
    except OSError:
        pass

    # Output convention (matches cmd_init and chain-transition.md):
    #   stdout = new slug (single line) — machine-readable output
    #   stderr = AR_NONINTERACTIVE=1 hint — intended to be visually copied or
    #   the ar caller reads the env overrides file directly
    # Authoritative env state is the overrides file below; the stderr hint is
    # advisory only, so the old `eval "$(ar chain-init ...)"` pattern must be
    # replaced with `source .autoresearch/.chain-session.json.env_overrides` in
    # any shell wrapper that needs to export the flag.
    env_overrides = ar_root / ".chain-session.json.env_overrides"
    dump_text_atomic(env_overrides, "AR_NONINTERACTIVE=1\n")
    print(args.new_slug)
    print("AR_NONINTERACTIVE=1 (exported via .chain-session.json.env_overrides)", file=sys.stderr)
    return 0


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def _parse_constraints(arg: Any) -> list[dict[str, Any]]:
    """Parse `name<=threshold,name2>=threshold2` into list of dicts."""
    if arg is None or arg == "inherit" or arg == "":
        return []
    import re

    out: list[dict[str, Any]] = []
    for piece in str(arg).split(","):
        piece = piece.strip()
        if not piece:
            continue
        m = re.match(r"^([\w./-]+)\s*(<=|>=|<|>)\s*([-+0-9.eE]+)$", piece)
        if not m:
            continue
        out.append(
            {
                "name": m.group(1),
                "op": m.group(2),
                "threshold": float(m.group(3)),
            }
        )
    return out


# ------------------------------ entry -----------------------------------


def build_parser() -> argparse.ArgumentParser:
    # Import here to avoid the __init__ -> cli circular import at module load.
    from . import __version__ as _version

    p = argparse.ArgumentParser(prog="ar", description="autoresearch helper CLI")
    p.add_argument("--version", action="version", version=f"ar {_version}")
    sub = p.add_subparsers(dest="subcommand", required=True)

    # init
    pi = sub.add_parser("init", help="scaffold .autoresearch/{slug}/")
    pi.add_argument("--expr", required=True)
    pi.add_argument("--goal", required=True)
    pi.add_argument("--primary-metric", required=True)
    pi.add_argument("--primary-direction", choices=["min", "max"], required=True)
    pi.add_argument(
        "--runner",
        required=True,
        help="verbatim runner string, e.g. 'accelerate launch --config_file configs/accelerate_8gpu.yaml --num_processes 8', 'torchrun --nproc-per-node 4', 'python'. Parsed by launcher.parse_runner_string into the structured spec build_command consumes.",
    )
    pi.add_argument("--mutation-scope", nargs="*", default=[])
    pi.add_argument("--auxiliary", nargs="*", default=[])
    pi.add_argument("--constraints", nargs="*", default=[],
                    help="list of {name, op, threshold} JSON or pre-parsed dicts")
    pi.add_argument("--seconds", type=int, default=300)
    pi.add_argument("--parent-expr", default=None)
    pi.add_argument("--parent-ckpt", default=None)
    pi.add_argument("--training-entrypoint", default=None)
    pi.add_argument("--loader-module", default=None)
    pi.add_argument("--initial-params", default=None,
                    help="JSON dict of initial parameters for train.py")
    pi.add_argument("--full-prep", action="store_true",
                    help="use the full-prep prepare.py template (Karpathy style)")
    # P1: entry-point introspection flags. All optional (None => legacy
    # function-mode behavior) so existing callers / chain-init keep working.
    pi.add_argument(
        "--entry-pattern",
        choices=["argparse-cli", "function", "custom"],
        default=None,
        help=(
            "shape of the host's training entrypoint. 'argparse-cli' renders "
            "train_wrapper.py.jinja which runpy-invokes --entry-main-module; "
            "'function' / 'custom' render the legacy train.py.jinja which "
            "expects an importable main(**kwargs)."
        ),
    )
    pi.add_argument(
        "--entry-main-module",
        default=None,
        help=(
            "dotted module path executed via runpy in argparse-cli mode, e.g. "
            "'your_project.training.main'. The module's `if __name__ == \"__main__\"` "
            "block runs with sys.argv rebuilt from --cli-args-json."
        ),
    )
    pi.add_argument(
        "--cli-args-json",
        default=None,
        help=(
            "JSON dict of baseline CLI args the agent starts from, e.g. "
            "'{\"config\":\"base_flat\",\"learning_rate\":1.5e-4}'. Rendered "
            "verbatim as CLI_OVERRIDES in train_wrapper.py."
        ),
    )
    pi.add_argument("--wandb-project", default=None,
                    help="wandb project name; recorded in program.md Entry point section.")
    pi.add_argument(
        "--distributed-framework",
        choices=["accelerate", "deepspeed", "fsdp", "ddp", "none"],
        default=None,
        help="distributed framework in use; default 'accelerate'.",
    )
    pi.add_argument(
        "--resume-flag-name",
        default=None,
        help=(
            "argparse flag the host accepts to resume from a checkpoint path "
            "(e.g. 'resume_from_checkpoint'). When set AND a parent ckpt is "
            "available, the wrapper injects the resolved path into CLI_OVERRIDES."
        ),
    )
    pi.set_defaults(func=cmd_init)

    # run
    pr = sub.add_parser("run", help="execute one run of current expr")
    pr.add_argument("--expr", default=None)
    pr.add_argument("--seconds", type=int, default=None)
    pr.set_defaults(func=cmd_run)

    # status
    ps = sub.add_parser("status", help="compact summary of expr state")
    ps.add_argument("--expr", default=None)
    ps.add_argument("--compact", action="store_true")
    ps.add_argument("--last", type=int, default=None,
                    help="show last N rows of results.tsv")
    ps.set_defaults(func=cmd_status)

    # tail
    pt = sub.add_parser("tail", help="show traceback region of a run.log")
    pt.add_argument("--expr", default=None)
    pt.add_argument("--run", required=True)
    pt.add_argument("--lines", type=int, default=80)
    pt.set_defaults(func=cmd_tail)

    # report
    prep = sub.add_parser("report", help="markdown summary of an expr")
    prep.add_argument("--expr", default=None)
    prep.set_defaults(func=cmd_report)

    # resume
    pres = sub.add_parser("resume", help="reload session + reconcile")
    pres.add_argument("--expr", default=None)
    pres.set_defaults(func=cmd_resume)

    # chain-init
    pc = sub.add_parser("chain-init", help="materialize chained child expr")
    pc.add_argument("--from-expr", required=True)
    pc.add_argument("--new-slug", required=True)
    pc.add_argument("--goal", required=True)
    pc.add_argument("--mutation-scope", nargs="*", default=[])
    pc.add_argument("--primary-metric", required=True)
    pc.add_argument("--primary-direction", choices=["min", "max"], required=True)
    pc.add_argument("--rationale", required=True)
    pc.add_argument("--runner", default="inherit")
    pc.add_argument("--seconds", default="inherit")
    pc.add_argument("--constraints", default="inherit")
    pc.add_argument("--parent-ckpt", default=None)
    pc.add_argument("--training-entrypoint", default=None)
    pc.add_argument("--loader-module", default=None)
    pc.add_argument("--auxiliary", nargs="*", default=[])
    # v0.2.0: wrapper-mode forwarding. Parent typically passes these verbatim
    # so the child materializes with the same entry-point contract.
    pc.add_argument(
        "--entry-pattern",
        choices=["argparse-cli", "function", "custom"],
        default=None,
    )
    pc.add_argument("--entry-main-module", default=None)
    pc.add_argument("--cli-args-json", default=None)
    pc.add_argument("--wandb-project", default=None)
    pc.add_argument(
        "--distributed-framework",
        choices=["accelerate", "deepspeed", "fsdp", "ddp", "none"],
        default=None,
    )
    pc.add_argument("--resume-flag-name", default=None)
    pc.set_defaults(func=cmd_chain_init)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Parse --initial-params JSON if present (init only).
    if getattr(args, "initial_params", None) and isinstance(args.initial_params, str):
        import json as _json

        try:
            args.initial_params = _json.loads(args.initial_params)
        except Exception:
            args.initial_params = {}
    # Parse --constraints (init only) if given as JSON list.
    if getattr(args, "subcommand", None) == "init" and args.constraints:
        if len(args.constraints) == 1 and args.constraints[0].startswith("["):
            import json as _json

            try:
                args.constraints = _json.loads(args.constraints[0])
            except Exception:
                args.constraints = []
        else:
            # Try pair-wise parsing using _parse_constraints for convenience.
            parsed = _parse_constraints(",".join(args.constraints))
            if parsed:
                args.constraints = parsed
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as e:
        # Always surface the traceback — agent's bash will read stderr.
        traceback.print_exc()
        print(f"ar: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
