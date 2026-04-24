"""Process launch + process-tree-aware watchdog.

Contract (spec §ar run flow step 5-6, §A):
  - `build_command(runner_spec, train_py_path)` produces the exact argv list
    — always prefixed with `uv run` — for the three supported runner kinds
    (`accelerate`, `torchrun`, `python`) plus a `custom` passthrough.
  - `run_with_watchdog(cmd, seconds, log_path, runner_kind)`:
      * spawns the subprocess in a fresh process group
        (`start_new_session=True`), redirecting stdout+stderr to log_path.
      * thread: at `seconds` -> `os.killpg(pgid, SIGTERM)`; after the grace
        window (15s for accelerate/torchrun, 5s for plain python) ->
        `os.killpg(pgid, SIGKILL)`; then sweep survivors via psutil and
        /proc scan.
      * if survivors remain after SIGKILL, writes the `.ar-unclean`
        sentinel next to the expr log_path's grandparent, and returns
        status=timeout with verdict=revert.
      * returns LaunchResult(status, exit_code, wall_seconds, pgid).

We intentionally bury all signal-handling complexity in this module. The
caller (ar.cli) just reads LaunchResult and decides downstream behavior.
"""

from __future__ import annotations

import errno
import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - declared dep, but degrade gracefully
    psutil = None  # type: ignore


_ACCELERATE_GRACE_S = 15
_PYTHON_GRACE_S = 5


@dataclass
class LaunchResult:
    status: str  # ok | crash | timeout | unclean
    exit_code: int | None
    wall_seconds: float
    pgid: int
    survivors: list[int]  # non-empty iff unclean kill


# ------------------------- command construction --------------------------


def parse_runner_string(runner: str) -> dict[str, Any]:
    """Parse the verbatim runner string the setup skill stores (e.g.
    `accelerate launch --config_file configs/accelerate_8gpu.yaml --num_processes 8`,
    `torchrun --nproc-per-node 4`, `python`, or an arbitrary custom prefix) into
    the structured runner_spec dict used throughout ar.

    Returns: {"kind", "config_file", "num_processes", "extra_args", "argv"?}

    The verbatim string is the source of truth the user confirmed during the
    interview — ar is responsible for splitting it into the fields build_command
    needs, never vice versa. Returns kind="custom" + argv=<tokens> when the
    leading command doesn't match a known runner.
    """
    import shlex

    if runner is None or not str(runner).strip():
        raise ValueError("runner string is empty")
    tokens = shlex.split(str(runner))
    head = tokens[0]

    spec: dict[str, Any] = {
        "kind": "custom",
        "config_file": None,
        "num_processes": None,
        "extra_args": [],
    }

    if head == "accelerate" and len(tokens) >= 2 and tokens[1] == "launch":
        spec["kind"] = "accelerate"
        rest = tokens[2:]
    elif head == "torchrun":
        spec["kind"] = "torchrun"
        rest = tokens[1:]
    elif head in ("python", "python3") or head.endswith("/python") or head.endswith("/python3"):
        spec["kind"] = "python"
        rest = tokens[1:]
    else:
        spec["argv"] = tokens
        return spec

    # Consume recognized flags; anything unrecognized goes into extra_args
    # verbatim so the exact CLI the user confirmed is preserved.
    extra: list[str] = []
    i = 0
    while i < len(rest):
        t = rest[i]
        if t in ("--config_file", "--config-file") and i + 1 < len(rest):
            spec["config_file"] = rest[i + 1]
            i += 2
            continue
        if t in ("--num_processes", "--num-processes", "--nproc-per-node", "--nproc_per_node") and i + 1 < len(rest):
            try:
                spec["num_processes"] = int(rest[i + 1])
            except ValueError:
                extra.extend([t, rest[i + 1]])
            i += 2
            continue
        extra.append(t)
        i += 1
    spec["extra_args"] = extra
    return spec


def build_command(runner_spec: dict[str, Any], train_py_path: Path) -> list[str]:
    """Build argv for `uv run <runner> ... {train_py}` per spec §Setup -> Runner."""
    train_py = str(train_py_path)
    kind = runner_spec.get("kind", "python")
    extra = list(runner_spec.get("extra_args") or [])

    if kind == "accelerate":
        config_file = runner_spec.get("config_file")
        num_processes = runner_spec.get("num_processes")
        args = ["uv", "run", "accelerate", "launch"]
        if config_file:
            args.extend(["--config_file", str(config_file)])
        if num_processes:
            args.extend(["--num_processes", str(num_processes)])
        args.extend(extra)
        args.append(train_py)
        return args

    if kind == "torchrun":
        num_processes = runner_spec.get("num_processes") or 1
        args = ["uv", "run", "torchrun", "--nproc-per-node", str(num_processes)]
        args.extend(extra)
        args.append(train_py)
        return args

    if kind == "python":
        args = ["uv", "run", "python"]
        args.extend(extra)
        args.append(train_py)
        return args

    if kind == "custom":
        # Custom: user-supplied argv template. We append train_py if not
        # already present. Always prefix `uv run`.
        template = list(runner_spec.get("argv") or [])
        if not template:
            raise ValueError("custom runner requires a non-empty 'argv'")
        args = ["uv", "run", *template]
        if train_py not in args:
            args.append(train_py)
        return args

    raise ValueError(f"unknown runner kind: {kind!r}")


# --------------------------- watchdog kill helpers -----------------------


def _killpg_safe(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        pass
    except OSError as e:
        if e.errno not in (errno.ESRCH, errno.EPERM):
            raise


def _survivors_via_psutil(root_pid: int) -> list[int]:
    if psutil is None:
        return []
    try:
        proc = psutil.Process(root_pid)
        descendants = proc.children(recursive=True)
        alive = [p.pid for p in descendants if p.is_running()]
        return alive
    except psutil.NoSuchProcess:
        return []
    except psutil.Error:
        return []


def _survivors_via_proc(pgid: int) -> list[int]:
    """Scan /proc for PIDs that share the given pgid. Linux-only fallback.

    Useful when the parent process is already reaped and psutil can't walk
    from it to the descendants.
    """
    procfs = Path("/proc")
    if not procfs.is_dir():
        return []
    alive: list[int] = []
    for entry in procfs.iterdir():
        if not entry.name.isdigit():
            continue
        stat_path = entry / "stat"
        try:
            with open(stat_path, "r", encoding="utf-8") as f:
                line = f.read()
        except OSError:
            continue
        # /proc/{pid}/stat has "(comm)" in field 2 which may contain spaces.
        # pgrp is field 5 in the space-split tail after the closing ")".
        rparen = line.rfind(")")
        if rparen < 0:
            continue
        tail = line[rparen + 1 :].strip().split()
        if len(tail) < 3:
            continue
        try:
            pid_pgrp = int(tail[2])
        except ValueError:
            continue
        if pid_pgrp == pgid:
            try:
                alive.append(int(entry.name))
            except ValueError:
                continue
    return alive


def _find_survivors(root_pid: int, pgid: int) -> list[int]:
    via_ps = _survivors_via_psutil(root_pid)
    if via_ps:
        return via_ps
    return _survivors_via_proc(pgid)


def _write_unclean_sentinel(expr_dir: Path, pgid: int, survivors: list[int]) -> None:
    sentinel = Path(expr_dir) / ".ar-unclean"
    payload = {
        "pgid": pgid,
        "surviving_pids": survivors,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        with open(sentinel, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except OSError:
        pass


# ------------------------------ main API ---------------------------------


def run_with_watchdog(
    cmd: list[str],
    seconds: int,
    log_path: Path,
    runner_kind: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    interrupt_event: threading.Event | None = None,
) -> LaunchResult:
    """Launch subprocess with timeout watchdog. Returns LaunchResult.

    log_path's grandparent is the expr_dir — that's where `.ar-unclean` is
    written if we end up with orphaned survivors.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    expr_dir = log_path.parent.parent  # runs/{id}/run.log -> {expr}/

    grace_s = _ACCELERATE_GRACE_S if runner_kind in ("accelerate", "torchrun") else _PYTHON_GRACE_S

    start = time.monotonic()
    log_f = open(log_path, "w", encoding="utf-8")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(cwd) if cwd else None,
            env=env,
        )
    except FileNotFoundError as e:
        log_f.write(f"\n[ar] launch failed: {e}\n")
        log_f.close()
        return LaunchResult(
            status="crash", exit_code=127, wall_seconds=0.0, pgid=-1, survivors=[]
        )
    except OSError as e:
        log_f.write(f"\n[ar] launch failed: {e}\n")
        log_f.close()
        return LaunchResult(
            status="crash", exit_code=1, wall_seconds=0.0, pgid=-1, survivors=[]
        )

    pgid = proc.pid  # start_new_session=True => pgid == pid of child leader

    state = {"terminated_by_watchdog": False, "timed_out": False}

    def _watchdog() -> None:
        deadline = start + seconds
        while True:
            remaining = deadline - time.monotonic()
            if proc.poll() is not None:
                return
            if interrupt_event is not None and interrupt_event.is_set():
                state["terminated_by_watchdog"] = True
                _killpg_safe(pgid, signal.SIGTERM)
                break
            if remaining <= 0:
                state["terminated_by_watchdog"] = True
                state["timed_out"] = True
                _killpg_safe(pgid, signal.SIGTERM)
                break
            time.sleep(min(1.0, max(0.1, remaining)))

        # Grace period for graceful shutdown.
        grace_deadline = time.monotonic() + grace_s
        while time.monotonic() < grace_deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.5)

        # Escalate to SIGKILL on the full group.
        _killpg_safe(pgid, signal.SIGKILL)
        # Give reaper a short window.
        t_end = time.monotonic() + 3.0
        while time.monotonic() < t_end:
            if proc.poll() is not None:
                break
            time.sleep(0.2)

    watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
    watchdog_thread.start()

    try:
        exit_code = proc.wait()
    except KeyboardInterrupt:
        # The caller (cli) also installs a SIGINT handler; here we make sure
        # we never leave a run dangling on Ctrl+C.
        state["terminated_by_watchdog"] = True
        _killpg_safe(pgid, signal.SIGTERM)
        time.sleep(min(grace_s, 5))
        _killpg_safe(pgid, signal.SIGKILL)
        exit_code = proc.wait()
        wall = time.monotonic() - start
        log_f.close()
        return LaunchResult(
            status="interrupted",
            exit_code=exit_code,
            wall_seconds=wall,
            pgid=pgid,
            survivors=[],
        )

    watchdog_thread.join(timeout=grace_s + 5)
    wall = time.monotonic() - start
    log_f.close()

    # Survivor sweep after the child is reaped (matters when the watchdog
    # SIGKILLed orphans).
    survivors: list[int] = []
    if state["terminated_by_watchdog"]:
        # Small settle delay so reaper can clean up cooperative shutdowns.
        time.sleep(0.5)
        survivors = _find_survivors(proc.pid, pgid)

    if state["timed_out"]:
        if survivors:
            _write_unclean_sentinel(expr_dir, pgid, survivors)
            return LaunchResult(
                status="unclean",
                exit_code=exit_code,
                wall_seconds=wall,
                pgid=pgid,
                survivors=survivors,
            )
        return LaunchResult(
            status="timeout",
            exit_code=exit_code,
            wall_seconds=wall,
            pgid=pgid,
            survivors=[],
        )

    if state["terminated_by_watchdog"]:
        # Non-timeout watchdog escalation — e.g. external interrupt.
        return LaunchResult(
            status="interrupted",
            exit_code=exit_code,
            wall_seconds=wall,
            pgid=pgid,
            survivors=survivors,
        )

    if exit_code == 0:
        return LaunchResult(
            status="ok", exit_code=0, wall_seconds=wall, pgid=pgid, survivors=[]
        )
    return LaunchResult(
        status="crash", exit_code=exit_code, wall_seconds=wall, pgid=pgid, survivors=[]
    )
