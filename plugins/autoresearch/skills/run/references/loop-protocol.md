# Loop Protocol — Operational Detail

This reference expands each phase of the 7-step loop protocol described in SKILL.md with edge cases, first-iteration handling, resume semantics, and worked `result.json` examples.

## First iteration of a fresh expr (no `best.json` yet)

The first run of an expr is the **baseline**. It establishes `best.json` regardless of the primary metric's value, as long as `valid=true` and `status=ok`.

Behavior differences for iteration 1 when `best.json` does not exist:
- Step 1 — `ar status --compact` will show `best: (none)` and `runs: 0 total`.
- Step 3 — do not attempt a "known better" edit against a non-existent baseline; the first edit is usually the program.md-specified default configuration. If the template already contains the defaults, no edit is needed; the first `ar run` exercises the baseline as-is.
- Step 5 — on `ok`/advance, `best.json` is created. On `crash` or `invalid`, no baseline exists yet; the next iteration still has no `best` to revert to. `ar` will revert to the original baseline snapshot (captured at setup time) if one exists, otherwise leave `train.py` as the agent wrote it.
- Step 6 — termination conditions that depend on a prior best (e.g. `plateau`) cannot fire on iteration 1.

## Resuming into a non-empty `runs/`

When preflight asks "resume mode?" because `runs/` is non-empty, the options have these semantics:

- **"continue from current train.py"** — keep the working `train.py` as-is (it may diverge from `best.json`). Iteration 1 treats the current file as the next edit. Appropriate when the previous session was interrupted mid-edit and the in-progress edit is salvageable.
- **"revert to best and continue"** — `ar` copies `runs/{best_id}/train.py` → `{expr}/train.py` before step 1. Iteration 1 starts from the known-good baseline. Appropriate when the previous session crashed and the working tree is suspect.
- **"start fresh"** — destructive. Wipes `runs/`, `best.json`, `best_ckpt/`, `results.tsv` (resets to header row only). Requires a second AskUserQuestion confirmation. Use only when the expr's semantics changed (e.g. the user rewrote program.md).

## Metric extraction — backend-agnostic

Regardless of metric backend (wandb / tensorboard / log / custom / auto), step 5's `result.json` has the same schema. The backend only affects HOW the primary-metric value is obtained; WHAT lands in `primary.value` is identical:

- **wandb**: `wandb_pointer.json` is written by train.py's AR-SAVE block during the run; `ar` reads `wandb/run-{id}/files/wandb-summary.json` post-exit and picks the user-configured metric key.
- **tensorboard**: `ar` expands the configured events glob (e.g. `outputs/*/tensorboard/events.out.tfevents.*`), picks the newest matching file, and reads the last scalar event for the user-configured tag name.
- **log**: `ar` regex-scans `run.log` (the captured subprocess stdout+stderr) and takes the last match's numeric group for the user-configured metric name.
- **custom**: `ar` evaluates the user-supplied snippet with `run_dir` and `run_log_text` in scope; expects a `dict[str, float]` return.
- **auto**: `ar` tries wandb → tensorboard → log in order; the first backend that returns a non-empty result wins for that run.

The primary-metric-missing case surfaces identically regardless of backend: `primary.value=null`, `status=invalid`, `verdict=revert`. The loop's reaction to this outcome (step 5 interpretation table in SKILL.md) does not vary.

## `result.json` — worked examples

### Advance

```json
{
  "run_id": "r0043",
  "status": "ok",
  "verdict": "advance",
  "metrics": {"val/loss": 0.844, "peak_vram_mb": 42800},
  "primary": {"name": "val/loss", "direction": "min", "value": 0.844},
  "constraints": [{"name": "peak_vram_mb", "op": "<=", "threshold": 45000, "value": 42800, "passed": true}],
  "valid": true,
  "improved_over_best": true,
  "previous_best_run_id": "r0037",
  "should_terminate": false,
  "terminated_by": null
}
```

Action: continue. `best.json` is now r0043 @ 0.844. `train.py` is unchanged (the advancing edit stays live).

### Revert (no improvement)

```json
{
  "run_id": "r0044",
  "status": "ok",
  "verdict": "revert",
  "primary": {"name": "val/loss", "direction": "min", "value": 0.847},
  "valid": true,
  "improved_over_best": false,
  "should_terminate": false
}
```

Action: `ar` already restored `train.py` from r0043 (new best). Next edit applies on top of that restored file.

### Revert (constraint violation)

```json
{
  "run_id": "r0045",
  "status": "ok",
  "verdict": "revert",
  "primary": {"name": "val/loss", "direction": "min", "value": 0.820},
  "constraints": [{"name": "peak_vram_mb", "op": "<=", "threshold": 45000, "value": 47100, "passed": false}],
  "valid": false,
  "improved_over_best": true,
  "should_terminate": false
}
```

Note: `valid=false` because the constraint failed, even though the primary metric improved. Verdict is `revert`. Lesson: the next edit must either reduce VRAM or accept the constraint as binding. Do not try the same change again — it will fail the same way.

### Crash

```json
{
  "run_id": "r0046",
  "status": "crash",
  "verdict": "crash",
  "metrics": {},
  "primary": {"name": "val/loss", "direction": "min", "value": null},
  "valid": false,
  "should_terminate": false
}
```

Action: call `ar tail --run r0046 --lines 80`, identify the bug, one fix attempt. `train.py` is already restored to r0043.

### Timeout

```json
{
  "run_id": "r0047",
  "status": "timeout",
  "verdict": "timeout",
  "metrics": {},
  "valid": false,
  "should_terminate": false
}
```

Action: the edit made the run too slow to produce metrics within `--seconds`. Either reduce model size / batch / steps, or bump `--seconds` for the next run (one-off, pass `ar run --seconds N`). `train.py` is restored.

### Terminal run

```json
{
  "run_id": "r0089",
  "status": "ok",
  "verdict": "revert",
  "primary": {"name": "val/loss", "direction": "min", "value": 0.833},
  "should_terminate": true,
  "terminated_by": "plateau"
}
```

Action: step 6 — if chain disabled, exit cleanly. If chain enabled, transition. See `chain-transition.md`.

## Interrupted (Ctrl+C) mid-run

If the user hits Ctrl+C during `ar run`:
- `ar` catches the signal, writes `result.json` with `status=interrupted`, `verdict=interrupted`.
- `train.py` is restored to last best.
- `ar run` exits non-zero.
- The loop should **stop immediately**. Do not invoke another `ar run`. The user intervened deliberately.

After Ctrl+C, the user can resume later with `/autoresearch:run {slug}`; `ar resume` restores `.ar-session.json` so settings persist.

## Unclean sentinel

If a previous run's watchdog could not kill all child processes, `.autoresearch/{expr}/.ar-unclean` exists. The next `ar run` will refuse to launch (step 0 startup reconciliation exits non-zero). The sentinel contains surviving PIDs + pgid + timestamp.

This is fatal for the loop. Exit with a message to the user instructing them to inspect surviving processes (`ps`, `nvidia-smi`) and delete the sentinel manually once verified clean. Do not attempt to continue — silently deleting the sentinel risks running a new experiment with orphan GPU processes still holding memory.

## Context-budget drift — how to notice

Signs the loop is drifting off the bounded-context invariant:
- Running `ar status --last N` with increasing N.
- Re-reading the same file (often `program.md` or a host-project module) every few iterations.
- Reading `runs/{id}/run.log` directly instead of `ar tail`.
- Composing long multi-paragraph `RUN_NOTE` strings (they get truncated to 200 chars anyway).
- Multiple `Edit` calls to `train.py` between `ar run` invocations (one edit per iteration is the target — not a hard rule, but more than 2-3 suggests hypothesis indecision).

Correction: trust `ar status`. Treat prior session context as the canonical state rather than re-deriving it from files.
