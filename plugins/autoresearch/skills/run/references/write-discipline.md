# Write Discipline — Exhaustive File Reference

This reference enumerates files under `.autoresearch/` and the host project with permitted operations. The invariant is strict because the entire advance/revert/chain machinery assumes **exactly one file mutates per iteration**: `{expr}/train.py`.

## Under `.autoresearch/`

| Path | Permission | Rationale |
|---|---|---|
| `.autoresearch/ar.py` | Read-only | Launcher shim. Maintained by `autoresearch:setup`. Editing breaks all `ar` invocations. |
| `.autoresearch/.chain-session.json` | Read-only | Top-level chain state. Written only by `ar chain-init` and the initial interactive preflight. |
| `.autoresearch/{expr}/program.md` | Read-only | Declarative spec for the expr. Changing it silently invalidates comparability across runs within the expr. |
| `.autoresearch/{expr}/prepare.py` | Read-only | Data + metric contract. Changing it means old runs and new runs measure different things — `best.json` becomes meaningless. |
| `.autoresearch/{expr}/train.py` | **Edit allowed** (the ONLY writable file) | The experiment loop's mutation surface. Parameter tweaks, monkey-patches, structural changes all go here. |
| `.autoresearch/{expr}/best.json` | Read-only | Written exclusively by `ar run`'s two-stage atomic swap. Manual edits break the pointer to `best_ckpt/` and to the best-run train.py snapshot. |
| `.autoresearch/{expr}/best.json.new` | Read-only | Transient — exists only during a commit. Startup reconciliation cleans it up. Never edit. |
| `.autoresearch/{expr}/best_ckpt/` | Read-only (dir + contents) | Exactly one live ckpt at steady state. Managed by `ar run`'s two-stage swap. |
| `.autoresearch/{expr}/best_ckpt.new/`, `best_ckpt.old/` | Read-only (transient) | Exist only mid-commit. Startup reconciliation removes stale ones. |
| `.autoresearch/{expr}/results.tsv` | Read-only (append-only by `ar`) | Tab-separated; header + one row per run. Free-text fields (`note`) are sanitized by `ar` (newline/tab stripped, truncate 200 chars, `~ ` prefix if touched). |
| `.autoresearch/{expr}/runs/` | Read-only (dir) | Managed by `ar`. Each run_id subdirectory is a committed record. |
| `.autoresearch/{expr}/runs/{run_id}/train.py` | Read-only | Snapshot of the editable file taken BEFORE that run. `ar` uses this to revert. Reading it is useful for diffing; editing it corrupts revert semantics. |
| `.autoresearch/{expr}/runs/{run_id}/result.json` | Read-only | Structured summary for one run. Canonical signal for step 5. |
| `.autoresearch/{expr}/runs/{run_id}/run.log` | Read-only (and do not read in bulk) | Raw stdout+stderr. Can be hundreds of thousands of tokens. Use `ar tail` for bounded extraction. |
| `.autoresearch/{expr}/runs/{run_id}/state.pt` | Read-only | Candidate ckpt written by train.py's frozen AR-SAVE block. `ar` promotes to `best_ckpt/` on advance. |
| `.autoresearch/{expr}/runs/{run_id}/wandb_pointer.json` | Read-only | Written by AR-SAVE before final metric log when metric backend is `wandb` or `auto`. `ar` uses to bind the correct wandb run under DDP. Absent when backend is `tensorboard` / `log` / `custom`. |
| `.autoresearch/{expr}/runs/{run_id}/metric_extract.json` | Read-only | Written by `ar run`'s metric-extraction step with the raw backend-specific payload (e.g. the full wandb summary dict, the TB scalar tag list, or the regex match span). Debugging aid only. |
| `.autoresearch/{expr}/.ar-session.json` | Read-only | Session settings. Written by `ar` at preflight and by `ar chain-init` on transition. Do not hand-edit. |
| `.autoresearch/{expr}/.ar-unclean` | Read-only (sentinel) | Present only when a watchdog kill left orphans. Requires manual human inspection before deletion — never delete from within the loop. |
| `.autoresearch/{expr}/batch_contract.json` | Read-only | Cached signature from prepare.py's self-test. Validated on every subsequent run. |
| `.autoresearch/{new-slug}/chain_decision.json` | Read-only (mode 0444 after write) | Audit record of a chain transition. Written once by `ar chain-init`; any later rewrite is an error. |

## Outside `.autoresearch/` (host project)

Everything in the host project is read-only from the loop's perspective:

| Path | Permission | Rationale |
|---|---|---|
| `pyproject.toml`, `uv.lock`, `.venv/` | Read-only | Changing dependencies invalidates comparability; `uv sync` would perturb cache and potentially trip the optional sha256 manifest warning. |
| `<your_project>/model/`, `<your_project>/training/`, `<your_project>/attn/`, `<your_project>/losses/`, or whatever layout the host project uses | Read-only (read as needed for comprehension) | Target of monkey-patching via train.py's sandbox. Patches live inside train.py, not in these files. |
| `CLAUDE.md`, `README.md` | Read-only | Reference only. |
| `wandb/` (if metric backend = wandb / auto) | Read-only | Output directory. `ar` reads `wandb-summary.json` via the pointer file; the loop never writes here. |
| `runs/`, `tb_logs/`, `lightning_logs/`, `outputs/*/tensorboard/` (if metric backend = tensorboard / auto) | Read-only | Tensorboard event dirs. `ar` reads `events.out.tfevents.*` via the configured events glob; the loop never writes here. |
| `mlruns/` (if present) | Read-only | MLflow tracking store. v0.3.0 treats mlflow as `custom` backend — a user-supplied extraction snippet reads this; the loop never writes here. |
| Host `output_dir/`, `checkpoints/`, `lightning_logs/*/checkpoints/`, `outputs/*/checkpoints/`, or whatever `--checkpoint-glob` matches | Read-only | Training's native checkpoint directory. AR-SAVE copies the newest matching file/dir into `runs/{run_id}/state.pt` (or `state/`) after each run — but only reads, never mutates, the host's copy. |
| `.git/`, git working tree | Read-only (no git operations) | The experiment loop lives entirely outside git. No commits, resets, branches, or adds during the loop. |

## Permitted runtime behavior that looks like a write but isn't

- **Monkey-patching host modules from within train.py** — allowed. The fresh-subprocess boundary (`uv run` spawns a new interpreter per run) guarantees patches do not leak across runs. This is the designed mechanism for structural exploration without touching host source.
- **train.py writing to `runs/{run_id}/` artifacts (state.pt or state/, wandb_pointer.json when applicable, metric_extract.json)** — allowed inside the frozen AR-SAVE block. The agent does not edit the AR-SAVE block; it is a template-frozen region.
- **Subprocess-level tracker writes** (wandb into `wandb/run-*/`, tensorboard scalar events into `runs/` / `lightning_logs/`, mlflow runs into `mlruns/`, or stdout into `run.log`) — allowed; that's the training run's normal output for whichever backend the host uses. AR-SAVE / `ar run` READ these; they do not modify them.
- **Subprocess-level checkpoint writes** into the `--checkpoint-glob`-matched directory (HF Trainer's `output_dir/checkpoint-*/`, Lightning's `*/checkpoints/*.ckpt`, accelerate's `*/state/`, plain `torch.save` to `checkpoints/*.pt`) — allowed; AR-SAVE's priority-0 discovery path expects the host to produce these.

## Detection mechanisms

The discipline is enforced primarily by this prose — `ar` does not monitor every filesystem write. However, two passive detectors are in play:

1. **Fresh-subprocess boundary.** In-process monkey-patches never leak into `ar`'s own bookkeeping process, so even if a train.py aggressively patches host internals at runtime, `ar`'s next invocation starts clean.
2. **Optional sha256 manifest warning.** `autoresearch:setup` can capture a hash of the protected file set; `ar run` logs a warning (not an error) when any protected file's hash changes between runs. This catches accidental edits without causing false-positive aborts from `uv sync`, IDE indexers, or cache population.

## Why this boundary is worth the friction

The design principle "writes are hard-gated, reads are soft-nudged" exists because:

- The revert-on-non-advance unified rule assumes `train.py` is the only thing that changed. If `prepare.py` or a host module was also edited, `ar` cannot restore it — the comparability of all prior runs is silently broken.
- The atomic `best.json` + `best_ckpt/` commit assumes `best_ckpt/` is written only by the two-stage swap. Manual writes corrupt the pointer invariant.
- Chain-init's `parent_report_sha256` audit assumes the parent's `ar report` output was derived from a clean results.tsv and program.md. Hand-editing either voids the audit.
- `results.tsv`'s column integrity assumes `note` is the only free-text column and that `ar`'s sanitizer is the only writer. Manual edits can introduce embedded tabs/newlines that shift columns.

A one-file mutation surface makes every one of these invariants trivially maintainable. The friction of "but I just want to tweak program.md mid-run" is the cost of having an auditable, revertable, chainable loop.
