# autoresearch (Claude Code plugin)

Autonomous LLM-research loop for real ML codebases. Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) but adapted to:

- Editable-install Python projects (`uv`, `pyproject.toml`, namespaced packages such as `<your_project>.model`, `<your_project>.training`, etc.)
- Multi-GPU launchers (`accelerate launch`, `torchrun`)
- wandb-instrumented training loops
- Overnight runs of hundreds–thousands of iterations in a single Claude Code session

## What it provides

Two skills (slash commands):

- **`/autoresearch:setup`** — scaffold an experiment directory under `{PROJECT_ROOT}/.autoresearch/{YYMMDD-slug}/`, interview the user for goal / primary metric / mutation scope / runner, and render `program.md`, `train.py`, `prepare.py` from templates.
- **`/autoresearch:run [slug]`** — drive the autonomous iteration loop. Agent edits only `train.py`; the `ar` helper CLI handles launching, wandb metric extraction, atomic checkpoint swap, and termination / chain-mode transitions.

Plus an `ar` Python CLI (in `lib/ar/`) that keeps main-session context light by absorbing all mechanical work.

## Installation

```
/plugin marketplace add Dev-Jahn/jahns-cc-marketplace
/plugin install autoresearch@jahns-cc-marketplace
/reload-plugins
```

After installation, `/autoresearch:setup` and `/autoresearch:run` appear as slash commands. On first run in a target project, `/autoresearch:setup` creates `.autoresearch/ar.py` — a shim that discovers this plugin via `AR_PLUGIN_ROOT` (if set) or by scanning `~/.claude/plugins/**/autoresearch/plugin/lib`.

## Helper CLI discovery

The `.autoresearch/ar.py` shim placed in the target project discovers the plugin's `lib/ar/` at runtime:

1. `$AR_PLUGIN_ROOT` env var (if set) takes precedence.
2. Otherwise, scan `~/.claude/plugins/**/autoresearch/plugin/lib`, sorted by mtime desc (newest wins).
3. If neither works, fail with a friendly error pointing at this README.

The shim also self-bootstraps `jinja2` and `psutil` via `uv run --with` when they're missing from the target project's venv, so users don't need to add these to their own `pyproject.toml`.

## Design

See `docs/superpowers/specs/2026-04-24-autoresearch-plugin-design.md` in the [source repo](https://github.com/Dev-Jahn/jahns-cc-marketplace) for the full design spec, which was trilateral-reviewed (Claude + Gemini + Codex adversarial review) and revised with fixes A–I before implementation.

## Verified (Phase 7, 2026-04-24)

End-to-end smoke on a 6-GPU Blackwell workstation using `accelerate launch --config_file configs/accelerate.yaml` (your DDP config) + wandb-online:

- `ar init` scaffolding across program.md / train.py / prepare.py / results.tsv
- `ar run` baseline advance (r0001, 24s wall)
- `ar run` iteration advance (r0002, 21s)
- `max_runs` termination condition
- `ar run` with deliberate crash → `status=crash`, train.py auto-reverted to last best, `best_ckpt/` preserved
- `wandb_pointer.json` binding (not the symlink race-prone `latest-run`)
- Atomic two-stage best swap survives verdict=revert / crash paths
- Process-tree-aware watchdog cleans up DDP workers on timeout
- `ar status`, `ar tail`, `ar report` subcommands

## Known limitations (v0.1.0)

- `prepare.py.jinja` thin-wrapper template assumes the host project exposes `build_train_loader` / `build_val_loader`. Projects whose loader factories use different names (e.g. a webdataset-specific builder, an HF `datasets` wrapper, or any other custom factory) need manual prepare.py adaptation after setup, or regeneration once setup skill's interview is extended to ask about loader names.
- `train.py.jinja` rendered body is a TODO-bearing scaffold; the agent is expected to wire it to the target project's real training entrypoint during the first `/autoresearch:run` iteration. The baseline run does not work out-of-the-box without this step.
- `prepare_full.py.jinja` (Karpathy-style inline dataset prep) is a placeholder — MVP didn't require it.
- `plateau(N)` counts only `status=ok` rows; `invalid`/`crash`/`timeout` are excluded from plateau counting so a streak of broken edits doesn't trigger premature termination.
- `chain-init` exports `AR_NONINTERACTIVE=1` via `.autoresearch/.chain-session.json.env_overrides`, which the run skill must `source` on chain re-entry. Python can't mutate a parent shell's env directly.

## License

MIT
