# autoresearch (Claude Code plugin)

Autonomous ML-research loop for real ML codebases. Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) but adapted to:

- Editable-install Python projects (`uv`, `pyproject.toml`, namespaced packages such as `<your_project>.model`, `<your_project>.training`, etc.)
- Any mainstream entry pattern: argparse-CLI, Hydra (`@hydra.main`), or importable function. (Fire / LightningCLI: v0.4.0.)
- Any mainstream distributed launcher: `accelerate launch`, `torchrun`, plain `python`, pytorch-lightning 2.x, DeepSpeed, FSDP, or single-process.
- Any mainstream metric backend: wandb, TensorBoard, stdout-log regex, or a user-supplied Python extraction snippet. Defaults to `auto` — tries wandb → tensorboard → log per run.
- Any mainstream checkpoint layout: HF Trainer's `output_dir/checkpoint-*/`, Lightning's `lightning_logs/*/checkpoints/*.ckpt`, Hydra + Lightning composed, plain `torch.save` paths, or `accelerate.save_state` dirs.
- Overnight runs of hundreds–thousands of iterations in a single Claude Code session.

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

## Verified

### Phase 7 (v0.1, 2026-04-24) — HF Trainer + wandb + accelerate baseline

End-to-end smoke on a 6-GPU Blackwell workstation using `accelerate launch --config_file configs/accelerate.yaml` (your DDP config) + wandb-online:

- `ar init` scaffolding across program.md / train.py / prepare.py / results.tsv
- `ar run` baseline advance (r0001, 24s wall) and iteration advance (r0002, 21s)
- `max_runs` termination condition
- `ar run` with deliberate crash → `status=crash`, train.py auto-reverted to last best, `best_ckpt/` preserved
- `wandb_pointer.json` binding (not the symlink race-prone `latest-run`)
- Atomic two-stage best swap survives verdict=revert / crash paths
- Process-tree-aware watchdog cleans up DDP workers on timeout
- `ar status`, `ar tail`, `ar report` subcommands

### v0.3.0 (2026-04-24) — framework-agnostic additions (Phase 7c pending)

- Three documented case studies exercised end-to-end in the skill references: HF Trainer + wandb, Hydra + TensorBoard, plain PyTorch + stdout log.
- Setup skill decoupled from wandb / accelerate mandatory imports; `import torch` is the only hard Phase 0 requirement.
- `auto` metric backend: tries wandb → tensorboard → log per run, so projects without explicit setup still work.

## Known limitations

### v0.3.0 current

- **Lightning-specific checkpoint hooks are generic.** `--distributed-framework=lightning` is a recognized option, but AR-SAVE uses the filesystem `--checkpoint-glob` discovery path rather than native `ModelCheckpoint` callback wiring. Projects with custom `on_save_checkpoint` overrides may need to adjust the glob to match.
- **MLflow / Comet / Neptune are `custom` backend only.** Phase 1 detects `mlruns/` and surfaces a v0.4.0 TODO; users must supply a `--metric-extract-code` snippet to read them in v0.3.0.
- **LightningCLI and Fire entry patterns are `custom`.** Phase 1 detects them and surfaces v0.4.0 TODOs; users wire the bridge manually in train.py for v0.3.0.
- **SLURM / multinode launchers out of scope.** `ar run` assumes single-node execution under the current shell.
- **DeepSpeed path is partially-generic.** `--distributed-framework=deepspeed` is recognized, but stage-aware AR-SAVE (ZeRO-3 parameter gathering) is not in v0.3.0 — DDP/FSDP-style checkpoint-glob capture only.

### Persistent (unchanged since v0.1)

- `plateau(N)` counts only `status=ok` rows; `invalid`/`crash`/`timeout` are excluded from plateau counting so a streak of broken edits doesn't trigger premature termination.
- `chain-init` exports `AR_NONINTERACTIVE=1` via `.autoresearch/.chain-session.json.env_overrides`, which the run skill must `source` on chain re-entry. Python can't mutate a parent shell's env directly.

## License

MIT
