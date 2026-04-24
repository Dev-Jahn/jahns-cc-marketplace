# Project analysis — extended heuristics

Read this when SKILL.md's Phase 1 quick scan produced fewer than 2 strong candidates for any of {goal, primary-metric, mutation-scope, runner, metric-backend, checkpoint-glob, entry-pattern}. These heuristics cover the unusual project layouts that the default scan misses.

## Editable-install detection

The default check is `[tool.setuptools.packages.find]` in `pyproject.toml`. Also accept:

- `[tool.hatch.build.targets.wheel]` (hatch)
- `[tool.uv.workspace]` (uv workspace with subpackages)
- `[tool.poetry.packages]` (poetry)
- A top-level `setup.py` or `setup.cfg` with `packages=find_packages()`

If none are present, the project may still be editable-installable via a `src/` layout implicit in the `[project]` table. Run `grep -l '__init__.py' -r src/ 2>/dev/null | head -5` to detect package roots.

## Metric-backend + metric discovery

Phase 1 signal 3 chooses the backend and extracts a list of metric-name candidates from it.

### Backend selection (ordered; first hit wins)

| Evidence | Backend | Extraction |
|---|---|---|
| `wandb/run-*/files/wandb-summary.json` exists (NOT `wandb/latest-run/` — symlink races under DDP) | `wandb` | Read up to 3 most recent by mtime; union of all top-level keys. |
| `events.out.tfevents.*` files anywhere under the project | `tensorboard` | Use `tensorboard.backend.event_processing.event_accumulator` to enumerate scalar tags; capture the glob that matches (`runs/*/events.out.tfevents.*`, `lightning_logs/*/events.out.tfevents.*`, `outputs/*/tensorboard/events.out.tfevents.*`, etc.). |
| `mlruns/` dir with `mlruns/*/metrics/` | `custom` (mlflow) | Surface a TODO + v0.4.0 note; capture the tag names by listing `mlruns/*/*/metrics/`. |
| Nothing detectable | `log` | Grep the training entrypoint for `print(f".*={'{'}.*:.2f|.4f|.6f{'}'}")`, `logger.info` / `.log` calls referencing numeric tokens, or any `f"{name}=..."` patterns. Offer the distinct metric-name tokens as candidates. |

If none of the above produce ≥ 2 candidates, fall back to synthetic defaults: `val/loss (min)`, `train/loss (min)`, `eval/accuracy (max)`.

### Low-confidence triggers

- A single stale tfevents file (mtime older than the most recent git commit).
- A `wandb/` dir with only `wandb/debug.log` and no run subdirs (wandb-offline failure earlier).
- An `mlruns/` dir with no experiments.

In any low-confidence case, mark the detection low-confidence so Phase 2 item 6 pushes harder for explicit user confirmation.

### Direction inference table

| Key substring | Direction |
|---|---|
| `loss`, `nll`, `ppl`, `perplexity`, `bpb`, `bits_per_byte`, `error`, `wer`, `cer`, `fid`, `lpips` | min |
| `acc`, `accuracy`, `top1`, `top5`, `auc`, `auroc`, `f1`, `recall`, `precision`, `bleu`, `rouge`, `exact_match`, `em`, `mAP`, `iou`, `dice`, `psnr`, `ssim`, `reward`, `score`, `win_rate` | max |
| `vram`, `memory`, `mb`, `gb`, `wall`, `time`, `latency`, `seconds` | min (resource bound, usually a constraint not a primary) |
| `throughput`, `tokens_per_sec`, `samples_per_sec`, `steps_per_sec` | max (resource bound) |

If a key matches none of these, ask the user (interview item 2 accepts free-text direction).

## Runner inference

Scan in this order and take the first hit:

1. **CLAUDE.md** — look for a fenced code block under headings like `## Training`, `## Commands`, `## How to run`. The first `accelerate launch ...` / `torchrun ...` / `python train.py ...` inside such a block is authoritative.
2. **`justfile`, `Makefile`, `scripts/train.sh`** — same grep.
3. **Entry-point Python file** — if `accelerate.Accelerator(...)` is instantiated, the runner is `accelerate launch`. If `torch.distributed.init_process_group` is called explicitly and Accelerator is absent, the runner is `torchrun`. Otherwise plain `python`.
4. **`configs/accelerate_*.yaml`** — presence implies accelerate; pick the most recently modified config file for the default.

For multi-GPU torchrun, extract `nproc-per-node` from:

- `CUDA_VISIBLE_DEVICES` environment hints in CLAUDE.md,
- `--nproc-per-node N` in existing command snippets,
- GPU count from `nvidia-smi -L | wc -l` as a last resort (only if the project clearly uses all visible GPUs).

## Mutation-scope candidates

Good scope candidates are:

- Classes / functions that the training entrypoint imports directly.
- Modules under `training/`, `model/`, `losses/`, `optim/`, `attn/` (or whatever the host project calls these subpackages) in the target package.
- Recently-modified files (`git log --name-only --since="2 weeks ago"`).

Poor candidates (do NOT suggest):

- `utils.py`, `config.py`, `__init__.py`, logging setup, dataset paths.
- Anything under `scripts/`, `tests/`, `docs/`.
- Third-party vendored code (often under `vendor/`, `third_party/`).

Format suggestions as full dotted paths the user can paste. Prefer the namespaced form — `<your_project>.training.losses.YourLoss`, `<your_project>.model.attn.YourBlock` — since most editable-install Python projects expose their source under a single top-level package named after the project. Resolve the package name from `pyproject.toml`'s `[project].name` or `[tool.setuptools.packages.find].include` (if the project exposes multiple top-level packages like bare `model`/`training`, surface those as-is but flag the layout to the user for confirmation — the more common pattern is a single namespace).

## Prepare.py mode decision

**thin-wrapper mode** signals (any one is sufficient):

- Grep hits for `def build_train_loader`, `def build_val_loader`, `def get_dataloader`.
- A `data/` or `datasets/` subpackage with an `__init__.py` exporting loader factories.
- `from torch.utils.data import DataLoader` with instantiation inside a module the entrypoint imports.
- Existence of a `datamodule.py` (lightning / lit-style).

**full-prep mode** signals:

- No data pipeline discoverable — single-file training script, inline dataset download with `requests` / `datasets.load_dataset` at module scope, or the project is a minimal Karpathy-style nanoXXX repo.

When in doubt: pick thin-wrapper. The generated stub has TODO markers the user can fill in; full-prep mode is heavier and harder to retro-convert.

## Git signal

`git log --oneline -n 20` against the current branch. Look for recurring scope tokens in commit subjects (`loss:`, `attn:`, `model:`, `train:`) — these are strong signals for what the user is actively iterating on, and should be elevated as the top goal candidate.

If the repo is a worktree dedicated to autoresearch testing (branch name contains `ar-test`, `autoresearch`, `experiment`), prefer the parent branch's commit log via `git log origin/main..HEAD` and `git log -n 20 origin/main`.

## Config system detection (new in v0.3.0)

Grep the entrypoint and its first-layer imports in this order; first hit wins. The config system directly chooses `--entry-pattern` and shapes the `--cli-args-json` payload.

| Evidence | Config system | Entry pattern | v0.3.0 supported? |
|---|---|---|---|
| `@hydra.main(...)`, `from hydra import main`, `from omegaconf import OmegaConf`, `hydra.utils.instantiate` | hydra | `hydra` | Yes |
| `LightningCLI(` / `from lightning.pytorch.cli import LightningCLI` | lightning-cli | `custom` (v0.3.0); native in v0.4.0 | No — TODO |
| `from fire import Fire` / `fire.Fire(` | fire | `custom` (v0.3.0); native in v0.4.0 | No — TODO |
| `argparse.ArgumentParser(` / `HfArgumentParser(` / `simple_parsing` | argparse-cli | `argparse-cli` | Yes |
| Top-level `def main(cfg: Config):` imported by a tiny wrapper | function | `function` | Yes |
| None of the above | unknown | `custom` | Yes (agent wires it) |

### Hydra-specific signals

When hydra is detected, also check:

- **Config search path**: grep the entrypoint for `@hydra.main(config_path="conf", config_name="config")`. If found, read `conf/config.yaml` (or the named file) to extract the top-level keys — these are the namespaces the user can override (e.g. `optimizer.lr=...`, `model.hidden_dim=...`). Use them to build the baseline `CLI_OVERRIDES` list in interview item 2.
- **Composition groups**: `conf/optimizer/`, `conf/model/`, `conf/trainer/` subdirectories signal composition-style configs. Override format then supports group swaps (`optimizer=adamw` selects `conf/optimizer/adamw.yaml`). Surface 2-3 concrete examples in the interview preview.
- **Resume anchor**: hydra apps typically accept a key like `trainer.resume_from_checkpoint=/path/to/ckpt` rather than a flag. Detect by grepping config files for `resume_from_checkpoint:` or `ckpt_path:`. Capture the dotted path as the `--resume-flag-name` value.

## Checkpoint convention detection (new in v0.3.0)

Grep the entrypoint and the modules it imports, then rank by filesystem evidence. The detection yields a `--checkpoint-glob` suggestion.

| Evidence | Suggested glob | Notes |
|---|---|---|
| `trainer.save_model(`, `Trainer(...).save_model(`, `output_dir="..."` in `TrainingArguments` | `{output_dir}/checkpoint-*/` (resolve `{output_dir}` from `TrainingArguments` default or a CLI-parseable argument) | HF Trainer default; `checkpoint-<step>` subdirs. |
| `save_pretrained(` on a top-level model/tokenizer | `{output_dir}/` (no glob needed — flat) | Usually paired with `output_dir`. |
| `ModelCheckpoint(` (lightning), `dirpath=...` arg | `lightning_logs/*/checkpoints/*.ckpt` (default) or the detected `dirpath` | Lightning default is `lightning_logs/version_N/checkpoints/`. |
| `@hydra.main` + Lightning `ModelCheckpoint` | `outputs/*/checkpoints/best.pt` or `outputs/*/checkpoints/*.ckpt` | Hydra's default cwd is `outputs/YYYY-MM-DD/HH-MM-SS/`, so Lightning's relative `lightning_logs/` becomes `outputs/*/lightning_logs/*/checkpoints/*.ckpt`. Simplify to the typical user-configured path. |
| `accelerator.save_state(` | `checkpoints/*/` or the detected dirpath | Accelerate writes a directory per state (optimizer, scheduler, sampler shards). |
| `torch.save(model.state_dict(), "...")` with a literal path | Extract the literal; convert to glob if it's templated | Plain PyTorch. |
| Nothing detectable | Suggest `skip — host doesn't save to a conventional location` | AR-SAVE falls back to the priority-1 in-process torch-state capture path. |

### Filesystem ranking

If multiple globs are plausible from grep alone, run a quick existence probe on the filesystem:

```bash
for g in "output_dir/checkpoint-*/" "lightning_logs/*/checkpoints/*.ckpt" "checkpoints/*.pt"; do
  matches=$(ls -d $g 2>/dev/null | wc -l)
  echo "$g: $matches match(es)"
done
```

A glob with ≥ 1 existing match outranks one with zero matches. Present the top 2–3 in the interview.

## Entry-pattern detection recap

When Phase 1 signal 4 needs to rank candidates across multiple entrypoint scripts, use this priority ladder:

1. File contains `@hydra.main(...)` decorator → **hydra**, highest rank.
2. File contains `argparse.ArgumentParser(` or `HfArgumentParser(` AND an `if __name__ == "__main__":` block → **argparse-cli**.
3. File contains `LightningCLI(` → **custom** (with v0.4.0 TODO).
4. File contains `fire.Fire(` → **custom** (with v0.4.0 TODO).
5. File defines a top-level `main(**kwargs)` and is imported by at least one other file in the package → **function**.
6. File has an `if __name__ == "__main__":` block with free-form bash-like dispatch (no argparse) → **custom**.
7. Fallback → **custom**.

Ties broken by: (a) which file is referenced in CLAUDE.md's training command block, (b) which file is named `train.py` or `main.py`, (c) mtime (newest wins).
