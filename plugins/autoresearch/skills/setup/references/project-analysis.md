# Project analysis — extended heuristics

Read this when SKILL.md's Phase 1 quick scan produced fewer than 2 strong candidates for any of {goal, primary-metric, mutation-scope, runner}. These heuristics cover the unusual project layouts that the default scan misses.

## Editable-install detection

The default check is `[tool.setuptools.packages.find]` in `pyproject.toml`. Also accept:

- `[tool.hatch.build.targets.wheel]` (hatch)
- `[tool.uv.workspace]` (uv workspace with subpackages)
- `[tool.poetry.packages]` (poetry)
- A top-level `setup.py` or `setup.cfg` with `packages=find_packages()`

If none are present, the project may still be editable-installable via a `src/` layout implicit in the `[project]` table. Run `grep -l '__init__.py' -r src/ 2>/dev/null | head -5` to detect package roots.

## wandb metric discovery

Primary source: `wandb/run-*/files/wandb-summary.json`. Each is a flat JSON dict. Pick the 3 most recent by mtime.

Glob pattern: `wandb/run-*/files/wandb-summary.json` (NOT `wandb/latest-run/` — the symlink races under DDP).

If no `wandb/` directory exists, fall back in order:

1. `tb_logs/`, `tensorboard/`, `runs/` (tensorboard) — grep for `.add_scalar("name", ...)` calls.
2. `logs/*.log`, `*.jsonl` — scan for lines matching `^{metric}=<number>$` or JSONL with a `metrics` key.
3. `trackio/` (huggingface trackio) — read the latest run's `metrics.jsonl`.
4. If still empty, propose synthetic defaults: `val/loss (min)`, `train/loss (min)`, `eval/accuracy (max)`.

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
- Modules under `training/`, `model/`, `losses/`, `optim/`, `attn/` in the target package.
- Recently-modified files (`git log --name-only --since="2 weeks ago"`).

Poor candidates (do NOT suggest):

- `utils.py`, `config.py`, `__init__.py`, logging setup, dataset paths.
- Anything under `scripts/`, `tests/`, `docs/`.
- Third-party vendored code (often under `vendor/`, `third_party/`).

Format suggestions as full dotted paths the user can paste: `training.losses.UnifiedLoss`, `model.na_gla_block.NAGLABlock`. Resolve the package name from `pyproject.toml`'s `[project].name` or `[tool.setuptools.packages.find].include`.

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
