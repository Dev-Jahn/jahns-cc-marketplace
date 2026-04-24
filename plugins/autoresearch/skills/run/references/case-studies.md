# Case studies — three framework-shaped iteration examples

Three worked examples showing that one `ar run` iteration = one Edit of `CLI_OVERRIDES` in `train.py` + one bash invocation + one Read of `result.json`, regardless of whether the host is HF Trainer, Hydra + TensorBoard, or plain PyTorch + stdout logging.

All three cases share:
- The same 6-step loop protocol from SKILL.md.
- The same read/write discipline (`train.py` is the only editable file).
- The same termination / chain semantics.

Only the `CLI_OVERRIDES` shape, metric backend, and checkpoint glob change per project. `ar init` captured these during `/autoresearch:setup` based on Phase 1 detection.

---

## Case A — HF Trainer + wandb

### Setup recap (what `ar init` captured)

- **Entry pattern**: `argparse-cli` (host has `if __name__ == "__main__": main()` in `myproject.training.main` with `HfArgumentParser`).
- **Distributed framework**: `accelerate`.
- **Runner**: `accelerate launch --config_file configs/accelerate.yaml`.
- **Metric backend**: `wandb` — reads `wandb/run-*/files/wandb-summary.json` via `wandb_pointer.json`.
- **Checkpoint glob**: `output_dir/checkpoint-*/` (HF Trainer convention — `output_dir` is configurable via `TrainingArguments`).
- **CLI_OVERRIDES shape**: flat dict of argparse flag values.
- **Primary metric**: `val/loss` (min).

### State of `train.py` at iteration start

```python
# .autoresearch/{slug}/train.py — wrapper mode, argparse-cli
CLI_OVERRIDES = {
    "output_dir": "./hf_output",
    "per_device_train_batch_size": 8,
    "learning_rate": 2.5e-4,
    "num_train_epochs": 1,
    "evaluation_strategy": "steps",
    "eval_steps": 200,
}
RUN_NOTE = "baseline: lr 2.5e-4"
# AR-INVOKE block follows (frozen) — calls runpy.run_module("myproject.training.main")
# with sys.argv synthesized from CLI_OVERRIDES.
```

### One iteration

```
# Step 1 — status
$ uv run python .autoresearch/ar.py status --compact
primary: val/loss (min) — best: 0.851 @ r0037
runs: 42 total  |  termination: plateau 2/5

# Step 2 — decide
# Hypothesis: drop lr further, r0039 showed 2.5e-4 beats 3e-4.

# Step 3 — Edit (one call, two fields)
CLI_OVERRIDES["learning_rate"] = 2.5e-4  →  2e-4
RUN_NOTE = "lr 2.5e-4 -> 2e-4"

# Step 4 — execute
$ uv run python .autoresearch/ar.py run
[r0043] runner: accelerate launch --config_file configs/accelerate.yaml ...
[r0043] metric extracted via wandb_pointer.json  |  val/loss=0.844
[r0043] checkpoint promoted: hf_output/checkpoint-1500/ → best_ckpt/
[r0043] advance

# Step 5 — read result
$ cat .autoresearch/{slug}/runs/r0043/result.json
# {"status":"ok","verdict":"advance","primary":{"value":0.844},"should_terminate":false}

# Step 6 — not terminating, loop to step 1.
```

Total: 3 bash, 0 additional Reads, 1 Edit.

---

## Case B — Hydra + TensorBoard

### Setup recap

- **Entry pattern**: `hydra` (host has `@hydra.main(config_path="conf", config_name="config")` decorating a `train` function in `myproject.train_hydra`).
- **Distributed framework**: `lightning` (the Hydra app constructs a `pl.Trainer(...)` internally).
- **Runner**: `python -m myproject.train_hydra` (Hydra's standard invocation — Lightning handles multi-GPU internally via `strategy=...`).
- **Metric backend**: `tensorboard` — reads scalar events via `tensorboard.backend.event_processing`.
- **TB events glob**: `outputs/*/tensorboard/events.out.tfevents.*` (Hydra's default cwd is `outputs/YYYY-MM-DD/HH-MM-SS/`; Lightning + TB logger drops events there).
- **Checkpoint glob**: `outputs/*/checkpoints/best.pt`.
- **CLI_OVERRIDES shape**: list of Hydra override strings.
- **Primary metric**: `val/loss` (min) — Lightning logs it as a scalar tag matching the Hydra-configured metric name.

### State of `train.py` at iteration start

```python
# .autoresearch/{slug}/train.py — wrapper mode, hydra
CLI_OVERRIDES = [
    "optimizer.lr=3e-4",
    "model.hidden_dim=512",
    "trainer.max_epochs=1",
    "trainer.precision=bf16-mixed",
    "data.batch_size=32",
]
RUN_NOTE = "baseline: lr=3e-4, hidden=512"
# AR-INVOKE block follows (frozen) — calls
# runpy.run_module("myproject.train_hydra", alter_sys=True)
# with sys.argv=[prog, *CLI_OVERRIDES].
```

### One iteration

Note the nested-override form — the agent tweaks `optimizer.lr` as a single list entry, NOT a flat argparse flag. This is still one Edit call, still one bash, still one result.json Read.

```
# Step 1 — status
$ uv run python .autoresearch/ar.py status --compact
primary: val/loss (min) — best: 0.732 @ r0012
runs: 18 total  |  termination: max_runs 18/50

# Step 2 — decide
# Hypothesis: hidden_dim=512 was under-parameterized; bump to 768 while holding lr constant.

# Step 3 — Edit (one Edit call, rewriting one string in the CLI_OVERRIDES list)
# Old: "model.hidden_dim=512",
# New: "model.hidden_dim=768",
# Old: RUN_NOTE = "... lr sweep ..."
# New: RUN_NOTE = "hidden_dim 512 -> 768 (lr held)"

# Step 4 — execute
$ uv run python .autoresearch/ar.py run
[r0019] runner: python -m myproject.train_hydra optimizer.lr=3e-4 model.hidden_dim=768 ...
[r0019] metric extracted via tensorboard events (outputs/2026-04-24/03-12-09/tensorboard/events.out.tfevents.*)
[r0019] primary val/loss=0.708
[r0019] checkpoint promoted: outputs/2026-04-24/03-12-09/checkpoints/best.pt → best_ckpt/
[r0019] advance

# Step 5 — read result
$ cat .autoresearch/{slug}/runs/r0019/result.json
# {"status":"ok","verdict":"advance","primary":{"value":0.708},"should_terminate":false}

# Step 6 — not terminating, loop.
```

Key observation: the user never interacts with tensorboard directly. `ar run` expands the events glob (which matches the subdirectory Hydra just created) and reads the last scalar value for `val/loss`. The `outputs/*/...` glob is non-deterministic across runs (different timestamp), but `ar` picks the newest matching subdir by mtime.

---

## Case C — Plain PyTorch + stdout logging

### Setup recap

- **Entry pattern**: `argparse-cli` (small single-file script — `myproject/train.py` with `argparse.ArgumentParser`, no fancy framework).
- **Distributed framework**: `none` (single-GPU or CPU — script uses `.to(device)` only, no `dist` / accelerate / lightning).
- **Runner**: `python -m myproject.train` (the `python` runner; single process).
- **Metric backend**: `log` — training prints `val_loss=0.234` to stdout each epoch; `ar` parses `run.log` with a user-supplied regex pattern `val_loss=([0-9.]+)`.
- **Checkpoint glob**: `checkpoints/*.pt` (the script does `torch.save(model.state_dict(), f"checkpoints/epoch{N}.pt")`).
- **CLI_OVERRIDES shape**: flat dict of argparse flag values.
- **Primary metric**: `val_loss` (min) — matched by regex from stdout.
- **No wandb, no tensorboard, no mlflow installed.** `pyproject.toml` has `torch` + `numpy` + `argparse` and nothing else from the tracking ecosystem.

### State of `train.py` at iteration start

```python
# .autoresearch/{slug}/train.py — wrapper mode, argparse-cli
CLI_OVERRIDES = {
    "lr": 1e-3,
    "batch_size": 64,
    "epochs": 5,
    "hidden_dim": 128,
}
RUN_NOTE = "baseline: mlp-128"
# AR-INVOKE block follows (frozen) — runpy.run_module("myproject.train") with sys.argv.
```

### One iteration

```
# Step 1 — status
$ uv run python .autoresearch/ar.py status --compact
primary: val_loss (min) — best: 0.412 @ r0003
runs: 7 total

# Step 2 — decide
# Hypothesis: widen hidden_dim 128 -> 256.

# Step 3 — Edit (one call)
# Old: "hidden_dim": 128,
# New: "hidden_dim": 256,
# Old: RUN_NOTE = "baseline: mlp-128"
# New: RUN_NOTE = "hidden_dim 128 -> 256"

# Step 4 — execute
$ uv run python .autoresearch/ar.py run
[r0008] runner: python -m myproject.train --lr 1e-3 --batch_size 64 --epochs 5 --hidden_dim 256
[r0008] metric backend=log, regex=val_loss=([0-9.]+)
[r0008] parsed last match from run.log  |  val_loss=0.394
[r0008] checkpoint promoted: checkpoints/epoch5.pt → best_ckpt/
[r0008] advance

# Step 5 — read result
$ cat .autoresearch/{slug}/runs/r0008/result.json
# {"status":"ok","verdict":"advance","primary":{"value":0.394},"should_terminate":false}

# Step 6 — not terminating, loop.
```

Key observation: the plugin required **zero** tracker installation. `pyproject.toml` did not gain `wandb` or `tensorboard`; the host script did not need a tracker integration. `ar run` captures the subprocess stdout+stderr into `run.log`, then matches the user's regex at extraction time.

When the regex matches multiple times (one per epoch), the last match wins — that's the "final epoch's value" convention. If no match is found, `ar` sets `primary.value=null` and `status=invalid`; the verdict is `revert`.

---

## Putting it together

| Dimension | Case A | Case B | Case C |
|---|---|---|---|
| Entry pattern | argparse-cli | hydra | argparse-cli |
| Distributed | accelerate | lightning | none |
| Runner | `accelerate launch --config_file ...` | `python -m myproject.train_hydra` | `python -m myproject.train` |
| Metric backend | wandb | tensorboard | log |
| Checkpoint glob | `output_dir/checkpoint-*/` | `outputs/*/checkpoints/best.pt` | `checkpoints/*.pt` |
| CLI_OVERRIDES | `{}` dict | `[]` list of `key=value` | `{}` dict |
| Agent edit per iter | one dict value | one list-string rewrite | one dict value |
| Tracker install required | wandb | tensorboard | none |
| Bash invocations per iter | 2-3 (status + run + result) | 2-3 | 2-3 |
| Read calls per iter | ≤ 2-3 after warmup | ≤ 2-3 | ≤ 2-3 |

The loop protocol is invariant. Only `CLI_OVERRIDES` shape, the extraction backend, and the checkpoint glob change per project — and all three were captured during `/autoresearch:setup`, never during the run-loop.
