---
name: autoresearch:setup
description: Scaffolds a new autonomous-research experiment directory (`.autoresearch/{YYMMDD}-{slug}/`) inside a deep-learning project so Claude can run a long train.py-mutation loop without blowing context. This skill should be used when the user asks to "start an autoresearch experiment", "set up autonomous research loop on this project", "create a new .autoresearch run", "scaffold autoresearch", "initialize autoresearch for this repo", "kick off an autonomous training loop", "set up Karpathy-style autoresearch here", or otherwise indicates they want Claude to begin autonomous iteration on their ML research code. The skill performs a venv preflight, analyzes the project's editable-install Python packages, surfaces primary-metric candidates from whichever tracker the host uses (wandb / tensorboard / plain stdout logs), introspects the host's training entrypoint (argparse-CLI script vs importable main() function vs hydra app), infers the distributed framework (accelerate / torchrun / FSDP / DDP / pytorch-lightning / none), detects checkpoint conventions (HF Trainer / Lightning / plain torch.save), runs a short interview, and then materializes the expr by calling `ar init` which renders the train.py / prepare.py / program.md templates.
argument-hint: [topic]
allowed-tools: Read, Bash, Glob, Grep, AskUserQuestion, Write
---

# autoresearch:setup

Scaffold a new `.autoresearch/{YYMMDD}-{slug}/` experiment in the current project so the autoresearch loop (driven by `autoresearch:run`) can begin.

The skill performs four jobs, in order:

0. **Preflight** the venv + dependency state so `ar init` and `ar run` have a working environment (Phase 0, new in v0.2.0).
1. **Analyze the project** to propose intelligent defaults for the interview — including the new entry-point / distributed-framework / CLI-args probes (Phase 1).
2. **Run an interview** via `AskUserQuestion` — the only user-facing interaction in this skill (Phase 2).
3. **Materialize the expr** by calling `ar init`, bootstrapping the `ar.py` shim first if this is the project's first setup (Phases 3–5).

The interview is the only authorized user prompt. Do not interrupt mid-flow for confirmations, clarifications, or "are you sure?" prompts outside of it. Two exceptions: the Phase 0 remediation prompt (e.g. "symlink sibling .venv?") and the Phase 3 slug-collision prompt both use `AskUserQuestion`.

---

## Phase 0 — venv preflight

Before any project analysis, verify the environment can actually run training. Each check is a one-line bash probe; stop and halt with actionable guidance on the first red signal.

### Step 0.1 — pyproject.toml

```bash
test -f pyproject.toml && echo "ok" || echo "missing"
```

If missing, halt immediately:

> `autoresearch:setup` must run from a Python project root — no `pyproject.toml` found in the current directory. Please `cd` into your project's root (the directory containing `pyproject.toml`) and re-invoke the skill.

Do NOT attempt to locate the project root automatically; this is almost always operator error and the correct response is to abort with a clear message.

### Step 0.2 — venv + mandatory core dep

```bash
uv run --no-sync python -c "import torch" 2>&1
```

The `--no-sync` flag prevents `uv` from silently mutating the environment during the probe. `torch` is the only hard requirement — it's universal across every framework v0.3.0 supports (HF Trainer, Hydra, Lightning, plain-PyTorch, custom loops). Everything else is a soft signal.

- **Exit 0, no output:** proceed to step 0.3.
- **Non-zero exit:** diagnose. See "Diagnosis flowchart" below. When consent is needed, use `AskUserQuestion`.

### Step 0.3 — optional-dep probe (soft signals, informational only)

Probe each in turn. Record each as `present` / `absent` for Phase 1 to consume — do NOT halt on any absence.

```bash
uv run --no-sync python -c "import accelerate"        2>/dev/null && echo "accelerate=present"        || echo "accelerate=absent"
uv run --no-sync python -c "import wandb"             2>/dev/null && echo "wandb=present"             || echo "wandb=absent"
uv run --no-sync python -c "import torch.distributed" 2>/dev/null && echo "torch_distributed=present" || echo "torch_distributed=absent"
uv run --no-sync python -c "import pytorch_lightning" 2>/dev/null && echo "lightning=present"         || echo "lightning=absent"
uv run --no-sync python -c "import lightning"         2>/dev/null && echo "lightning2=present"        || echo "lightning2=absent"
uv run --no-sync python -c "import tensorboard"       2>/dev/null && echo "tensorboard=present"       || echo "tensorboard=absent"
```

These signals feed Phase 1 detection; none of them block. If `wandb=absent`, the metric-backend interview question (Phase 2 item 6) will default to `log`. If `accelerate=absent` and `lightning=absent` and `torch_distributed=absent`, the distributed-framework question will default to `none` via the `auto` sentinel.

### Step 0.4 — wandb login (soft, skipped if wandb absent)

Run only if step 0.3 reported `wandb=present`:

```bash
(test -f ~/.netrc && grep -q "api.wandb.ai" ~/.netrc) && echo "ok" || echo "missing"
```

If missing, warn once:

> Note: `wandb` is installed but `wandb login` does not appear to be configured (`~/.netrc` lacks an `api.wandb.ai` entry). Training will still run, but wandb metrics will be offline-only. Run `uv run wandb login` any time to enable — does not block setup.

Proceed regardless — this is a soft signal. If wandb itself is absent, skip this step entirely.

### Diagnosis flowchart (step 0.2 red path)

First, classify whether `cwd` is a git worktree:

```bash
if [ -f .git ]; then
  head -1 .git | grep -q "^gitdir: " && echo "worktree"
elif [ -d .git ]; then
  echo "main-repo"
else
  echo "not-a-git-repo"
fi
```

**If "worktree":** check siblings for a populated `.venv` (same project, different branch — likely has the same deps already downloaded). Candidate probe:

```bash
MAIN_WORKTREE=$(dirname "$(git rev-parse --git-common-dir)")
for cand in "$MAIN_WORKTREE" "$MAIN_WORKTREE"/../*; do
  if [ -d "$cand/.venv" ]; then
    uv run --project "$cand" --no-sync python -c "import torch" 2>/dev/null && \
      echo "candidate: $cand"
  fi
done
```

If a candidate is found, ask the user via `AskUserQuestion` with options:

- `"Symlink: rm -rf .venv && ln -s <candidate>/.venv .venv"` (fast — reuses downloaded torch/accelerate/flash-attn)
- `"Run uv sync fresh (re-download deps)"` (slow but hygienic)
- `"Abort — I'll fix it myself"`

Do NOT execute the symlink or sync without explicit consent.

**If "main-repo" or "not-a-git-repo":** don't offer the symlink path. Ask via `AskUserQuestion`:

- `"Run uv sync to install/repair deps"` (then re-run step 0.2)
- `"Abort — I'll fix it myself"`

**After user confirms `uv sync`:** run it, wait, re-probe. If still red after one remediation attempt, halt with:

> Preflight probe still failing after `uv sync`. See `references/venv-preflight.md` for failure-mode table; fix manually and re-invoke `/autoresearch:setup`.

See **`references/venv-preflight.md`** for the complete failure-mode table (broken venv symlinks, CUDA mismatches, Python version drift, etc.).

### Halt vs proceed

Proceed to Phase 1 only if steps 0.1 and 0.2 are both green. Steps 0.3 and 0.4 produce informational signals and soft warnings respectively — neither blocks.

---

## Phase 1 — Project analysis

Goal: gather enough signal to propose strong defaults for the interview. Treat this phase as read-only reconnaissance. Target ≤ 14 reads total in this phase (up from 12 in v0.2 because the metric-backend + config-system + checkpoint-convention probes each add one read on average); prefer `Glob`/`Grep` over full-file reads. Do not read files already in session context.

### Signals to collect

Read / glob these in priority order, and stop early once proposals are well-formed:

1. **`pyproject.toml`** — Read full. Extract project name, dependencies (torch / accelerate / wandb / deepspeed / pytorch-lightning / hydra-core / tensorboard / mlflow presence), package roots (`[tool.setuptools.packages.find]` / `[project.scripts]` / `[tool.uv]`).

2. **`CLAUDE.md`** (project-level, plus any under subpackages) — Read if present. Extract:
   - The project's own "how to run training" command block (usually under a "Commands" or "Training" heading). **This is the strongest signal for both the runner and the CLI args** — the exact command the user types to train locally.
   - Project description / research themes for goal candidates.

3. **Metric-backend artifacts** — do not read payloads; glob + stat only.
   - `wandb/run-*/files/` exists? → **wandb** is a strong candidate. If up to 3 most recent runs exist, read `wandb-summary.json` (metric keys), `config.yaml` (the flattened training config — **strongest signal for `--cli-args-json`** — and contains the wandb project via `_wandb.project`), and `wandb-metadata.json` (`program`, `args`, surfaces the exact entry script and argv).
   - `events.out.tfevents.*` files anywhere under the project (common roots: `runs/`, `tb_logs/`, `tensorboard/`, `lightning_logs/*/`, `outputs/*/tensorboard/`) → **tensorboard** is a strong candidate. Record the glob that captures them; it feeds `--tb-events-glob`.
   - `mlruns/` directory → **mlflow** — mark as `custom` for v0.3.0 and emit a TODO: "native mlflow backend planned for v0.4.0; a Python `--metric-extract-code` snippet is required until then".
   - None of the above → **log** (stdout-parsing fallback).
   Rank: wandb ≻ tensorboard ≻ custom (mlflow) ≻ log. Lowest-confidence detection (only one run dir, or only a stale `events.out.tfevents.*`) → mark detection confidence low so Phase 2 item 6 pushes harder for the user to confirm.

4. **Entry-point training scripts** — Glob `training/**/*.py`, `train.py`, `scripts/train*.py`, `tools/train*.py`, `src/**/train*.py`. For each, check:
   - `if __name__ == "__main__":` block present? → candidate for **argparse-cli** pattern.
   - Contains `argparse.ArgumentParser(` or `HfArgumentParser(` or `simple_parsing`? → strongly argparse-cli. Record the dotted-module path (e.g. `myproject.training.main`).
   - Defines a top-level `def main(...)` with typed kwargs and is imported elsewhere? → candidate for **function** pattern.
   - Contains `@hydra.main(...)` decorator? → **hydra** pattern (v0.3.0-native entry pattern; see signal 5 below for config-system confirmation).
   - Neither clearly → **custom** pattern (agent will wire it manually in train.py).
   - **Rank candidates by:** (a) Hydra decorator wins over (b) presence of ArgumentParser + `__main__` block wins over (c) `main()` function; ties broken by which file's path appears in `CLAUDE.md`'s training command.

5. **Config system detection** (new in v0.3.0) — grep the entrypoint and its imports:
   - `@hydra.main` / `from hydra` / `hydra.utils` / `from omegaconf import OmegaConf` → **hydra**. Feeds `--entry-pattern hydra`.
   - `LightningCLI(` → **lightning-cli**. **Not natively supported in v0.3.0** — record as `custom` with a TODO pointer to v0.4.0.
   - `argparse.ArgumentParser(` / `HfArgumentParser(` / `simple_parsing` → **argparse-cli**.
   - `from fire import Fire` / `fire.Fire(` → **fire**. **Not natively supported in v0.3.0** — record as `custom` with a TODO pointer to v0.4.0.
   - None of the above but a top-level `main()` is importable → **function**.
   - Otherwise → **custom**.

6. **Distributed framework** — grep the training entrypoint (and nearby modules) for:
   - `import pytorch_lightning` / `from pytorch_lightning` / `import lightning` / `from lightning.pytorch` / `pl.Trainer(` / `L.Trainer(` → **lightning** (v0.3.0 recognizes this as a distinct framework; the launcher delegates to Lightning's own strategy resolver rather than wrapping with accelerate).
   - `from accelerate` / `Accelerator(` → **accelerate**
   - `import deepspeed` / `deepspeed.initialize` / `deepspeed_config` → **deepspeed**
   - `FullyShardedDataParallel` / `FSDP(` / `torch.distributed.fsdp` → **fsdp**
   - `torch.distributed.init_process_group` / `DistributedDataParallel` without the above → **ddp**
   - Nothing dist-related → **none**
   - Cross-check against Phase 0 step 0.3 signals: if grep says accelerate but `accelerate=absent`, downgrade to a lower-confidence candidate (the user may have inadvertently removed the dep; surface this tension in the interview).
   - If detection is ambiguous or multiple candidates tie, emit `auto` — the new v0.3.0 default that tells `ar run` to introspect the runner command at launch time.

7. **Resume flag detection** — grep the training entrypoint for argparse definitions matching `add_argument\([\"']--?resume` or similar. Common flag names in order of likelihood: `resume_from_checkpoint`, `resume`, `resume_checkpoint`, `ckpt_path`, `pretrained_checkpoint`. Record the detected flag **without** the `--` prefix. If multiple candidates or none, mark as "unclear — ask user". For Hydra entrypoints, the equivalent is a config key (e.g. `trainer.resume_from_checkpoint`) rather than a flag — note that and let Phase 2 capture it as a Hydra override.

8. **Checkpoint convention detection** (expanded in v0.3.0) — grep the entrypoint and nearby modules for save/load patterns, then propose a `--checkpoint-glob` matching where files actually land:
   - `trainer.save_model(`, `save_pretrained(`, `output_dir/checkpoint-*/` (HF Trainer convention) → suggest `output_dir/checkpoint-*/` or the detected `output_dir`-relative equivalent.
   - `*.ckpt` references, `ModelCheckpoint(`, `lightning_logs/` on disk → Lightning → suggest `lightning_logs/*/checkpoints/*.ckpt` (or `outputs/*/checkpoints/best.pt` if Hydra + Lightning).
   - `accelerator.save_state(` → accelerate → suggest `checkpoints/*/` or `<save_dir>/*/`.
   - `torch.save(` with explicit path patterns → extract the path format from the literal and suggest `checkpoints/*.pt` or the project-specific equivalent.
   - Hydra projects with default `outputs/` cwd + safetensors/torch saves → suggest `outputs/*/model.safetensors` or `outputs/*/checkpoints/best.pt`.
   - Multiple candidates → rank by filesystem evidence (a glob that matches existing files wins over a purely grep-derived guess). Offer the top 2-3 in the interview.

9. **`accelerate` / `deepspeed` / Hydra configs** — Glob `configs/accelerate*.yaml`, `accelerate_config*.yaml`, `**/accelerate/*.yaml`, `conf/**/*.yaml`, `configs/**/*.yaml`. If found, note exact filenames — used verbatim in the runner spec and, for Hydra, as the config search path.

10. **Recent git commits** — `git log --oneline -n 20` if inside a git repo. Use commit messages as additional signal for goal candidates.

### What to derive

From the above signals, derive the proposal lists that feed the interview:

| Proposal | Source | Output |
|---|---|---|
| Goal candidates (3–4) | CLAUDE.md + recent commits + top-level module names | Short phrases. |
| Primary-metric candidates | wandb-summary.json keys, tensorboard scalar tags, or grepped stdout regex anchors | Ordered list with inferred direction. |
| Mutation-scope candidates | Training entrypoint imports + obvious module roots | Dotted paths. |
| Runner inference | CLAUDE.md + accelerate configs + entrypoint | `accelerate launch --config_file ...`, `torchrun --nproc-per-node N`, `python`, `custom`. |
| **Entry pattern** | Signals 4 + 5 | `argparse-cli` / `function` / `hydra` / `custom`. |
| **Entry main module** | Signal 4 | Dotted path, e.g. `myproject.training.main`. Only required if pattern is argparse-cli or hydra. |
| **CLI args baseline** | wandb config.yaml + CLAUDE.md command block | Flat JSON-like dict of scalar flag values (e.g. `{"config":"base","learning_rate":1.5e-4}`) for argparse-cli, or a list of Hydra override strings (e.g. `["optimizer.lr=3e-4","model.hidden_dim=512"]`) for hydra. Pick the top 5–10 most-distinctive keys; skip obviously-cosmetic ones (`log_level`, `wandb_mode`). |
| **Metric backend** (new in v0.3.0) | Signal 3 | `wandb` / `tensorboard` / `log` / `custom` / `auto`. |
| **TensorBoard events glob** (new in v0.3.0) | Signal 3 | Glob string, e.g. `runs/*/events.out.tfevents.*`. Only required if metric backend is `tensorboard`. |
| **Custom metric snippet** (new in v0.3.0) | Phase 2 follow-up AskUserQuestion | Short Python body that reads `run_dir` / `run_log_text` and returns a `dict[str, float]`. Required if metric backend is `custom`. |
| **wandb project** | wandb config.yaml (`_wandb.project`) | Project name string. Only relevant if metric backend is `wandb`. |
| **Distributed framework** | Signal 6 | `accelerate` / `deepspeed` / `fsdp` / `ddp` / `lightning` / `none` / `auto`. |
| **Resume flag name** | Signal 7 | String without `--` prefix, or null. For Hydra, capture as an override key instead (e.g. `trainer.resume_from_checkpoint`). |
| **Checkpoint glob** (new in v0.3.0) | Signal 8 | Glob string relative to project cwd (e.g. `output_dir/checkpoint-*/`, `lightning_logs/*/checkpoints/*.ckpt`, `outputs/*/model.safetensors`) or null (host doesn't save to a conventional location). |

prepare.py mode selection (thin-wrapper vs full-prep) stays unchanged — `ar init` handles it based on presence of `build_train_loader` / data-module imports.

### Read budget

Soft cap: 14 reads across this phase. If probes 4–6 produce enough signal on the first training script read, skip additional entry-point candidates. If signal 3 finds no tracker artifacts at all, fall back to `log` + stdout-regex scan (the user confirms the metric name in the interview).

Do not read the same file twice. Do not read `.pt` / `.safetensors` / `.ckpt` / `wandb/run-*/logs/` / `events.out.tfevents.*` binary payloads — large and worthless (globs + stat are enough).

See `references/project-analysis.md` for extended grep patterns if any of {goal, metric, scope, runner, entry-pattern, metric-backend, checkpoint-glob} produced fewer than 2 candidates.

---

## Phase 2 — Interview

One `AskUserQuestion` call with the items below, in this exact order. Use Phase 1 detections to pre-populate `options`; always include a free-text escape hatch. Present detected values with a preview so the user can confirm or override.

### Items

**1. Entry pattern** (`entry_pattern`)
- `question`: "What shape is your training entrypoint?" — include a preview line showing the detected main module, e.g. `Detected: myproject.training.main (argparse.ArgumentParser present + __main__ block)`.
- `options`:
  - `"argparse-cli — runpy-invoke an `if __name__ == '__main__':` script"` (pre-selected if Phase 1 detected ArgumentParser + `__main__`)
  - `"hydra — invoke a @hydra.main-decorated entrypoint with key=value overrides"` (pre-selected if Phase 1 detected `@hydra.main` / hydra deps)
  - `"function — call an importable main(**kwargs)"`
  - `"custom — I'll wire it manually in train.py"`
- Detected main module feeds `--entry-main-module` for argparse-cli / hydra / function. If detection was ambiguous, ask as a follow-up free-text.

**2. Baseline CLI args / Hydra overrides** (`cli_args`)
- Skip entirely if user picked `function` or `custom` in item 1.
- `question`: "These are the baseline overrides the agent will start from (edit `CLI_OVERRIDES` in `train.py` per run). Keep as-is, or edit?" — show the detected dict/list as preview. For argparse-cli:
  ```
  {"config": "base", "learning_rate": 1.5e-4, "num_train_epochs": 1}
  ```
  For hydra:
  ```
  ["optimizer.lr=3e-4", "model.hidden_dim=512", "trainer.max_epochs=1"]
  ```
- `options`:
  - `"Keep as-is"` (pre-selected)
  - `"Edit — I'll paste a JSON value"` (free-text; accept either a dict for argparse-cli or a list of strings for hydra)
  - `"Empty — start with {} / []"` (useful if agent should rebuild from scratch)
- Record as a JSON string; becomes `--cli-args-json`.

**3. Resume flag / override key** (`resume_flag_name`)
- Skip entirely if user picked `function` or `custom` in item 1.
- `question`: "Which flag (argparse) or override key (hydra) resumes from a checkpoint?" — with detected preview if Phase 1 found one.
- `options` (argparse-cli): detected flag first (pre-selected, e.g. `resume_from_checkpoint (detected)`), then common alternatives `resume`, `ckpt_path`, `pretrained_checkpoint`, `"None — no resume flag (use AR_RESUME_CKPT env var instead)"`, `"Other (type)"`.
- `options` (hydra): detected override first (e.g. `trainer.resume_from_checkpoint`), common alternatives `trainer.ckpt_path`, `resume_from`, `"None — use AR_RESUME_CKPT env var"`, `"Other (type)"`.
- Record name without `--` prefix. Becomes `--resume-flag-name` (argparse) or `--resume-override-key` (hydra — same underlying template field, surfaced as an override).

**4. Distributed framework** (`distributed_framework`)
- `question`: "Which distributed-training framework does the host use?"
- `options`: Phase 1 detection first (pre-selected), then `accelerate`, `deepspeed`, `fsdp`, `ddp`, `lightning`, `none`, `auto` (let `ar run` introspect the runner command at launch time — new v0.3.0 default when Phase 1 detection is ambiguous).
- Becomes `--distributed-framework`.

**5. Research goal** (`research_goal`)
- `question`: "What should this experiment optimize for?"
- `options`: 3–4 candidate themes from Phase 1, plus `"Other (describe)"`. Free text accepted.
- Becomes `--goal`. Kebab form seeds the slug if `$ARGUMENTS` was not provided.

**6. Metric backend** (`metric_backend`) — NEW in v0.3.0
- `question`: "How does the host training run report metrics the agent should read?"
- `options`: detected primary first (pre-selected), then the alternatives:
  - `wandb — read `wandb/run-*/files/wandb-summary.json`` (pre-selected if Phase 1 found a `wandb/` dir)
  - `tensorboard — read `events.out.tfevents.*` files via `tensorboard.backend.event_processing`` (pre-selected if Phase 1 found event files)
  - `log — parse training's stdout/stderr (captured by `ar` into `run.log`) with a user-supplied regex`
  - `custom — paste a Python snippet that reads `run_dir` and returns a `dict[str, float]`` (forces follow-up AskUserQuestion below)
  - `auto — let `ar run` try wandb → tensorboard → log in that order on each run`
- **Pushy branch:** if Phase 1 detection confidence is low (e.g. a single stale tfevents file and no wandb runs), phrase the question more assertively: include a preview line like `Low-confidence detection — please confirm; defaulting to log may be safer than auto.`
- Becomes `--metric-backend`.
- Follow-up AskUserQuestion fires **only** if `custom` was selected. Question: "Paste a short Python body (indent-free). It receives `run_dir: Path` and `run_log_text: str`; it must return `dict[str, float]`. Example: `return {\"val_loss\": float(re.search(r'val_loss=([0-9.]+)', run_log_text).group(1))}`". Free-text only. Becomes `--metric-extract-code` (passed as a base64-encoded argument; `ar init` writes it into program.md and train.py).
- If `tensorboard` was selected, the follow-up is a free-text AskUserQuestion for the events glob ("Which glob matches the event files you want to read?"), pre-populated with Phase 1's suggestion. Becomes `--tb-events-glob`.

**7. Checkpoint glob** (`checkpoint_glob`) — NEW in v0.3.0
- `question`: "Where does training save checkpoints? The AR-SAVE block uses this glob to discover new files after each run and promote the best one."
- `options`: Phase 1-detected candidates first (pre-selected), plus common alternatives:
  - `output_dir/checkpoint-*/` (HF Trainer)
  - `lightning_logs/*/checkpoints/*.ckpt` (Lightning defaults)
  - `outputs/*/checkpoints/best.pt` (Hydra + Lightning composed)
  - `checkpoints/*.pt` (plain torch.save)
  - `accelerate_state/*/` (accelerate.save_state)
  - `"skip — host doesn't save to a conventional location"` (AR-SAVE falls back to its priority-1 torch-state snapshot path; agent may need to monkey-patch a save call from train.py)
  - `"Other (glob)"`
- Becomes `--checkpoint-glob`. Drives the wrapper-mode AR-SAVE priority-0 discovery path: when set, AR-SAVE expands the glob after each run, picks the newest matching file/dir by mtime, and copies into `runs/{id}/state.pt` (or `state/` if a directory). When `skip`, AR-SAVE uses the legacy in-process torch-state capture.

**8. Primary metric** (`primary_metric`)
- `question`: "Which metric is the single source of truth for 'did this run improve'?"
- `options`: top metric candidates from Phase 1 with inferred direction, e.g. `val/loss (min)`, `eval/top1 (max)`. Include `"Other (type metric=direction)"`.
- Becomes `--primary-metric` + `--primary-direction`.

**9. Hard constraints** (`hard_constraints`)
- `question`: "Any hard constraints? (Violations mark the run invalid regardless of primary.)"
- `options`: `"None"` (default), plus 2–3 common candidates if auxiliary metrics look like resource bounds (`peak_vram_mb`, `tokens_per_sec`). Multi-select + free-text.
- Parse each as `{name, op, threshold}`; operators `<=`, `<`, `>=`, `>`.

**10. Mutation scope** (`mutation_scope`)
- `question`: "Which modules/classes are in-scope for train.py to monkey-patch?"
- `options`: dotted-path candidates from Phase 1; multi-select + free-text. Include `"(advisory — will skip)"`.

**11. Default run duration** (`seconds`)
- `question`: "Default per-run time budget?"
- `options`: `300`, `600`, `900`, `1800`, `"Custom"`.

**12. Runner** (`runner`)
- `question`: "How should `ar` launch training?"
- `options`: Phase 1's inferred runner first (pre-selected), then alternatives, then `"Custom"`. Examples:
  - `accelerate launch --config_file configs/accelerate_8gpu.yaml`
  - `torchrun --nproc-per-node 8`
  - `python -m <your_project>.training.main` (Hydra apps often do this)
  - `python`
  - `custom`
- Stored verbatim.

**13. wandb project** (`wandb_project`)
- Skip entirely if metric backend (item 6) is not `wandb` or `auto`.
- `question`: "Which wandb project should runs log to?" — with detected preview if Phase 1 found one.
- `options`: detected value (pre-selected), then `"Other (type)"`. Optional — accept empty.
- Becomes `--wandb-project`.

### Structuring the AskUserQuestion call

The 13 items go into **one** `AskUserQuestion` invocation (the tool supports multiple items per call). Items 2, 3, and 13 are conditionally relevant based on items 1 and 6. The skill may split into two rounds — first item 1 alone (to branch on entry pattern), then items 2–13 with the CLI-args / resume-flag / wandb-project items included or omitted accordingly. Both approaches are acceptable; the one-call variant is preferred when Phase 1 detected a high-confidence entry pattern + metric backend.

The `custom` metric-backend path and the `tensorboard` events-glob capture are follow-up AskUserQuestion calls by design — each needs free-text input that doesn't fit the main interview's option/multi-select schema. These follow-ups are exempt from the "three permitted prompts" rule.

### Why these items

Items 1–3 + 12 exist because a v0.1-style flow would generate a TODO-stub `train.py` that crashes on first `ar run`. Capturing the entry pattern, baseline overrides, resume anchor, and runner lets `ar init` render a working wrapper-mode `train.py` that actually executes on iteration 1.

Items 4 + 6 + 7 are the v0.3.0 framework-agnostic additions: they decouple the rendered templates from the HF-Trainer-plus-wandb assumption baked into earlier versions, letting any mainstream pipeline (HF, Hydra, Lightning, plain PyTorch, DeepSpeed, FSDP) drop in with the right metric probe and checkpoint discovery wired up.

Items 5 + 8–11 + 13 are the minimum program-spec items needed to stamp program.md with enough context for the agent's loop.

See `references/interview-questions.md` for verbatim option copy and direction-inference heuristics.

---

## Phase 3 — Slug and collision handling

### Slug generation

Format: `{YYMMDD}-{kebab}`, lowercase, ASCII only.

- `YYMMDD` = today's date in local time.
- `kebab` = kebab-case of either:
  - the `[topic]` positional argument (`$ARGUMENTS`), if provided; or
  - the chosen goal phrase from interview item 5, truncated to 4–5 meaningful words.

Strip articles, stopwords, punctuation. Collapse whitespace to `-`. Max kebab length: 40 chars.

### Collision handling

If `.autoresearch/{slug}/` already exists, `AskUserQuestion` with: "Reuse existing / New slug with suffix / Abort". This is the third and final permitted mid-flow prompt (after Phase 0 remediation and Phase 2 interview).

---

## Phase 4 — Bootstrap `ar.py` shim (first-time-only)

Before calling `ar init`, check whether `.autoresearch/ar.py` exists. If it does, skip this phase. Otherwise:

```bash
mkdir -p .autoresearch
```

Write the shim (unchanged from v0.1) to `.autoresearch/ar.py` via the `Write` tool:

```python
#!/usr/bin/env python
"""Launcher shim for the autoresearch plugin.

Resolves the plugin's lib/ directory, injects it on sys.path, and delegates
to ar.cli.main(). Runs under the target project's uv-managed venv so torch,
and any framework the host uses (accelerate / lightning / hydra / etc.), are
all resolvable.

Deps `jinja2` and `psutil` are not part of typical ML project dependencies, so
if they're missing in the active interpreter, this shim re-exec's itself via
`uv run --with jinja2 --with psutil python <self> <args>`. The re-exec is
self-suppressing (AR_SHIM_BOOTSTRAPPED sentinel) to avoid infinite loops.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_deps_if_needed() -> None:
    if os.environ.get("AR_SHIM_BOOTSTRAPPED") == "1":
        return
    missing: list[str] = []
    try:
        import jinja2  # noqa: F401
    except ImportError:
        missing.append("jinja2")
    try:
        import psutil  # noqa: F401
    except ImportError:
        missing.append("psutil")
    if not missing:
        return
    with_flags: list[str] = []
    for pkg in missing:
        with_flags.extend(["--with", pkg])
    new_env = dict(os.environ)
    new_env["AR_SHIM_BOOTSTRAPPED"] = "1"
    cmd = ["uv", "run", *with_flags, "python", __file__, *sys.argv[1:]]
    os.execvpe("uv", cmd, new_env)


def _resolve_plugin_lib() -> Path:
    override = os.environ.get("AR_PLUGIN_ROOT")
    if override:
        lib = Path(override).expanduser().resolve() / "lib"
        if (lib / "ar" / "cli.py").exists():
            return lib
        raise SystemExit(f"AR_PLUGIN_ROOT set but {lib}/ar/cli.py not found")

    home = Path.home() / ".claude" / "plugins"
    candidates = sorted(
        home.rglob("autoresearch/plugin/lib/ar/cli.py"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].parent.parent
    raise SystemExit(
        "autoresearch plugin not found; set AR_PLUGIN_ROOT or install the plugin"
    )


_bootstrap_deps_if_needed()
sys.path.insert(0, str(_resolve_plugin_lib()))
from ar.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
```

### Why this design

- `AR_PLUGIN_ROOT` is the local-dev escape hatch: export it when iterating on the plugin source itself.
- The `rglob` fallback tolerates the nested `~/.claude/plugins/<cache>/<marketplace>/<plugin-name>/` layout without hardcoding paths.

---

## Phase 5 — Call `ar init`

With the shim in place, materialize the expr. Build the command with the flags derived from the interview:

```
uv run python .autoresearch/ar.py init \
  --expr {slug} \
  --goal "<from interview 5>" \
  --primary-metric "<from interview 8 name>" \
  --primary-direction <min|max> \
  --runner "<from interview 12 verbatim>" \
  --seconds <from interview 11> \
  [--constraints "<name> <op> <threshold>" ...] \
  [--mutation-scope "<comma,separated,dotted.paths>"] \
  [--entry-pattern <argparse-cli|function|hydra|custom>] \
  [--entry-main-module <dotted.path>] \
  [--cli-args-json '<JSON value>'] \
  [--distributed-framework <accelerate|deepspeed|fsdp|ddp|lightning|none|auto>] \
  [--resume-flag-name <flag_or_override_key_without_leading_dashes>] \
  [--metric-backend <wandb|tensorboard|log|custom|auto>] \
  [--tb-events-glob '<glob>'] \
  [--metric-extract-code '<base64-encoded python body>'] \
  [--checkpoint-glob '<glob>'] \
  [--wandb-project <name>]
```

### Flag semantics (must match `ar init` exactly)

Core flags:

- `--expr` — the generated slug. Required.
- `--goal` — free-form goal string. Required. Quote it.
- `--primary-metric` — metric key verbatim (e.g. `val/loss`, `val_loss`, `eval/top1`). Required.
- `--primary-direction` — literal `min` or `max`. Required.
- `--runner` — full invocation string. Required.
- `--seconds` — integer seconds. Required.
- `--constraints` — repeatable; each value is `"name op value"`. Optional.
- `--mutation-scope` — single flag, comma-separated dotted paths. Optional.

Entry-point flags:

- `--entry-pattern` — `argparse-cli` / `function` / `hydra` / `custom`. Default is equivalent to `custom`. Set to `argparse-cli` or `hydra` to render the wrapper-mode `train_wrapper.py.jinja`; Hydra mode additionally formats `CLI_OVERRIDES` as a list of override strings.
- `--entry-main-module` — dotted path for argparse-cli / hydra / function modes, e.g. `myproject.training.main`. **Required when `--entry-pattern` is `argparse-cli` or `hydra`**; `ar init` rejects the call otherwise.
- `--cli-args-json` — single JSON value string. Dict for argparse-cli (becomes `CLI_OVERRIDES = {...}`), list-of-strings for hydra (becomes `CLI_OVERRIDES = [...]`). Default `{}`.
- `--resume-flag-name` — string without `--` prefix (e.g. `resume_from_checkpoint`) for argparse-cli, or a Hydra override key (e.g. `trainer.resume_from_checkpoint`) for hydra. If unset, the wrapper exports `AR_RESUME_CKPT` env var but doesn't inject a flag/override.
- `--distributed-framework` — one of `accelerate` / `deepspeed` / `fsdp` / `ddp` / `lightning` / `none` / `auto`. Default `auto` (new v0.3.0 default — `ar run` introspects the `--runner` command at launch time).

Metric + checkpoint flags (v0.3.0):

- `--metric-backend` — `wandb` / `tensorboard` / `log` / `custom` / `auto`. Default `auto`. Drives how `ar run` extracts the primary metric after each run:
  - `wandb` → read `wandb/run-*/files/wandb-summary.json` via pointer file (v0.2.0 behavior).
  - `tensorboard` → expand `--tb-events-glob` and read the last scalar event for `--primary-metric`.
  - `log` → regex-scan `run.log` for a user-supplied pattern (default: `{primary_metric}=<float>`).
  - `custom` → eval the body from `--metric-extract-code` with `run_dir` and `run_log_text` in scope.
  - `auto` → try wandb → tensorboard → log in order; first non-empty result wins.
- `--tb-events-glob` — glob relative to project cwd, e.g. `runs/*/events.out.tfevents.*` or `lightning_logs/*/events.out.tfevents.*`. Required when `--metric-backend=tensorboard`.
- `--metric-extract-code` — base64-encoded Python body. Required when `--metric-backend=custom`; ignored otherwise. `ar init` decodes and embeds into program.md + train.py so both are self-contained.
- `--checkpoint-glob` — glob relative to project cwd, e.g. `output_dir/checkpoint-*/`, `lightning_logs/*/checkpoints/*.ckpt`, `outputs/*/model.safetensors`. Drives AR-SAVE's priority-0 path: the wrapper expands this glob after the training subprocess exits, picks the newest match, and copies into `runs/{id}/state.pt` (or `state/` for a directory). If unset, AR-SAVE uses the priority-1 in-process torch-state capture.
- `--wandb-project` — project name string. Only relevant when metric backend is `wandb` or `auto`. Becomes the `wandb.init(project=...)` value in wrapper mode.

Do NOT pass `--parent-ckpt` — that flag is reserved for `chain-init`.

### What `ar init` does

Source of truth for:

- Creating `.autoresearch/{slug}/` with `program.md`, `prepare.py`, `train.py` (or `train_wrapper.py` body if wrapper mode), empty `runs/`, `results.tsv` with header, and `.ar-session.json`.
- Selecting train.py template by `--entry-pattern`:
  - `argparse-cli` → `train_wrapper.py.jinja` in argparse mode (uses `runpy.run_module` with synthesized `sys.argv`).
  - `hydra` → `train_wrapper.py.jinja` in hydra mode (invokes the entry module with a list of `key=value` overrides).
  - `function` / `custom` → legacy `train.py.jinja` (function-mode or pure-agent-wiring entry).
- Stamping `program.md` with the "Entry point" section (pattern, main module, metric backend, checkpoint glob, distributed framework, resume flag, baseline overrides).
- Picking prepare.py mode (thin-wrapper vs full-prep).

The skill does NOT render any templates itself. All file materialization is `ar init`'s job.

### Error handling

- Non-zero exit from `ar init`: print the tail of its stderr (10–20 lines) and stop. Common causes: missing `pyproject.toml`, broken venv, `--entry-pattern=argparse-cli` or `hydra` without `--entry-main-module`, `--metric-backend=tensorboard` without `--tb-events-glob`, `--metric-backend=custom` without `--metric-extract-code`. Do not retry automatically.
- If `uv` is not installed: print `install uv: https://docs.astral.sh/uv/` and stop.

---

## Phase 6 — Completion

On successful `ar init`, print exactly one next-step line:

```
Next: /autoresearch:run {slug}
```

Nothing else. Do not summarize interview answers or list created files — the user knows what they asked for. Users can inspect `.autoresearch/{slug}/program.md` (and the new "Entry point" section there) for the full record.

---

## Rules of engagement

- **Three permitted prompts only.** Phase 0 remediation (if probe red), Phase 2 interview, Phase 3 collision. No other AskUserQuestion calls — with two exempted micro-followups: (a) when Phase 2 item 6 chooses `tensorboard`, one free-text prompt for the events glob; (b) when item 6 chooses `custom`, one free-text prompt for the Python extraction snippet. Both are mechanical captures of free-text input that the main interview's option schema cannot carry.
- **Read budget applies to Phase 1 only.** Once the interview starts, no further project reads unless `ar init` fails.
- **Write budget: exactly one file via the `Write` tool.** That file is `.autoresearch/ar.py`, only if absent. Everything else is created by `ar init` via `Bash`.
- **No git operations.** Setup does not commit, branch, or stash.
- **Honor `$ARGUMENTS`.** If invoked with a topic argument, use it as the kebab seed and as additional context for goal candidates — but still run the full interview.
- **Phase 0 cannot be skipped.** Even if the user has run setup before in the same session, re-run the preflight — `.venv` state can change between invocations.

---

## Resources

- **`references/venv-preflight.md`** — Complete failure-mode table + detection recipes for Phase 0. Read when step 0.2 red path needs diagnosis beyond the inline flowchart.
- **`references/project-analysis.md`** — Extended heuristics for Phase 1 on unusual project layouts. Read if any of {goal, metric, scope, runner, entry-pattern} produced fewer than 2 strong candidates.
- **`references/interview-questions.md`** — Verbatim `AskUserQuestion` question/option copy + direction-inference table. Read before constructing the interview call if uncertain about wording.
