---
name: autoresearch:setup
description: Scaffolds a new autonomous-research experiment directory (`.autoresearch/{YYMMDD}-{slug}/`) inside a deep-learning project so Claude can run a long train.py-mutation loop without blowing context. This skill should be used when the user asks to "start an autoresearch experiment", "set up autonomous research loop on this project", "create a new .autoresearch run", "scaffold autoresearch", "initialize autoresearch for this repo", "kick off an autonomous training loop", "set up Karpathy-style autoresearch here", or otherwise indicates they want Claude to begin autonomous iteration on their ML research code. The skill performs a venv preflight, analyzes the project's editable-install Python packages, scans recent `wandb/run-*/files/wandb-summary.json` files to surface primary-metric candidates, introspects the host's training entrypoint (argparse-CLI script vs importable main() function), infers the distributed framework (accelerate / torchrun / FSDP / DDP / none) and resume-flag convention, captures baseline CLI args, runs a short interview, and then materializes the expr by calling `ar init` which renders the train.py / prepare.py / program.md templates.
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

### Step 0.2 — venv + core deps

```bash
uv run --no-sync python -c "import torch, accelerate, wandb" 2>&1
```

The `--no-sync` flag prevents `uv` from silently mutating the environment during the probe.

- **Exit 0, no output:** proceed to step 0.3.
- **Non-zero exit:** diagnose. See "Diagnosis flowchart" below. When consent is needed, use `AskUserQuestion`.

### Step 0.3 — wandb login (soft)

```bash
(test -f ~/.netrc && grep -q "api.wandb.ai" ~/.netrc) && echo "ok" || echo "missing"
```

If missing, warn once:

> Note: `wandb login` does not appear to be set up (`~/.netrc` lacks an `api.wandb.ai` entry). Training will still run, but metrics will be offline-only. Run `uv run wandb login` any time to enable — does not block setup.

Proceed regardless — this is a soft signal.

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
    uv run --project "$cand" --no-sync python -c "import torch, accelerate, wandb" 2>/dev/null && \
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

Proceed to Phase 1 only if steps 0.1 and 0.2 are both green. 0.3 is a soft warning that never blocks.

---

## Phase 1 — Project analysis

Goal: gather enough signal to propose strong defaults for the interview. Treat this phase as read-only reconnaissance. Target ≤ 12 reads total in this phase (up from 10 in v0.1 because the entry-point + CLI-args probes require extra reads); prefer `Glob`/`Grep` over full-file reads. Do not read files already in session context.

### Signals to collect

Read / glob these in priority order, and stop early once proposals are well-formed:

1. **`pyproject.toml`** — Read full. Extract project name, dependencies (torch / accelerate / wandb / deepspeed presence), package roots (`[tool.setuptools.packages.find]` / `[project.scripts]` / `[tool.uv]`).

2. **`CLAUDE.md`** (project-level, plus any under subpackages) — Read if present. Extract:
   - The project's own "how to run training" command block (usually under a "Commands" or "Training" heading). **This is the strongest signal for both the runner and the CLI args** — the exact command the user types to train locally.
   - Project description / research themes for goal candidates.

3. **`wandb/`** directory — Glob `wandb/run-*/files/`. For up to 3 of the most recent runs, read:
   - `wandb-summary.json` — metric keys (union across samples), last numeric values.
   - `config.yaml` — the flattened training config (argparse namespace). **This is the strongest signal for `--cli-args-json`** — copy scalar keys directly. Also contains the wandb project name via the `_wandb.project` or enclosing metadata.
   - `wandb-metadata.json` — `program`, `args`, `username`, `root` — surfaces the exact entry script and argv.

4. **Entry-point training scripts** — Glob `training/**/*.py`, `train.py`, `scripts/train*.py`, `tools/train*.py`. For each, check:
   - `if __name__ == "__main__":` block present? → candidate for **argparse-cli** pattern.
   - Contains `argparse.ArgumentParser(` or `HfArgumentParser(` or `simple_parsing`? → strongly argparse-cli. Record the dotted-module path (e.g. `pivot.training.main`).
   - Defines a top-level `def main(...)` with typed kwargs and is imported elsewhere? → candidate for **function** pattern.
   - Neither clearly → **custom** pattern (agent will wire it manually in train.py).
   - **Rank candidates by:** (a) presence of ArgumentParser + `__main__` block wins over (b) `main()` function; ties broken by which file's path appears in `CLAUDE.md`'s training command.

5. **Distributed framework** — grep the training entrypoint (and nearby modules) for:
   - `from accelerate` / `Accelerator(` → **accelerate**
   - `import deepspeed` / `deepspeed.initialize` / `deepspeed_config` → **deepspeed**
   - `FullyShardedDataParallel` / `FSDP(` / `torch.distributed.fsdp` → **fsdp**
   - `torch.distributed.init_process_group` / `DistributedDataParallel` without the above → **ddp**
   - Nothing dist-related → **none**

6. **Resume flag detection** — grep the training entrypoint for argparse definitions matching `add_argument\([\"']--?resume` or similar. Common flag names in order of likelihood: `resume_from_checkpoint`, `resume`, `resume_checkpoint`, `ckpt_path`, `pretrained_checkpoint`. Record the detected flag **without** the `--` prefix. If multiple candidates or none, mark as "unclear — ask user".

7. **Checkpoint save convention** — grep for `accelerator.save_state(`, `trainer.save_state(`, `torch.save(`, `model.save_pretrained(`. Record the dominant pattern; feeds the AR-SAVE block's checkpoint-discovery heuristics in wrapper mode (see `train_wrapper.py` (a)/(b)/(c) candidates).

8. **`accelerate` / `deepspeed` configs** — Glob `configs/accelerate*.yaml`, `accelerate_config*.yaml`, `**/accelerate/*.yaml`. If found, note exact filenames — used verbatim in the runner spec.

9. **Recent git commits** — `git log --oneline -n 20` if inside a git repo. Use commit messages as additional signal for goal candidates.

### What to derive

From the above signals, derive the proposal lists that feed the interview:

| Proposal | Source | Output |
|---|---|---|
| Goal candidates (3–4) | CLAUDE.md + recent commits + top-level module names | Short phrases. |
| Primary-metric candidates | wandb-summary.json keys (union) | Ordered list with inferred direction. |
| Mutation-scope candidates | Training entrypoint imports + obvious module roots | Dotted paths. |
| Runner inference | CLAUDE.md + accelerate configs + entrypoint | `accelerate launch --config_file ...`, `torchrun --nproc-per-node N`, `python`, `custom`. |
| **Entry pattern** (new) | Signal 4 | `argparse-cli` / `function` / `custom`. |
| **Entry main module** (new) | Signal 4 | Dotted path, e.g. `pivot.training.main`. Only required if pattern is argparse-cli. |
| **CLI args baseline** (new) | wandb config.yaml + CLAUDE.md command block | Flat JSON-like dict of scalar flag values (e.g. `{"config":"base_flat","learning_rate":1.5e-4}`). Pick the top 5–10 most-distinctive keys; skip obviously-cosmetic ones (`log_level`, `wandb_mode`). |
| **wandb project** (new) | wandb config.yaml (`_wandb.project`) | Project name string. |
| **Distributed framework** (new) | Signal 5 | `accelerate` / `deepspeed` / `fsdp` / `ddp` / `none`. |
| **Resume flag name** (new) | Signal 6 | String without `--` prefix, or null. |

prepare.py mode selection (thin-wrapper vs full-prep) stays unchanged from v0.1 — `ar init` handles it based on presence of `build_train_loader` / data-module imports.

### Read budget

Soft cap: 12 reads across this phase. If probes 4–6 produce enough signal on the first training script read, skip additional entry-point candidates. If no wandb runs exist, skip signal 3 entirely (the user fills CLI args manually in the interview).

Do not read the same file twice. Do not read `.pt` / `.safetensors` / `wandb/run-*/logs/` — large and worthless.

See `references/project-analysis.md` for extended grep patterns if any of {goal, metric, scope, runner, entry-pattern} produced fewer than 2 candidates.

---

## Phase 2 — Interview

One `AskUserQuestion` call with the items below, in this exact order. Use Phase 1 detections to pre-populate `options`; always include a free-text escape hatch. Present detected values with a preview so the user can confirm or override.

### Items

**1. Entry pattern** (`entry_pattern`) — NEW in v0.2.0
- `question`: "What shape is your training entrypoint?" — include a preview line showing the detected main module, e.g. `Detected: pivot.training.main (argparse.ArgumentParser present + __main__ block)`.
- `options`:
  - `"argparse-cli — runpy-invoke an `if __name__ == '__main__':` script"` (pre-selected if Phase 1 detected ArgumentParser + `__main__`)
  - `"function — call an importable main(**kwargs)"`
  - `"custom — I'll wire it manually in train.py"`
- If user picks `argparse-cli`, the detected main module feeds `--entry-main-module`. If detection was ambiguous, ask as a follow-up free-text.

**2. Baseline CLI args** (`cli_args`) — NEW in v0.2.0
- Skip entirely if user picked `function` or `custom` in item 1.
- `question`: "These are the baseline CLI args the agent will start from (edit `CLI_OVERRIDES` in `train.py` per run). Keep as-is, or edit?" — show the detected dict as preview, e.g.
  ```
  {
    "config": "base_flat",
    "datasets": "openvid_wds",
    "learning_rate": 1.5e-4,
    "num_train_epochs": 1
  }
  ```
- `options`:
  - `"Keep as-is"` (pre-selected)
  - `"Edit — I'll paste a JSON dict"` (free-text; accept multi-line JSON)
  - `"Empty — start with {}"` (useful if agent should rebuild from scratch)
- Record as a JSON string; becomes `--cli-args-json`.

**3. Resume flag** (`resume_flag_name`) — NEW in v0.2.0
- Skip entirely if user picked `function` or `custom` in item 1.
- `question`: "Which argparse flag does your training script accept to resume from a checkpoint?" — with detected preview if Phase 1 found one.
- `options`: detected flag first (pre-selected, shown as e.g. `resume_from_checkpoint (detected)`), then common alternatives `resume`, `ckpt_path`, `pretrained_checkpoint`, `"None — no resume flag (use AR_RESUME_CKPT env var instead)"`, `"Other (type)"`.
- Record name without `--`. Becomes `--resume-flag-name`.

**4. wandb project** (`wandb_project`) — NEW in v0.2.0
- `question`: "Which wandb project should runs log to?" — with detected preview if Phase 1 found one.
- `options`: detected value (pre-selected), then `"Other (type)"`. Optional — accept empty.
- Becomes `--wandb-project`.

**5. Distributed framework** (`distributed_framework`) — NEW in v0.2.0
- `question`: "Which distributed-training framework does the host use?"
- `options`: Phase 1 detection first (pre-selected), then `accelerate`, `deepspeed`, `fsdp`, `ddp`, `none`.
- Becomes `--distributed-framework`.

**6. Research goal** (`research_goal`)
- `question`: "What should this experiment optimize for?"
- `options`: 3–4 candidate themes from Phase 1, plus `"Other (describe)"`. Free text accepted.
- Becomes `--goal`. Kebab form seeds the slug if `$ARGUMENTS` was not provided.

**7. Primary metric** (`primary_metric`)
- `question`: "Which wandb metric is the single source of truth for 'did this run improve'?"
- `options`: top metric candidates from Phase 1 with inferred direction, e.g. `val/loss (min)`, `eval/ucf101_top1 (max)`. Include `"Other (type metric=direction)"`.
- Becomes `--primary-metric` + `--primary-direction`.

**8. Hard constraints** (`hard_constraints`)
- `question`: "Any hard constraints? (Violations mark the run invalid regardless of primary.)"
- `options`: `"None"` (default), plus 2–3 common candidates if auxiliary metrics look like resource bounds (`peak_vram_mb`, `tokens_per_sec`). Multi-select + free-text.
- Parse each as `{name, op, threshold}`; operators `<=`, `<`, `>=`, `>`.

**9. Mutation scope** (`mutation_scope`)
- `question`: "Which modules/classes are in-scope for train.py to monkey-patch?"
- `options`: dotted-path candidates from Phase 1; multi-select + free-text. Include `"(advisory — will skip)"`.

**10. Default run duration** (`seconds`)
- `question`: "Default per-run time budget?"
- `options`: `300`, `600`, `900`, `1800`, `"Custom"`.

**11. Runner** (`runner`)
- `question`: "How should `ar` launch training?"
- `options`: Phase 1's inferred runner first (pre-selected), then alternatives, then `"Custom"`. Examples:
  - `accelerate launch --config_file configs/accelerate_8gpu.yaml`
  - `torchrun --nproc-per-node 8`
  - `python`
  - `custom`
- Stored verbatim.

### Structuring the AskUserQuestion call

The 11 items go into **one** `AskUserQuestion` invocation (the tool supports multiple items per call). Items 2 and 3 are conditionally relevant — if the user is expected to pick `function` or `custom` in item 1, the skill may choose to split into two rounds: first item 1 alone (to branch), then items 2–11 with the CLI-args / resume-flag items included or omitted accordingly. Both approaches are acceptable; the one-call variant is preferred when the detected entry pattern is high-confidence `argparse-cli`.

### Why these items

The new v0.2.0 items (1–5) exist because the v0.1 flow generated a TODO-stub `train.py` that crashed on first `ar run`. Capturing the entry pattern, baseline CLI args, and resume-flag name lets `ar init` render a working wrapper-mode `train.py` that actually executes on iteration 1.

The classic items (6–11) are unchanged from v0.1 — they remain the minimum needed to stamp program.md with enough context for the agent's loop.

See `references/interview-questions.md` for verbatim option copy and direction-inference heuristics.

---

## Phase 3 — Slug and collision handling

### Slug generation

Format: `{YYMMDD}-{kebab}`, lowercase, ASCII only.

- `YYMMDD` = today's date in local time.
- `kebab` = kebab-case of either:
  - the `[topic]` positional argument (`$ARGUMENTS`), if provided; or
  - the chosen goal phrase from interview item 6, truncated to 4–5 meaningful words.

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
wandb, accelerate, etc. are all resolvable.

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
  --goal "<from interview 6>" \
  --primary-metric "<from interview 7 name>" \
  --primary-direction <min|max> \
  --runner "<from interview 11 verbatim>" \
  --seconds <from interview 10> \
  [--constraints "<name> <op> <threshold>" ...] \
  [--mutation-scope "<comma,separated,dotted.paths>"] \
  [--entry-pattern <argparse-cli|function|custom>] \
  [--entry-main-module <dotted.path>] \
  [--cli-args-json '<JSON dict>'] \
  [--wandb-project <name>] \
  [--distributed-framework <accelerate|deepspeed|fsdp|ddp|none>] \
  [--resume-flag-name <flag_without_leading_dashes>]
```

### Flag semantics (must match `ar init` exactly)

Core flags (unchanged from v0.1):

- `--expr` — the generated slug. Required.
- `--goal` — free-form goal string. Required. Quote it.
- `--primary-metric` — wandb key verbatim (e.g. `val/loss`). Required.
- `--primary-direction` — literal `min` or `max`. Required.
- `--runner` — full invocation string. Required.
- `--seconds` — integer seconds. Required.
- `--constraints` — repeatable; each value is `"name op value"`. Optional.
- `--mutation-scope` — single flag, comma-separated dotted paths. Optional.

New v0.2.0 flags (all optional — unset means legacy `function`-mode rendering for back-compat):

- `--entry-pattern` — `argparse-cli` / `function` / `custom`. Default is equivalent to `custom`. Set to `argparse-cli` to render `train_wrapper.py.jinja`.
- `--entry-main-module` — dotted path for argparse-cli mode, e.g. `pivot.training.main`. **Required when `--entry-pattern=argparse-cli`**; `ar init` rejects the call otherwise.
- `--cli-args-json` — single JSON-object string. Becomes `CLI_OVERRIDES` dict in wrapper train.py. Default `{}`.
- `--wandb-project` — project name string.
- `--distributed-framework` — one of the choices. Default `accelerate`.
- `--resume-flag-name` — string without `--` prefix (e.g. `resume_from_checkpoint`). If unset, wrapper exports `AR_RESUME_CKPT` env var but doesn't inject a CLI flag.

Do NOT pass `--parent-ckpt` — that flag is reserved for `chain-init`.

### What `ar init` does

Source of truth for:

- Creating `.autoresearch/{slug}/` with `program.md`, `prepare.py`, `train.py` (or `train_wrapper.py` body if wrapper mode), empty `runs/`, `results.tsv` with header, and `.ar-session.json`.
- Selecting train.py template by `--entry-pattern`:
  - `argparse-cli` → `train_wrapper.py.jinja` (new in v0.2.0; uses `runpy.run_module`).
  - `function` / `custom` → legacy `train.py.jinja` (function-mode entry).
- Stamping `program.md` with the new "Entry point" section (pattern, main module, wandb project, distributed framework, resume flag, baseline CLI args).
- Picking prepare.py mode (thin-wrapper vs full-prep).

The skill does NOT render any templates itself. All file materialization is `ar init`'s job.

### Error handling

- Non-zero exit from `ar init`: print the tail of its stderr (10–20 lines) and stop. Common causes: missing `pyproject.toml`, broken venv, `--entry-pattern=argparse-cli` without `--entry-main-module`. Do not retry automatically.
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

- **Three permitted prompts only.** Phase 0 remediation (if probe red), Phase 2 interview, Phase 3 collision. No other AskUserQuestion calls.
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
