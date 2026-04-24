---
name: autoresearch:setup
description: Scaffolds a new autonomous-research experiment directory (`.autoresearch/{YYMMDD}-{slug}/`) inside a deep-learning project so Claude can run a long train.py-mutation loop without blowing context. This skill should be used when the user asks to "start an autoresearch experiment", "set up autonomous research loop on this project", "create a new .autoresearch run", "scaffold autoresearch", "initialize autoresearch for this repo", "kick off an autonomous training loop", "set up Karpathy-style autoresearch here", or otherwise indicates they want Claude to begin autonomous iteration on their ML research code. The skill analyzes the project's editable-install Python packages, scans recent `wandb/run-*/files/wandb-summary.json` files to surface primary-metric candidates, infers the runner (accelerate / torchrun / python) from CLAUDE.md and entry-point scripts, proposes mutation-scope modules, runs a short 6-question interview, and then materializes the expr by calling `ar init` which renders the train.py / prepare.py / program.md templates.
argument-hint: [topic]
allowed-tools: Read, Bash, Glob, Grep, AskUserQuestion, Write
---

# autoresearch:setup

Scaffold a new `.autoresearch/{YYMMDD}-{slug}/` experiment in the current project so the autoresearch loop (driven by `autoresearch:run`) can begin.

The skill performs three jobs, in order:

1. **Analyze the project** to propose intelligent defaults for the interview.
2. **Run a short 6-question interview** via `AskUserQuestion` — the only user-facing interaction in this skill.
3. **Materialize the expr** by calling `ar init`, bootstrapping the `ar.py` shim first if this is the project's first setup.

The interview is the only authorized user prompt. Do not interrupt mid-flow for confirmations, clarifications, or "are you sure?" prompts outside of it. Re-entry collision (existing slug) is the one exception — it uses `AskUserQuestion` too.

---

## Phase 1 — Project analysis

Goal: gather enough signal to propose strong defaults for the interview. Treat this phase as read-only reconnaissance. Target ≤ 10 reads total in this phase; prefer `Glob`/`Grep` over full-file reads. Do not read files already in session context.

### Signals to collect

Read / glob these, in priority order, and stop early once proposals are well-formed:

1. **`pyproject.toml`** at project root — Read full. Extract:
   - Project name, dependencies (torch, accelerate, wandb presence).
   - `[tool.setuptools.packages.find]` / `[project.scripts]` / `[tool.uv]` — identifies the editable-install package roots.
2. **`CLAUDE.md`** (project-level, plus any under subpackages) — Read if present. Extract:
   - The project's own "how to run training" command block (usually under a "Commands" or "Training" heading). This is the strongest signal for the runner.
   - Project description / research themes for goal candidates.
3. **`wandb/`** directory — Glob `wandb/run-*/files/wandb-summary.json`. Read up to 3 of the most recent ones. Extract:
   - Flat list of metric keys across samples (union). Typical names: `val/loss`, `val/bpb`, `eval/*`, `train/loss`, `peak_vram_mb`, `tokens_per_sec`, etc.
   - For each candidate metric, note the last numeric value (for sanity).
4. **Entry-point training scripts** — Glob `training/**/*.py`, `train.py`, `scripts/train*.py`, `tools/train*.py`. Read a few to find:
   - The Python function that is the true training entrypoint (import path like `training.main.run_training`). Used for mutation-scope suggestions.
   - `accelerate.Accelerator(...)` call → accelerate runner.
   - `torch.distributed.init_process_group` or `torchrun` references → torchrun runner.
5. **`accelerate` / `deepspeed` configs** — Glob `configs/accelerate*.yaml`, `accelerate_config*.yaml`, `**/accelerate/*.yaml`. If found, note exact filenames — used verbatim in the runner spec.
6. **Recent git commits** — `git log --oneline -n 20` if inside a git repo. Use commit messages as additional signal for "what is the user actively working on" → feeds goal candidates.

### What to derive

From the above signals, derive the four proposal lists that feed the interview:

| Proposal | Source | Output |
|---|---|---|
| Goal candidates (3–4) | CLAUDE.md description + recent commits + top-level module names | Short phrases ("improve val/loss via loss-term ablation", "reduce VRAM in attention block", "explore projector depth"). |
| Primary-metric candidates | wandb-summary.json keys (union) | Ordered list with likely direction (loss/bpb/nll → min; acc/top1/auroc → max). |
| Mutation-scope candidates | Training entrypoint imports + obvious module roots | Dotted paths like `training.losses.UnifiedLoss`, `model.attn.NAGLABlock`. |
| Runner inference | CLAUDE.md command block + accelerate configs + entrypoint | One of: `accelerate launch --config_file <path>`, `torchrun --nproc-per-node <N>`, `python`, `custom`. |

Additionally, decide **prepare.py mode** (recorded later as an `ar init` hint — the actual rendering is done by `ar`):

- **thin-wrapper mode** if the project already ships dataset / dataloader factories (grep for `build_train_loader`, `DataLoader(`, `datasets.load_dataset`, `webdataset`, a `data/` or `datasets/` subpackage with `__init__.py`).
- **full-prep mode** otherwise (tiny single-file projects, notebooks-graduated-to-script, etc.).

The detection should be conservative: if in doubt, pick **thin-wrapper** — the generated stub can always be filled in by the user.

### Read budget

Soft cap: 10 reads across this entire phase. If the project is large (>50 candidate wandb runs, dozens of training scripts), do not exhaustively enumerate — sample. The interview lets the user override every inferred value, so the priority is "good enough to propose", not "exhaustively correct".

Do not read the same file twice in a single session. Do not read checkpoint files, `.pt` / `.safetensors`, or anything in `wandb/run-*/logs/` — they're large and worthless here.

See `references/project-analysis.md` for example grep patterns and a longer list of wandb key heuristics if the project does not match the common shape.

---

## Phase 2 — Interview

One `AskUserQuestion` call with six items, in this exact order. Use the analysis from Phase 1 to populate `options` with strong candidates; always include a free-text escape hatch where the spec permits.

### Items

**1. Research goal** (`research_goal`)
- `question`: "What should this experiment optimize for?"
- `options`: the 3–4 candidate themes derived from Phase 1, each as a short phrase. Add `"Other (describe)"` as the last option and accept free text.
- Record the chosen phrase verbatim — it becomes the `--goal` flag to `ar init`, and the kebab form of it seeds the slug if the user did not pass `$ARGUMENTS`.

**2. Primary metric** (`primary_metric`)
- `question`: "Which wandb metric is the single source of truth for 'did this run improve'?"
- `options`: top metric candidates from wandb summary scan, each labeled with inferred direction, e.g. `val/loss (min)`, `eval/ucf101_top1 (max)`. Include `"Other (type metric=direction)"` for custom.
- Record both `name` and `direction` (min/max). Becomes `--primary-metric` + `--primary-direction`.

**3. Hard constraints** (`hard_constraints`)
- `question`: "Any hard constraints? (Violations mark the run invalid regardless of primary.)"
- `options`: `"None"` (default), plus 2–3 common candidates if any auxiliary metric in Phase 1 looks like a resource bound (`peak_vram_mb`, `tokens_per_sec`, `wall_seconds`). Accept multi-select + free-text.
- Parse each selected constraint into `{name, op, threshold}`. Operators allowed: `<=`, `<`, `>=`, `>`. Becomes zero or more `--hard-constraint "<name> <op> <threshold>"` flags.

**4. Mutation scope** (`mutation_scope`)
- `question`: "Which modules/classes are in-scope for train.py to monkey-patch?"
- `options`: the dotted-path candidates from Phase 1; allow multi-select and free-text. Always add `"(advisory — will skip)"` so the user can defer this to later.
- Record as a comma-separated string. Becomes `--mutation-scope "dotted.path,dotted.path"`.

**5. Default run duration** (`seconds`)
- `question`: "Default per-run time budget?"
- `options`: `300`, `600`, `900`, `1800`, `"Custom"`.
- Becomes `--seconds N`.

**6. Runner** (`runner`)
- `question`: "How should `ar` launch training?"
- `options`: Phase 1's inferred runner first (pre-selected), then the other three, then `"Custom"`. Examples:
  - `accelerate launch --config_file configs/accelerate_8gpu.yaml`
  - `torchrun --nproc-per-node 8`
  - `python`
  - `custom`
- The exact string is stored — `ar` uses it verbatim, so include config-file args if applicable. Becomes `--runner "..."`.

### Why these six and no more

The spec deliberately limits the interview to six items to keep setup under 30 seconds of user time. Do not add questions for dataset path, model checkpoint, optimizer — those live in the target project already and are read by `prepare.py`. If the user asks about them, direct them to `prepare.py` after scaffolding completes.

See `references/interview-questions.md` for verbatim option copy and direction-inference heuristics for metric names.

---

## Phase 3 — Slug and collision handling

### Slug generation

Format: `{YYMMDD}-{kebab}`, lowercase, ASCII only.

- `YYMMDD` = today's date in local time (e.g. `260424` for 2026-04-24).
- `kebab` = kebab-case of either:
  - the `[topic]` positional argument (`$ARGUMENTS`), if provided by the user; or
  - the chosen goal phrase from interview item 1, truncated to the first 4–5 meaningful words.

Strip articles ("the", "a", "an"), stopwords, and punctuation. Collapse whitespace to `-`. Max kebab length: 40 chars.

Examples:
- goal = "improve val/loss via loss-term ablation" → slug = `260424-loss-term-ablation`
- `$ARGUMENTS` = "projector depth sweep" → slug = `260424-projector-depth-sweep`

### Collision handling

If `.autoresearch/{slug}/` already exists, call `AskUserQuestion` with three options:

1. **Reuse existing** — skip scaffolding, print the next-step pointer against the existing slug, and exit.
2. **New slug with suffix** — append `-v2`, `-v3`, ... until free, then proceed.
3. **Abort** — print a short explanation and exit without writing anything.

This is the only mid-flow prompt permitted after the main interview.

---

## Phase 4 — Bootstrap `ar.py` shim (first-time-only)

Before calling `ar init`, check whether `.autoresearch/ar.py` exists in the target project. If it does, skip this phase. If not, this is a first-time setup and the shim must be created.

### Create `.autoresearch/` directory

```bash
mkdir -p .autoresearch
```

### Write the shim

The shim is a small launcher that resolves the plugin's `lib/` directory and invokes `ar.cli.main()`. It must work whether the plugin was installed via the marketplace (under `~/.claude/plugins/...`) or via a local `--plugin-dir` override (via `AR_PLUGIN_ROOT`).

Write the following content verbatim to `.autoresearch/ar.py` using the `Write` tool:

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
    # Re-exec under `uv run --with ...` so the missing deps are injected
    # ephemerally for this invocation without mutating the target project's
    # pyproject.toml. We mark AR_SHIM_BOOTSTRAPPED=1 in the child env to
    # prevent recursion if uv still can't satisfy the deps.
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
    # Sort rglob hits by mtime descending so multiple installed versions
    # resolve deterministically to the most recently updated copy. Without
    # this, rglob iteration order is filesystem-dependent and can silently
    # flip between runs.
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

After writing, `chmod +x .autoresearch/ar.py` is not required (it's invoked via `uv run python`, not directly), but does no harm.

### Why this design

- `AR_PLUGIN_ROOT` env var is the user's escape hatch for local dev: if they're iterating on the plugin itself (e.g. a cloned working copy at `~/src/autoresearch/plugin`), they export `AR_PLUGIN_ROOT=~/src/autoresearch/plugin` and the shim picks it up immediately.
- The `rglob` fallback tolerates the nested `~/.claude/plugins/<cache>/<marketplace>/<plugin-name>/` layout without hardcoding the marketplace name.
- The shim resolves at import time, not at module-scan time, so moving / renaming the plugin doesn't leave stale absolute paths inside the target project.

---

## Phase 5 — Call `ar init`

With the shim in place, materialize the expr via the helper CLI. Build the command with the flags derived from the interview:

```
uv run python .autoresearch/ar.py init \
  --expr {slug} \
  --goal "<from interview 1>" \
  --primary-metric "<from interview 2 name>" \
  --primary-direction <min|max> \
  --runner "<from interview 6 verbatim>" \
  --seconds <from interview 5> \
  [--hard-constraint "<name> <op> <threshold>" ...] \
  [--mutation-scope "<comma,separated,dotted.paths>"]
```

### Flag semantics (must match `ar init` exactly)

- `--expr` — the generated slug (NOT the project root). Required.
- `--goal` — free-form goal string; becomes the header of `program.md`. Quote it.
- `--primary-metric` — the wandb key verbatim (e.g. `val/loss`, `eval/ucf101_top1`). Required.
- `--primary-direction` — literal `min` or `max`. Required.
- `--runner` — the full invocation string, e.g. `"accelerate launch --config_file configs/accelerate_8gpu.yaml"`. Required.
- `--seconds` — integer seconds for default per-run budget. Required.
- `--hard-constraint` — repeatable; each value is `"name op value"` (three space-separated tokens). Optional.
- `--mutation-scope` — single flag, comma-separated dotted paths. Optional; if the user chose `"(advisory — will skip)"` omit entirely.

Do NOT pass `--parent-ckpt` — that flag is reserved for chain-init, not setup.

### What `ar init` does

`ar init` is the source of truth for:

- Creating `.autoresearch/{slug}/` with `program.md`, `prepare.py`, `train.py`, empty `runs/`, empty `results.tsv` with header row, and `.ar-session.json` seeded with duration / runner / constraints.
- Picking prepare.py mode (thin-wrapper vs full-prep) based on its own detection logic; the skill's detection is advisory and not passed through as a flag.
- Verifying `pyproject.toml` is editable-installable and warning if wandb is missing.

The skill does NOT render any templates itself and does NOT create `program.md` / `prepare.py` / `train.py` — those are `ar init`'s job. The skill is purely orchestration.

### Error handling

- Non-zero exit from `ar init`: print the tail of its stderr (10–20 lines) and stop. Do not attempt to retry automatically — `ar init` failures usually mean a missing `pyproject.toml`, a broken venv, or the user is running the skill outside a project root. Those are user-fix situations.
- If `uv` is not installed: print a one-line hint (`install uv: https://docs.astral.sh/uv/`) and stop.

---

## Phase 6 — Completion

On successful `ar init`, print exactly one next-step line:

```
Next: /autoresearch:run {slug}
```

Nothing else. Do not summarize the interview answers, do not list the files that were created, do not offer to kick off the first run. The user knows what they asked for; the single command pointer is the most useful thing to show.

If the user wants to inspect the generated files, they can read `.autoresearch/{slug}/program.md` themselves — it contains the full goal / scope / metric / runner / constraints record.

---

## Rules of engagement

- **One interview, six items, no other interruptions.** The collision prompt in Phase 3 is the only other permitted `AskUserQuestion` call.
- **Read budget applies to Phase 1 only.** Once the interview starts, do no further Reads of the project unless `ar init` fails and the user needs diagnostic help.
- **Write budget: exactly one file via the `Write` tool.** That file is `.autoresearch/ar.py`, and only when it does not already exist. Everything else on disk is created by `ar init` via `Bash`.
- **No git operations.** Setup does not commit, branch, or stash. The user's git state is their concern.
- **Honor `$ARGUMENTS`.** If the user invoked the skill with a topic argument (`/autoresearch:setup projector depth sweep`), use it as the kebab seed for the slug and as additional context when proposing goal candidates — but still run the full interview. The argument is a hint, not a bypass.

---

## Resources

- **`references/project-analysis.md`** — Extended heuristics for reading large / unusual project layouts, plus grep patterns for detecting dataset pipelines and runners. Read this if Phase 1's quick scan produced fewer than 2 strong candidates for any of {goal, metric, scope, runner}.
- **`references/interview-questions.md`** — Verbatim `AskUserQuestion` question/option copy, plus the direction-inference table for mapping metric names → min/max. Read this before constructing the interview call if uncertain about wording or direction mapping.
