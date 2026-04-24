# Phase 0 preflight — recipes and troubleshooting

Reference for Phase 0 in `SKILL.md`. Read this only if the inline probe in
SKILL.md surfaced a red / ambiguous diagnosis.

## Fast-path probes

### 1. pyproject.toml at project root

```bash
test -f pyproject.toml && echo "ok" || echo "missing"
```

If missing, `autoresearch:setup` cannot continue: `ar init` expects an
editable-installable Python project.

### 2. .venv + mandatory core dep

```bash
uv run --no-sync python -c "import torch" 2>&1
```

`--no-sync` prevents `uv` from silently mutating the environment during the
probe. `torch` is the only hard requirement — every framework v0.3.0 supports
(HF Trainer, Hydra, Lightning, plain PyTorch, DeepSpeed, FSDP, custom) is
built on top of it. If the command exits 0 with no output, the venv is green.
Non-zero exit means one of:

- `.venv/` doesn't exist (`uv` will refuse: `error: No virtual environment found`)
- `torch` is missing from pyproject.toml (surface the import error verbatim)
- The venv exists but was created for a different Python version (surface the
  symlink-broken error verbatim — `uv` points at this clearly)

### 3. Optional-dep signals (soft, informational)

None of these block Phase 0 — they feed Phase 1 detection.

```bash
uv run --no-sync python -c "import accelerate"        2>/dev/null && echo "accelerate=present"        || echo "accelerate=absent"
uv run --no-sync python -c "import wandb"             2>/dev/null && echo "wandb=present"             || echo "wandb=absent"
uv run --no-sync python -c "import torch.distributed" 2>/dev/null && echo "torch_distributed=present" || echo "torch_distributed=absent"
uv run --no-sync python -c "import pytorch_lightning" 2>/dev/null && echo "lightning=present"         || echo "lightning=absent"
uv run --no-sync python -c "import lightning"         2>/dev/null && echo "lightning2=present"        || echo "lightning2=absent"
uv run --no-sync python -c "import tensorboard"       2>/dev/null && echo "tensorboard=present"       || echo "tensorboard=absent"
```

Absences feed defaults: `wandb=absent` → metric backend defaults to `log`;
`accelerate=absent` + `lightning=absent` + `torch_distributed=absent` →
distributed framework defaults to `none`.

### 4. wandb login state (only if wandb present)

Skip entirely if step 3 reported `wandb=absent`.

```bash
test -f ~/.netrc && grep -q "api.wandb.ai" ~/.netrc && echo "ok" || echo "missing"
```

Alternative (slower but authoritative):

```bash
uv run --no-sync python -c "import wandb; print(wandb.api.api_key is not None)"
```

Missing wandb login is NOT fatal — the loop can still run; only metrics written
via wandb will be offline-only. Surface as a warning, not a block.

## Git worktree diagnosis

The cwd is a worktree if either:

- `.git` is a **file** (not a directory) whose first line reads
  `gitdir: <path>/worktrees/<name>`
- The parent git repo's `.git/worktrees/*` directory contains an entry
  pointing back at cwd

Bash probe:

```bash
if [ -f .git ]; then
  head -1 .git | grep -q "^gitdir: " && echo "worktree"
elif [ -d .git ]; then
  echo "main-repo"
else
  echo "not-a-git-repo"
fi
```

### Sibling-worktree .venv reuse recipe

When the cwd is a worktree and `.venv` is missing, a sibling worktree (or the
main checkout) often already has a populated `.venv` that's schema-compatible
with the current branch. Detection:

```bash
# Resolve the main gitdir, then walk siblings looking for a populated .venv.
MAIN_GITDIR=$(git rev-parse --git-common-dir)
MAIN_WORKTREE=$(dirname "$MAIN_GITDIR")
for cand in "$MAIN_WORKTREE" "$MAIN_WORKTREE"/../*; do
  if [ -d "$cand/.venv" ] && [ -d "$cand/.git" ] || [ -f "$cand/.git" ]; then
    if uv run --project "$cand" --no-sync python -c "import torch" 2>/dev/null; then
      # Only torch is mandatory — optional deps vary per project
      echo "candidate: $cand/.venv"
    fi
  fi
done
```

If a candidate is found, suggest (DO NOT execute without user consent via
`AskUserQuestion`):

```bash
rm -rf .venv && ln -s /absolute/path/to/sibling/.venv .venv
```

Rationale: on a fresh worktree, re-running `uv sync` re-downloads torch +
accelerate + flash-attention (~4–8 GB depending on the project), which is
wasteful when a sibling has identical deps.

### Plain `uv sync` fallback

If no sibling venv is a candidate (or the user rejects the symlink suggestion):

```bash
uv sync
```

This is idempotent; if it succeeds, re-run probe 2 to confirm green.

## Halt conditions

Halt Phase 0 — do NOT proceed to Phase 1 — if any of the following are true:

- `pyproject.toml` missing
- Probe 2 still fails after the user attempted remediation
- The user explicitly declines to fix (print a one-line pointer back to this
  reference and exit)

Proceed to Phase 1 if:

- `pyproject.toml` present
- Probe 2 exits 0 (torch importable)
- Probes 3 and 4 produce informational signals only (any absence is fine)

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `No virtual environment found` | `.venv/` never created | `uv sync` |
| `ModuleNotFoundError: No module named 'torch'` | pyproject.toml missing torch dep OR venv out of sync | Check pyproject; `uv sync` |
| `RuntimeError: The detected CUDA version ... mismatches ...` | venv built against a different CUDA toolkit than driver | Recreate venv with matching `torch` build: delete .venv, `uv sync --refresh` |
| `.venv` symlink is broken (`readlink .venv` points nowhere) | sibling worktree was deleted | `rm .venv && uv sync` |
| Python version mismatch | `.python-version` changed, venv stale | `rm -rf .venv && uv sync` |
| `ModuleNotFoundError: No module named 'accelerate' / 'pytorch_lightning' / 'wandb' / 'tensorboard'` | Optional dep not installed in this project | Informational only — feeds Phase 1 detection. Do not block. If the user expected the dep to be present, surface the absence; otherwise the interview's `auto` defaults handle the fallback. |
| `wandb: (error) api_key not configured` | `wandb login` never run | `uv run wandb login` (warning only, not a block; skipped entirely if wandb itself is absent) |
