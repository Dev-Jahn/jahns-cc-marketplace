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

### 2. .venv + core deps

```bash
uv run --no-sync python -c "import torch, accelerate, wandb" 2>&1
```

`--no-sync` prevents `uv` from silently mutating the environment during the
probe. If the command exits 0 with no output, the venv is green. Non-zero exit
means one of:

- `.venv/` doesn't exist (`uv` will refuse: `error: No virtual environment found`)
- A dependency is missing from pyproject.toml (surface the import error verbatim)
- The venv exists but was created for a different Python version (surface the
  symlink-broken error verbatim — `uv` points at this clearly)

### 3. wandb login state

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
- Probe 2 exits 0
- (wandb login may be missing — warn, don't block)

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `No virtual environment found` | `.venv/` never created | `uv sync` |
| `ModuleNotFoundError: No module named 'torch'` | pyproject.toml missing torch dep OR venv out of sync | Check pyproject; `uv sync` |
| `RuntimeError: The detected CUDA version ... mismatches ...` | venv built against a different CUDA toolkit than driver | Recreate venv with matching `torch` build: delete .venv, `uv sync --refresh` |
| `.venv` symlink is broken (`readlink .venv` points nowhere) | sibling worktree was deleted | `rm .venv && uv sync` |
| Python version mismatch | `.python-version` changed, venv stale | `rm -rf .venv && uv sync` |
| `wandb: (error) api_key not configured` | `wandb login` never run | `uv run wandb login` (warning only, not a block) |
