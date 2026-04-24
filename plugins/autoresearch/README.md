# autoresearch (Claude Code plugin)

Autonomous LLM-research loop for real ML codebases. Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) but adapted to:

- Editable-install Python projects (`uv`, `pyproject.toml`, packages like `model`, `training`, `attn`)
- Multi-GPU launchers (`accelerate launch`, `torchrun`)
- wandb-instrumented training loops
- Overnight runs of hundreds–thousands of iterations in a single Claude Code session

## What it provides

Two skills (slash commands):

- **`/autoresearch:setup`** — scaffold an experiment directory under `{PROJECT_ROOT}/.autoresearch/{YYMMDD-slug}/`, interview the user for goal / primary metric / mutation scope / runner, and render `program.md`, `train.py`, `prepare.py` from templates.
- **`/autoresearch:run [slug]`** — drive the autonomous iteration loop. Agent edits only `train.py`; the `ar` helper CLI handles launching, wandb metric extraction, atomic checkpoint swap, and termination / chain-mode transitions.

Plus an `ar` Python CLI (in `lib/ar/`) that keeps main-session context light by absorbing all mechanical work.

## Design

See `docs/superpowers/specs/2026-04-24-autoresearch-plugin-design.md` in the repo root for the full spec, trilateral-reviewed and revised.

## Installation

Clone this repo, then either:

- **For local development of the plugin itself**: export `AR_PLUGIN_ROOT=/absolute/path/to/autoresearch/plugin` in the shell you use Claude Code from, and copy/symlink the `plugin/` directory into a location Claude Code discovers (typically `~/.claude/plugins/<marketplace>/<version>/autoresearch/`). Re-launch Claude Code so it picks up the new plugin directory.
- **Via a marketplace**: add this repo to a `.claude-plugin/marketplace.json` file Claude Code is configured to read.

After installation, `/autoresearch:setup` and `/autoresearch:run` appear as slash commands. On first run in a target project, `autoresearch:setup` creates `.autoresearch/ar.py` — a shim that discovers this plugin via `AR_PLUGIN_ROOT` (if set) or by scanning `~/.claude/plugins/**/autoresearch/plugin/lib`.

## Helper CLI discovery

The `.autoresearch/ar.py` shim placed in the target project discovers the plugin's `lib/ar/` at runtime:

1. `$AR_PLUGIN_ROOT` env var (if set) takes precedence.
2. Otherwise, scan `~/.claude/plugins/**/autoresearch/plugin/lib`.
3. If neither works, fail with a friendly error pointing at this README.

## License

MIT
