# jahns-cc-marketplace

Personal Claude Code plugin marketplace (Dev-Jahn).

## Plugins

- **jahns-workflow** — SSOT-anchored development workflow: a validated task registry with one
  naming convention, generated roadmaps, round-based progress, and an external review loop (packet,
  or a SHA-bound PR merge gate). [repo](https://github.com/Dev-Jahn/jahns-workflow)
- **codex** — Fork of `openai/codex-plugin-cc` with OS-level sandboxing replaced by prompt-level
  self-enforcement, for environments where bubblewrap can't run (cloud containers, GPU hosts).
  [repo](https://github.com/Dev-Jahn/codex-plugin-cc)

## Install

```
/plugin marketplace add Dev-Jahn/jahns-cc-marketplace
/plugin install jahns-workflow@jahns-cc-marketplace
```

Then `/reload-plugins` if the session is already running.

## License

MIT (per individual plugin licenses).
