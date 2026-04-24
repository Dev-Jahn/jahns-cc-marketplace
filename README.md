# jahns-cc-marketplace

Personal Claude Code plugin marketplace (Dev-Jahn).

## Plugins

### autoresearch

Autonomous LLM-research loop adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) for real ML codebases — editable-install Python projects with `accelerate launch` / `torchrun` multi-GPU training and wandb metrics. Two skills (setup and run) plus an `ar` helper CLI keep the main Claude Code session context-light across hundreds to thousands of iterations.

See [`plugins/autoresearch/README.md`](plugins/autoresearch/README.md) for details.

## Install

```
/plugin marketplace add Dev-Jahn/jahns-cc-marketplace
/plugin install autoresearch@jahns-cc-marketplace
```

Then `/reload-plugins` if the session is already running.

## License

MIT (per individual plugin licenses).
