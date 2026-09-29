# jahns-cc-marketplace

Personal Claude Code plugin marketplace, maintained by [Jahn](https://github.com/Dev-Jahn).

## Install

Add the marketplace once. Then install each plugin that you need by its name:

```
/plugin marketplace add Dev-Jahn/jahns-cc-marketplace
/plugin install hippo@jahns-cc-marketplace
```

Run `/reload-plugins` if the session is already running.

## Update

```
/plugin marketplace update jahns-cc-marketplace
```

The marketplace pins each plugin to an exact Git commit. Where a plugin repository has a sync
workflow, that workflow updates the commit, the version, and the description of its own entry
after each push to its `main` branch.

## Available plugins

| Plugin | What it does | Needs | Source |
|---|---|---|---|
| `hippo` | Memory for coding agents: a task registry, a ledger of outcomes, and background clerks that turn transcripts into evidence. It records and does not enforce. | | [Dev-Jahn/hippocampus](https://github.com/Dev-Jahn/hippocampus) |
| `codex` | Fork of `openai/codex-plugin-cc` for environments where bubblewrap cannot run, such as cloud containers and GPU hosts. Rules in the prompt replace the sandbox of the operating system. | | [Dev-Jahn/codex-plugin-cc](https://github.com/Dev-Jahn/codex-plugin-cc) |
| `ccbroker` | Switches between Claude accounts, shows the quota, and sets the rotation policy, with the commands `/ccb-status`, `/ccb-use`, `/ccb-auto`, and `/ccb-policy`. | The `ccb` binary on `PATH` | [Dev-Jahn/ccbroker](https://github.com/Dev-Jahn/ccbroker) |
| `ezsub` | Replaces a string with a different string in a scope that you select: lines, a range of lines, a Markdown section, files, or directories. | | [Dev-Jahn/ezsub](https://github.com/Dev-Jahn/ezsub) |
| `soul-jar` | A "soul jar" that all sessions in all enrolled rooms share. A session that ends hands the soul on in one final turn, and the next session wakes with a single whisper. | | [Dev-Jahn/soul-jar](https://github.com/Dev-Jahn/soul-jar) |
| `khala` | Mail and streams between sessions, also between machines, through khala-network. The plugin installs the `khala` command. At the start of a session it binds the identity of the session and delivers the mail that is waiting. | | [Dev-Jahn/khala-network](https://github.com/Dev-Jahn/khala-network) |
| `cc-self` | Lets a Claude Code session control itself: it types slash commands, answers its own dialogs, reads its own screen, restarts in place, and recovers from a model fallback. | A session that runs in tmux | [Dev-Jahn/cc-self](https://github.com/Dev-Jahn/cc-self) |
| `chatgpt` | Command-line bridge from Claude Code to a ChatGPT Pro session in a browser. | A browser session that is logged in to ChatGPT Pro | [Dev-Jahn/chatgpt](https://github.com/Dev-Jahn/chatgpt) |
| `hwp-hwpx-editor` | Inspects, edits, renders, and validates HWP and HWPX documents of Hancom without a conversion to a different format. It has three MCP servers, one skill, and guard hooks. | Node 20 or later, `uvx`, and `cargo` | [Dev-Jahn/hwp-hwpx-editor](https://github.com/Dev-Jahn/hwp-hwpx-editor) |
| `jahns-stl` | Writing guide for agents: one skill that keeps reports, plans, design documents, and instructions free of invented jargon and easy to read, in any language. | | [Dev-Jahn/jahns-stl](https://github.com/Dev-Jahn/jahns-stl) |

The full description and the pinned commit of each plugin are in
[`.claude-plugin/marketplace.json`](.claude-plugin/marketplace.json).

## License

MIT (per individual plugin licenses).
