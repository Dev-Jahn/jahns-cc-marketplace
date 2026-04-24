"""ar — Python helper CLI for the autoresearch Claude Code plugin.

The package is imported by the target project via `.autoresearch/ar.py`, which
inserts this directory on sys.path and calls `ar.cli.main()`. All heavy logic
(run orchestration, atomic swaps, wandb parsing, watchdog) lives here so the
main agent loop only needs to consume small structured JSON outputs.
"""

from .cli import main

__version__ = "0.3.0"
__all__ = ["main", "__version__"]
