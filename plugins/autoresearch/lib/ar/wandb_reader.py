"""Backward-compat alias for `metric_reader`.

Historically (v0.1.x / v0.2.x) this module owned the wandb pointer + log-scan
readers directly. v0.3.0 renamed it to `metric_reader` so the same module can
house tensorboard and custom-backend helpers. We keep `wandb_reader.py` as a
thin re-export so any external caller (tests, downstream plugins) that imported
the wandb-specific helpers by their historical names keeps working without a
breaking API change.
"""

from __future__ import annotations

from .metric_reader import (  # noqa: F401
    read_from_log,
    read_pointer,
    read_via_pointer,
)

__all__ = ["read_from_log", "read_pointer", "read_via_pointer"]
