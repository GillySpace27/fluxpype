"""Execution backends. Local, fake (simulated), and remote (SSH, Slurm; see remote.py)."""

from .base import Backend, BackendCaps, Handle, TaskSpec, TaskStatus  # noqa: F401
