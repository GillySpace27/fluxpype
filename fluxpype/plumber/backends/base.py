"""The backend interface the scheduler talks to.

A backend starts one task (one ``flux-config-run --only ...`` process), reports
its state, can cancel it, and can read files the task wrote. SSH and Slurm
backends fit this shape (see docs/gui/SCOPE.md 4.5) but are not part of v1.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol


@dataclass
class BackendCaps:
    kind: str  # "local", "fake", later "ssh" / "slurm"
    slots: int  # CPU slots the backend offers
    shared_fs: bool = True  # events and artifacts readable without a transfer
    description: str = ""


@dataclass
class TaskSpec:
    task_id: str
    job_id: str
    run_dir: str
    snapshot_path: str
    profile: str
    cr: str
    nflux: str
    adapt: str
    method: str
    timeout_s: int = 2000
    extra: Dict[str, str] = field(default_factory=dict)  # backend-specific knobs (fake scenario, speed)


@dataclass
class Handle:
    task_id: str
    run_dir: str


@dataclass
class TaskStatus:
    """state: starting | queued | running | exited | lost | unknown.

    ``queued`` means a remote scheduler accepted the task but has not started it.
    ``unknown`` means the backend cannot tell right now (unreachable); the
    scheduler keeps the task running and never finalizes it on ``unknown``.
    ``lost`` means the process is gone with no exit code (the launcher died or
    the machine rebooted); the scheduler records that as ``crashed`` unless the
    task's events.jsonl holds a ``run.end`` record, which then decides.
    The local and fake backends never return ``queued`` or ``unknown``.
    """

    state: str  # starting | queued | running | exited | lost | unknown
    pid: Optional[int] = None
    pid_start: Optional[str] = None
    exit_code: Optional[int] = None
    signal: Optional[int] = None
    started: Optional[float] = None
    ended: Optional[float] = None
    cancelled: bool = False


class Backend(Protocol):
    name: str

    def capabilities(self) -> BackendCaps: ...

    def probe(self) -> List[str]:
        """Runner features the target install supports (``flux-config-run --capabilities``)."""

    def submit(self, task: TaskSpec) -> Handle: ...

    def poll(self, handle: Handle) -> TaskStatus: ...

    def cancel(self, handle: Handle, force: bool = False) -> None: ...

    def read_file(self, handle: Handle, name: str, offset: int = 0) -> bytes:
        """Bytes of ``run_dir/name`` from `offset` to the end ("" if the file does not exist yet)."""
