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
    """state: starting | running | exited | lost.

    ``lost`` means the process is gone with no exit code (the launcher died or
    the machine rebooted); the scheduler records that as ``crashed``.
    """

    state: str
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
