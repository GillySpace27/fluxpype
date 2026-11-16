"""Run tasks as detached local processes."""

import json
import os
import signal
import subprocess
import sys
import time
from typing import List

from .. import launcher, paths
from .base import BackendCaps, Handle, TaskSpec, TaskStatus

REQUIRED_FEATURES = ["only", "events", "config_file", "log_file", "task_timeout"]
START_GRACE_S = 30.0  # launcher has this long to write its pid file


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def spawn(run_dir: str, argv: List[str], env: dict, cwd: str) -> None:
    """Start `argv` detached under the launcher; its files (pid, exit_code, stdout.log) land in `run_dir`."""
    os.makedirs(run_dir, exist_ok=True)
    launcher.write_json_atomic(os.path.join(run_dir, "launch.json"), {"argv": argv, "env": env, "cwd": cwd})
    log = open(os.path.join(run_dir, "launcher.log"), "ab")
    subprocess.Popen(
        [sys.executable, "-m", "fluxpype.plumber.launcher", run_dir],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=str(paths.checkout_root()),
        env={**os.environ, "PYTHONPATH": env["PYTHONPATH"]},
    )
    log.close()


def poll_run_dir(run: str) -> TaskStatus:
    cancelled = os.path.exists(os.path.join(run, "cancelled"))
    done = _read_json(os.path.join(run, "exit_code"))
    pid = _read_json(os.path.join(run, "pid"))
    started = pid.get("started") if pid else None
    if done is not None:
        return TaskStatus(
            state="exited",
            pid=pid.get("pid") if pid else None,
            pid_start=pid.get("pid_start") if pid else None,
            exit_code=done.get("exit_code"),
            signal=done.get("signal"),
            started=started,
            ended=done.get("ended"),
            cancelled=cancelled,
        )
    if pid is not None:
        alive = launcher.process_alive(pid["pid"], pid.get("pid_start"))
        launcher_alive = launcher.process_alive(pid.get("launcher_pid"))
        if alive or launcher_alive:
            return TaskStatus(
                state="running",
                pid=pid["pid"],
                pid_start=pid.get("pid_start"),
                started=started,
                cancelled=cancelled,
            )
        # the launcher may have written exit_code between our two reads
        done = _read_json(os.path.join(run, "exit_code"))
        if done is not None:
            return poll_run_dir(run)
        return TaskStatus(
            state="lost", pid=pid["pid"], pid_start=pid.get("pid_start"), started=started, cancelled=cancelled
        )
    try:
        submitted = os.path.getmtime(os.path.join(run, "launch.json"))
    except OSError:
        return TaskStatus(state="lost", cancelled=cancelled)
    if time.time() - submitted > START_GRACE_S:
        return TaskStatus(state="lost", cancelled=cancelled)
    return TaskStatus(state="starting", cancelled=cancelled)


class LocalBackend:
    name = "local"
    kind = "local"
    description = "Detached processes on this machine"

    def __init__(self, python: str = None, cpu_slots: int = None):
        self.python = python or sys.executable
        self.cpu_slots = cpu_slots or os.cpu_count() or 1

    # -- interface
    def capabilities(self) -> BackendCaps:
        return BackendCaps(
            kind=self.kind,
            slots=self.cpu_slots,
            shared_fs=True,
            description=f"{self.description}; jobs run {self.python}" if self.kind == "local" else self.description,
        )

    def probe(self) -> List[str]:
        """Ask the installed runner what it supports; an old install gives an empty list."""
        try:
            r = subprocess.run(
                [self.python, "-m", "fluxpype.config_runner", "--capabilities"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(paths.checkout_root()),
            )
            line = [x for x in r.stdout.splitlines() if x.strip().startswith("{")][-1]
            return list(json.loads(line).get("features", []))
        except Exception:
            return []

    def command(self, task: TaskSpec) -> List[str]:
        return [
            self.python,
            "-m",
            "fluxpype.config_runner",
            "--config",
            task.profile,
            "--only",
            f"{task.cr},{task.nflux},{task.adapt},{task.method}",
        ]

    def environment(self, task: TaskSpec) -> dict:
        root = str(paths.checkout_root())
        env = {
            "FLUXPYPE_CONFIG": task.profile,
            "FLUXPYPE_CONFIG_FILE": task.snapshot_path,
            "FLUXPYPE_EVENTS": os.path.join(task.run_dir, "events.jsonl"),
            "FLUXPYPE_LOG_FILE": os.path.join(task.run_dir, "run.log"),
            "FLUXPYPE_TASK_ID": task.task_id,
            "FLUXPYPE_TASK_TIMEOUT": str(task.timeout_s),
            "PYTHONPATH": root + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
            "PYTHONUNBUFFERED": "1",
        }
        return env

    def submit(self, task: TaskSpec) -> Handle:
        spawn(task.run_dir, self.command(task), self.environment(task), str(paths.checkout_root()))
        return Handle(task_id=task.task_id, run_dir=task.run_dir)

    def poll(self, handle: Handle) -> TaskStatus:
        return poll_run_dir(handle.run_dir)

    def cancel(self, handle: Handle, force: bool = False) -> None:
        run = handle.run_dir
        try:
            with open(os.path.join(run, "cancelled"), "a"):
                pass
        except OSError:
            pass
        pid = _read_json(os.path.join(run, "pid"))
        if not pid or not launcher.process_alive(pid["pid"], pid.get("pid_start")):
            return
        try:
            os.killpg(pid["pid"], signal.SIGKILL if force else signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    def read_file(self, handle: Handle, name: str, offset: int = 0) -> bytes:
        if os.sep in name or name.startswith("."):
            raise ValueError("bad file name")
        try:
            with open(os.path.join(handle.run_dir, name), "rb") as f:
                f.seek(offset)
                return f.read(4 * 1024 * 1024)
        except OSError:
            return b""
