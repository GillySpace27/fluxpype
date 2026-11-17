"""Run tasks on another machine: plain SSH (``SSHBackend``) or a Slurm queue (``SlurmBackend``).

Both reuse the stdlib-only launcher (``fluxpype.plumber.launcher``) on the remote
side, so a remote run directory holds the same ``pid`` / ``exit_code`` /
``events.jsonl`` files as a local one. ``poll`` mirrors the growing text files
(events, logs) into the local run directory, which is what the monitor reads, so
the rest of Plumber needs no remote awareness.

All remote access goes through a ``Transport`` (one method: run a shell command
with optional stdin). ``SSHTransport`` uses the system ``ssh`` binary, so
``~/.ssh/config``, keys, ProxyJump and ControlMaster apply and there is no
paramiko dependency. ``LocalTransport`` runs the command here, for a Plumber that
itself runs on a Slurm login node. Tests pass a fake transport.

Not done yet: output data products stay on the remote ``data_dir``; nothing
copies them back, so the Results and Cache views do not see them.
"""

import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

from .base import BackendCaps, Handle, TaskSpec, TaskStatus

MIRRORED = ("events.jsonl", "run.log", "stdout.log")  # copied into the local run dir as they grow
CHUNK = 4 * 1024 * 1024
START_GRACE_S = 60.0


class TransportError(Exception):
    """The remote machine could not be reached (as opposed to a command failing there)."""


class Transport(Protocol):
    def run(self, command: str, stdin: bytes = None, timeout: float = 60) -> Tuple[int, bytes, bytes]:
        """Run a POSIX shell command remotely: (exit code, stdout, stderr). Raises TransportError if unreachable."""


class SSHTransport:
    def __init__(self, host: str, options: List[str] = None, ssh: str = "ssh"):
        self.host = host
        self.options = list(options or [])
        self.ssh = ssh

    def argv(self, command: str) -> List[str]:
        return [
            self.ssh,
            "-o",
            "BatchMode=yes",  # never hang on a password prompt
            "-o",
            "ConnectTimeout=15",
            *self.options,
            self.host,
            command,
        ]

    def run(self, command: str, stdin: bytes = None, timeout: float = 60) -> Tuple[int, bytes, bytes]:
        try:
            r = subprocess.run(self.argv(command), input=stdin or b"", capture_output=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise TransportError(str(e)) from e
        if r.returncode == 255:  # ssh's own failure code (connection, auth)
            raise TransportError(r.stderr.decode("utf-8", "replace").strip() or "ssh failed")
        return r.returncode, r.stdout, r.stderr


class LocalTransport:
    """Run the command on this machine (Plumber on a login node)."""

    def run(self, command: str, stdin: bytes = None, timeout: float = 60) -> Tuple[int, bytes, bytes]:
        try:
            r = subprocess.run(["sh", "-c", command], input=stdin or b"", capture_output=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise TransportError(str(e)) from e
        return r.returncode, r.stdout, r.stderr


@dataclass
class RemoteConfig:
    """One entry of clusters.toml."""

    name: str
    kind: str = "ssh"  # ssh | slurm
    host: str = ""  # ssh destination; empty with kind=slurm means "sbatch on this machine"
    fluxpype_root: str = ""  # remote checkout (the folder that contains the fluxpype package)
    python: str = "python3"
    work_dir: str = "~/.fluxpype/plumber-remote"  # remote run directories live here
    slots: int = 1  # CPU slots Plumber may use there at once
    setup: List[str] = field(default_factory=list)  # shell lines run before the job (module load, conda activate)
    ssh_options: List[str] = field(default_factory=list)
    # Slurm only
    partition: str = ""
    account: str = ""
    cpus_per_task: int = 1
    mem: str = ""
    sbatch_extra: List[str] = field(default_factory=list)  # extra "#SBATCH ..." argument strings
    sbatch: str = "sbatch"
    squeue: str = "squeue"
    scancel: str = "scancel"

    @classmethod
    def from_dict(cls, name: str, d: Dict) -> "RemoteConfig":
        known = {f for f in cls.__dataclass_fields__ if f != "name"}
        bad = sorted(set(d) - known)
        if bad:
            raise ValueError(f"cluster {name!r}: unknown keys {bad}")
        cfg = cls(name=name, **d)
        if cfg.kind not in ("ssh", "slurm"):
            raise ValueError(f"cluster {name!r}: kind must be 'ssh' or 'slurm', not {cfg.kind!r}")
        if cfg.kind == "ssh" and not cfg.host:
            raise ValueError(f"cluster {name!r}: kind 'ssh' needs a host")
        if not cfg.fluxpype_root:
            raise ValueError(f"cluster {name!r}: fluxpype_root is required")
        return cfg


def _q(s: str) -> str:
    """Shell-quote, but let a leading ~/ expand on the remote side."""
    if s.startswith("~/"):
        return '"$HOME"/' + shlex.quote(s[2:])
    return shlex.quote(s)


def _sections(out: bytes) -> Dict[str, str]:
    """Split ``@@name`` delimited output into {name: text}."""
    res: Dict[str, str] = {}
    cur = None
    for line in out.decode("utf-8", "replace").splitlines():
        if line.startswith("@@"):
            cur = line[2:].strip()
            res[cur] = ""
        elif cur is not None:
            res[cur] += line + "\n"
    return {k: v.strip() for k, v in res.items()}


def _json(text: str) -> Optional[dict]:
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else None
    except ValueError:
        return None


class SSHBackend:
    kind = "ssh"

    def __init__(self, config: RemoteConfig, transport: Transport = None):
        self.config = config
        self.name = config.name
        self.transport = transport or (
            SSHTransport(config.host, config.ssh_options) if config.host else LocalTransport()
        )

    # -- interface
    def capabilities(self) -> BackendCaps:
        c = self.config
        where = c.host or "this machine"
        what = "Slurm queue" if self.kind == "slurm" else "SSH"
        return BackendCaps(kind=self.kind, slots=c.slots, shared_fs=False, description=f"{what} on {where}")

    def probe(self) -> List[str]:
        cmd = f"{self._prelude()}{_q(self.config.python)} -m fluxpype.config_runner --capabilities"
        try:
            _, out, _ = self.transport.run(cmd, timeout=90)
            line = [x for x in out.decode("utf-8", "replace").splitlines() if x.strip().startswith("{")][-1]
            return list(json.loads(line).get("features", []))
        except Exception:
            return []

    def test(self) -> Dict[str, object]:
        """Connection test for the UI: reachable, python found, runner features."""
        try:
            rc, out, err = self.transport.run(f"{self._prelude()}{_q(self.config.python)} --version", timeout=30)
        except TransportError as e:
            return {"ok": False, "reachable": False, "error": str(e)}
        if rc != 0:
            return {"ok": False, "reachable": True, "error": err.decode("utf-8", "replace").strip()}
        feats = self.probe()
        return {
            "ok": bool(feats),
            "reachable": True,
            "python": (out or err).decode("utf-8", "replace").strip(),
            "features": feats,
            "error": "" if feats else "fluxpype.config_runner did not answer --capabilities at fluxpype_root",
        }

    def submit(self, task: TaskSpec) -> Handle:
        rdir = self.remote_dir(task)
        files = {"config.ini": task.snapshot_path}
        if task.extra.get("kind") == "scene":
            files["scene.json"] = os.path.join(os.path.dirname(task.snapshot_path), f"scene-{task.task_id}.json")
        self._must(f"mkdir -p {_q(rdir)}", "create the remote run folder")
        for rname, lpath in files.items():
            with open(lpath, "rb") as f:
                self._must(f"cat > {_q(rdir)}/{rname}", f"upload {rname}", stdin=f.read())
        launch = {"argv": self.command(task), "env": self.environment(task), "cwd": self.config.fluxpype_root}
        self._must(f"cat > {_q(rdir)}/launch.json", "upload launch.json", stdin=json.dumps(launch).encode())
        os.makedirs(task.run_dir, exist_ok=True)
        with open(os.path.join(task.run_dir, "remote.json"), "w", encoding="utf-8") as f:
            json.dump({"backend": self.name, "remote_dir": rdir}, f)
        self._start(task, rdir)
        return Handle(task_id=task.task_id, run_dir=task.run_dir)

    def poll(self, handle: Handle) -> TaskStatus:
        rdir = self._remote_dir_of(handle)
        if rdir is None:
            return TaskStatus(state="lost")
        try:
            rc, out, _ = self.transport.run(self._poll_script(rdir), timeout=45)
        except TransportError:
            return TaskStatus(state="unknown")
        s = _sections(out)
        if "end" not in s:  # truncated or garbled answer: do not guess
            return TaskStatus(state="unknown")
        status = self._status(s)
        try:
            self._mirror(handle, rdir, s)
        except TransportError:
            if status.state == "exited":  # finalize only once the logs are home
                return TaskStatus(state="unknown")
        return status

    def cancel(self, handle: Handle, force: bool = False) -> None:
        rdir = self._remote_dir_of(handle)
        if rdir is None:
            return
        sig = "KILL" if force else "TERM"
        cmd = (
            f"cd {_q(rdir)} && touch cancelled && "
            "p=$(sed -n 's/.*\"pid\": *\\([0-9][0-9]*\\).*/\\1/p' pid 2>/dev/null); "
            f'[ -n "$p" ] && kill -{sig} -- "-$p" 2>/dev/null; true'
        )
        try:
            self.transport.run(cmd, timeout=45)
        except TransportError:
            pass

    def read_file(self, handle: Handle, name: str, offset: int = 0) -> bytes:
        if "/" in name or name.startswith("."):
            raise ValueError("bad file name")
        rdir = self._remote_dir_of(handle)
        if rdir is None:
            return b""
        return self._tail(rdir, name, offset)

    # -- pieces
    def remote_dir(self, task: TaskSpec) -> str:
        return f"{self.config.work_dir.rstrip('/')}/{task.job_id}/{task.task_id}"

    def command(self, task: TaskSpec) -> List[str]:
        py = self.config.python
        if task.extra.get("kind") == "scene":
            return [py, "-m", "fluxpype.plumber.scene_task", "scene.json"]
        only = f"{task.cr},{task.nflux},{task.adapt},{task.method}"
        return [py, "-m", "fluxpype.config_runner", "--config", task.profile, "--only", only]

    def environment(self, task: TaskSpec) -> Dict[str, str]:
        # relative paths: the launcher is started with the run folder as its working directory
        return {
            "FLUXPYPE_CONFIG": task.profile,
            "FLUXPYPE_CONFIG_FILE": "config.ini",
            "FLUXPYPE_EVENTS": "events.jsonl",
            "FLUXPYPE_LOG_FILE": "run.log",
            "FLUXPYPE_TASK_ID": task.task_id,
            "FLUXPYPE_TASK_TIMEOUT": str(task.timeout_s),
            "PYTHONUNBUFFERED": "1",
        }

    def _prelude(self) -> str:
        lines = [*self.config.setup, f"export PYTHONPATH={_q(self.config.fluxpype_root)}${{PYTHONPATH:+:$PYTHONPATH}}"]
        return "".join(f"{line}\n" for line in lines)

    def _job_script(self, rdir: str) -> str:
        """Shell text that runs the launcher in the foreground from inside the run folder.

        The launcher resolves launch.json's relative paths against the run folder; the
        runner itself gets absolute paths through the exports below.
        """
        return (
            f"{self._prelude()}"
            f"cd {_q(rdir)} || exit 1\n"
            'export FLUXPYPE_CONFIG_FILE="$PWD/config.ini" FLUXPYPE_EVENTS="$PWD/events.jsonl" '
            'FLUXPYPE_LOG_FILE="$PWD/run.log"\n'
            f"exec {_q(self.config.python)} -m fluxpype.plumber.remote_launch .\n"
        )

    def _start(self, task: TaskSpec, rdir: str) -> None:
        self._must(f"cat > {_q(rdir)}/job.sh", "upload job.sh", stdin=self._job_script(rdir).encode())
        self._must(
            f"cd {_q(rdir)} && nohup sh job.sh > launcher.log 2>&1 < /dev/null &",
            "start the task",
        )

    def _must(self, cmd: str, what: str, stdin: bytes = None, timeout: float = 120) -> bytes:
        rc, out, err = self.transport.run(cmd, stdin=stdin, timeout=timeout)
        if rc != 0:
            raise RuntimeError(f"{self.name}: could not {what}: {err.decode('utf-8', 'replace').strip() or rc}")
        return out

    def _remote_dir_of(self, handle: Handle) -> Optional[str]:
        try:
            with open(os.path.join(handle.run_dir, "remote.json"), "r", encoding="utf-8") as f:
                return json.load(f)["remote_dir"]
        except (OSError, ValueError, KeyError):
            return None

    def _poll_extra(self) -> str:
        return ""

    def _poll_script(self, rdir: str) -> str:
        sizes = "".join(f"echo @@size:{n}; wc -c < {n} 2>/dev/null\n" for n in MIRRORED)
        return (
            f"cd {_q(rdir)} 2>/dev/null || {{ echo @@norun; echo @@end; exit 0; }}\n"
            "echo @@pid; cat pid 2>/dev/null; echo\n"
            "echo @@exit_code; cat exit_code 2>/dev/null; echo\n"
            "[ -e cancelled ] && echo @@cancelled\n"
            "p=$(sed -n 's/.*\"pid\": *\\([0-9][0-9]*\\).*/\\1/p' pid 2>/dev/null)\n"
            'echo @@alive; [ -n "$p" ] && kill -0 "$p" 2>/dev/null && echo yes\n'
            "echo @@age; echo $(( $(date +%s) - $(date -r launch.json +%s 2>/dev/null || echo 0) ))\n"
            f"{sizes}{self._poll_extra()}echo @@end\n"
        )

    def _status(self, s: Dict[str, str]) -> TaskStatus:
        cancelled = "cancelled" in s
        if "norun" in s:
            return TaskStatus(state="lost")
        pid = _json(s.get("pid", "")) or {}
        done = _json(s.get("exit_code", ""))
        base = dict(pid=pid.get("pid"), pid_start=pid.get("pid_start"), started=pid.get("started"), cancelled=cancelled)
        if done is not None:
            return TaskStatus(
                state="exited",
                exit_code=done.get("exit_code"),
                signal=done.get("signal"),
                ended=done.get("ended"),
                **base,
            )
        if pid:
            return TaskStatus(state="running" if s.get("alive") == "yes" else "lost", **base)
        return self._status_not_started(s, cancelled)

    def _status_not_started(self, s: Dict[str, str], cancelled: bool) -> TaskStatus:
        try:
            age = float(s.get("age", "0") or 0)
        except ValueError:
            age = 0.0
        return TaskStatus(state="lost" if age > START_GRACE_S else "starting", cancelled=cancelled)

    def _tail(self, rdir: str, name: str, offset: int) -> bytes:
        cmd = f"tail -c +{int(offset) + 1} {_q(rdir)}/{shlex.quote(name)} 2>/dev/null | head -c {CHUNK}"
        _, out, _ = self.transport.run(cmd, timeout=120)
        return out

    def _mirror(self, handle: Handle, rdir: str, s: Dict[str, str]) -> None:
        """Append what the remote text files gained since the last poll to their local copies."""
        for name in MIRRORED:
            try:
                remote_size = int(s.get(f"size:{name}", "") or 0)
            except ValueError:
                continue
            local = os.path.join(handle.run_dir, name)
            have = os.path.getsize(local) if os.path.exists(local) else 0
            while have < remote_size:
                data = self._tail(rdir, name, have)
                if not data:
                    break
                with open(local, "ab") as f:
                    f.write(data)
                have += len(data)


class SlurmBackend(SSHBackend):
    """Submit each task with ``sbatch``; ``squeue`` tells queued from running, ``scancel`` cancels."""

    kind = "slurm"

    def sbatch_script(self, task: TaskSpec, rdir: str) -> str:
        c = self.config
        minutes = max(1, -(-int(task.timeout_s) // 60) + 5)  # the runner's own timeout, plus slack to write exit_code
        head = [
            "#!/bin/sh",
            f"#SBATCH --job-name=flux-{task.task_id}",
            "#SBATCH --output=slurm.out",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={int(c.cpus_per_task)}",
            f"#SBATCH --time={minutes}",
        ]
        if c.partition:
            head.append(f"#SBATCH --partition={c.partition}")
        if c.account:
            head.append(f"#SBATCH --account={c.account}")
        if c.mem:
            head.append(f"#SBATCH --mem={c.mem}")
        head += [f"#SBATCH {x}" for x in c.sbatch_extra]
        return "\n".join(head) + "\n" + self._job_script(rdir)

    def _start(self, task: TaskSpec, rdir: str) -> None:
        self._must(f"cat > {_q(rdir)}/job.sh", "upload job.sh", stdin=self.sbatch_script(task, rdir).encode())
        out = self._must(
            f"{self._setup()}cd {_q(rdir)} && {_q(self.config.sbatch)} --parsable job.sh", "submit to Slurm (sbatch)"
        )
        job_id = out.decode("utf-8", "replace").strip().splitlines()[-1].split(";")[0].strip() if out.strip() else ""
        if not job_id.isdigit():
            raise RuntimeError(f"{self.name}: sbatch gave no job id: {out!r}")
        self._must(f"echo {job_id} > {_q(rdir)}/slurm_job_id", "record the Slurm job id")

    def _setup(self) -> str:
        return "".join(f"{line}\n" for line in self.config.setup)

    def _poll_extra(self) -> str:
        return (
            "echo @@slurm_job_id; cat slurm_job_id 2>/dev/null; echo\n"
            f'echo @@squeue; j=$(cat slurm_job_id 2>/dev/null); [ -n "$j" ] && '
            f'{{ {_q(self.config.squeue)} -h -j "$j" -o %T 2>/dev/null; echo "rc=$?"; }}\n'
        )

    def _status(self, s: Dict[str, str]) -> TaskStatus:
        st = super()._status(s)
        if st.state == "running" or st.state == "exited" or "norun" in s:
            return st
        # no live process seen from the login node (the job runs on a compute node): ask the queue
        lines = s.get("squeue", "").split()
        state = next((x for x in lines if not x.startswith("rc=")), "")
        pid = _json(s.get("pid", "")) or {}
        base = dict(pid=pid.get("pid"), pid_start=pid.get("pid_start"), started=pid.get("started"), cancelled=st.cancelled)
        if state in ("PENDING", "CONFIGURING", "REQUEUED", "RESIZING", "SUSPENDED"):
            return TaskStatus(state="queued", **base)
        if state in ("RUNNING", "COMPLETING", "STAGE_OUT"):
            return TaskStatus(state="running" if pid else "starting", **base)
        if not s.get("slurm_job_id") or "rc=0" not in lines:
            # squeue itself failed (slurmctld busy) or the id is not recorded yet: cannot tell
            return TaskStatus(state="unknown" if s.get("slurm_job_id") else st.state, **base)
        return TaskStatus(state="lost", **base)  # left the queue without writing exit_code

    def _status_not_started(self, s: Dict[str, str], cancelled: bool) -> TaskStatus:
        return TaskStatus(state="starting", cancelled=cancelled)  # the queue decides, not a grace period

    def cancel(self, handle: Handle, force: bool = False) -> None:
        rdir = self._remote_dir_of(handle)
        if rdir is None:
            return
        flag = "--signal=KILL --full " if force else ""
        cmd = (
            f"{self._setup()}cd {_q(rdir)} && touch cancelled && "
            f'j=$(cat slurm_job_id 2>/dev/null); [ -n "$j" ] && {_q(self.config.scancel)} {flag}"$j"; true'
        )
        try:
            self.transport.run(cmd, timeout=45)
        except TransportError:
            pass


def load_clusters(path) -> Dict[str, SSHBackend]:
    """Backends from a clusters.toml (one table per cluster). A missing file means no clusters."""
    if not os.path.isfile(path):
        return {}
    try:
        import tomllib
    except ImportError:  # Python 3.10
        try:
            import tomli as tomllib
        except ImportError as e:
            raise ValueError("reading clusters.toml on Python 3.10 needs: pip install tomli") from e
    with open(path, "rb") as f:
        data = tomllib.load(f)
    out: Dict[str, SSHBackend] = {}
    for name, d in data.items():
        if not isinstance(d, dict):
            raise ValueError(f"clusters.toml: [{name}] must be a table")
        if name in ("local", "fake"):
            raise ValueError(f"clusters.toml: {name!r} is a reserved backend name")
        cfg = RemoteConfig.from_dict(name, d)
        out[name] = SlurmBackend(cfg) if cfg.kind == "slurm" else SSHBackend(cfg)
    return out
