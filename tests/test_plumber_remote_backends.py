"""SSH and Slurm backends, driven through a transport that runs the 'remote' shell commands here."""

import json
import os
import stat
import sys
import time

import pytest

from fluxpype.plumber import paths
from fluxpype.plumber.backends.base import Handle, TaskSpec
from fluxpype.plumber.backends.remote import (
    LocalTransport,
    RemoteConfig,
    SlurmBackend,
    SSHBackend,
    SSHTransport,
    TransportError,
    load_clusters,
)


class Recording(LocalTransport):
    """Runs commands locally, keeps a log, and can pretend the host is down."""

    def __init__(self):
        self.commands = []
        self.down = False

    def run(self, command, stdin=None, timeout=60):
        if self.down:
            raise TransportError("host unreachable")
        self.commands.append(command)
        return super().run(command, stdin=stdin, timeout=timeout)


def spec(tmp_path, **kw):
    snap = tmp_path / "snap.ini"
    snap.write_text("[DEFAULT]\n")
    return TaskSpec(
        task_id="j1-01",
        job_id="j1",
        run_dir=str(tmp_path / "local" / "r"),
        snapshot_path=str(snap),
        profile="quickstart",
        cr="2225",
        nflux="100",
        adapt="0",
        method="parker",
        timeout_s=77,
        **kw,
    )


def config(tmp_path, **kw):
    return RemoteConfig(
        name="box",
        host="box.example",
        fluxpype_root=str(paths.checkout_root()),
        python=sys.executable,
        work_dir=str(tmp_path / "remote"),
        **kw,
    )


PAYLOAD = (
    "import os,sys;print('hello');"
    "open(os.environ['FLUXPYPE_EVENTS'],'a').write('{\"event\": \"run.end\"}\\n');"
    "assert os.path.isfile(os.environ['FLUXPYPE_CONFIG_FILE']);sys.exit(%d)"
)


def scripted(cls, code=0, sleep=0.0):
    class B(cls):
        def command(self, task):
            body = (f"import time;time.sleep({sleep});" if sleep else "") + PAYLOAD % code
            return [self.config.python, "-c", body]

    return B


def wait(backend, handle, states=("exited", "lost"), timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        st = backend.poll(handle)
        if st.state in states:
            return st
        time.sleep(0.2)
    raise AssertionError(f"still {st.state}")


def test_ssh_argv_never_prompts():
    argv = SSHTransport("me@box", ["-J", "gate"]).argv("echo hi")
    assert argv[0] == "ssh" and "BatchMode=yes" in argv and argv[-2:] == ["me@box", "echo hi"]
    assert argv.index("-J") < argv.index("me@box")


def test_ssh_command_and_environment(tmp_path):
    b = SSHBackend(config(tmp_path), transport=Recording())
    t = spec(tmp_path)
    assert b.command(t)[1:] == ["-m", "fluxpype.config_runner", "--config", "quickstart", "--only", "2225,100,0,parker"]
    env = b.environment(t)
    assert env["FLUXPYPE_TASK_TIMEOUT"] == "77" and env["FLUXPYPE_CONFIG_FILE"] == "config.ini"
    caps = b.capabilities()
    assert caps.kind == "ssh" and caps.shared_fs is False and "box.example" in caps.description


def test_ssh_run_end_to_end_mirrors_logs(tmp_path):
    tr = Recording()
    b = scripted(SSHBackend, code=3)(config(tmp_path), transport=tr)
    t = spec(tmp_path)
    h = b.submit(t)
    st = wait(b, h)
    assert st.state == "exited" and st.exit_code == 3 and not st.cancelled
    rdir = tmp_path / "remote" / "j1" / "j1-01"
    assert (rdir / "config.ini").read_text() == "[DEFAULT]\n"
    # the monitor reads the local run dir: events and stdout were copied home
    assert "run.end" in open(os.path.join(t.run_dir, "events.jsonl")).read()
    assert open(os.path.join(t.run_dir, "stdout.log")).read().strip() == "hello"
    assert b.read_file(h, "stdout.log", offset=2) == b"llo\n"
    with pytest.raises(ValueError):
        b.read_file(h, "../x")
    n = os.path.getsize(os.path.join(t.run_dir, "events.jsonl"))
    b.poll(h)  # a second poll must not duplicate lines
    assert os.path.getsize(os.path.join(t.run_dir, "events.jsonl")) == n


def test_ssh_unreachable_is_unknown_not_lost(tmp_path):
    tr = Recording()
    b = scripted(SSHBackend, sleep=30)(config(tmp_path), transport=tr)
    h = b.submit(spec(tmp_path))
    assert wait(b, h, states=("running",)).state == "running"
    tr.down = True
    assert b.poll(h).state == "unknown"
    b.cancel(h)  # must not raise while the host is down
    tr.down = False
    b.cancel(h)
    st = wait(b, h)
    assert st.cancelled and st.state == "exited" and st.signal == 15


def test_poll_without_a_remote_record_is_lost(tmp_path):
    b = SSHBackend(config(tmp_path), transport=Recording())
    assert b.poll(Handle("x", str(tmp_path / "nowhere"))).state == "lost"


def test_submit_failure_names_the_step(tmp_path):
    class Refuses(Recording):
        def run(self, command, stdin=None, timeout=60):
            return 1, b"", b"Permission denied"

    b = SSHBackend(config(tmp_path), transport=Refuses())
    with pytest.raises(RuntimeError, match="create the remote run folder: Permission denied"):
        b.submit(spec(tmp_path))


def fake_slurm(tmp_path):
    """sbatch that prints an id and runs the script in the background; squeue that prints a state file."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    state = tmp_path / "queue_state"
    held = tmp_path / "held"
    sbatch = bindir / "sbatch"
    sbatch.write_text(
        f'#!/bin/sh\n[ "$1" = --parsable ] || exit 2\necho PENDING > {state}\n'
        f'if [ -e {held} ]; then echo 4242; exit 0; fi\n'
        f'( echo RUNNING > {state}; sh "$2" > slurm.out 2>&1; : > {state} ) > /dev/null 2>&1 &\necho "4242;cluster"\n'
    )
    squeue = bindir / "squeue"
    squeue.write_text(f'#!/bin/sh\n[ -e {tmp_path}/squeue_down ] && exit 1\ncat {state}\n')
    scancel = bindir / "scancel"
    scancel.write_text(f'#!/bin/sh\necho "$@" > {tmp_path}/scancelled\n: > {state}\n')
    for f in (sbatch, squeue, scancel):
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return dict(sbatch=str(sbatch), squeue=str(squeue), scancel=str(scancel)), held


def test_slurm_script_has_the_resource_lines(tmp_path):
    cfg = config(
        tmp_path, kind="slurm", partition="short", account="p960", cpus_per_task=4, mem="8G",
        sbatch_extra=["--constraint=avx2"], setup=["module load python"],
    )
    text = SlurmBackend(cfg, transport=Recording()).sbatch_script(spec(tmp_path), "/scratch/r")
    for line in (
        "#SBATCH --partition=short", "#SBATCH --account=p960", "#SBATCH --cpus-per-task=4", "#SBATCH --mem=8G",
        "#SBATCH --constraint=avx2", "#SBATCH --time=7", "module load python", "cd /scratch/r",
    ):
        assert line in text
    assert text.index("#SBATCH --time") < text.index("module load python")  # directives before any command


def test_slurm_run_end_to_end(tmp_path):
    tools, _ = fake_slurm(tmp_path)
    b = scripted(SlurmBackend)(config(tmp_path, kind="slurm", **tools), transport=Recording())
    t = spec(tmp_path)
    h = b.submit(t)
    assert (tmp_path / "remote" / "j1" / "j1-01" / "slurm_job_id").read_text().strip() == "4242"
    st = wait(b, h)
    assert st.state == "exited" and st.exit_code == 0
    assert "run.end" in open(os.path.join(t.run_dir, "events.jsonl")).read()


def test_slurm_queued_then_cancelled(tmp_path):
    tools, held = fake_slurm(tmp_path)
    held.write_text("")
    b = scripted(SlurmBackend)(config(tmp_path, kind="slurm", **tools), transport=Recording())
    h = b.submit(spec(tmp_path))
    assert b.poll(h).state == "queued"
    (tmp_path / "squeue_down").write_text("")
    assert b.poll(h).state == "unknown"  # squeue failing is not the job vanishing
    (tmp_path / "squeue_down").unlink()
    b.cancel(h)
    assert (tmp_path / "scancelled").read_text().strip() == "4242"
    st = b.poll(h)
    assert st.state == "lost" and st.cancelled  # gone from the queue with no exit code


def test_load_clusters(tmp_path):
    assert load_clusters(tmp_path / "none.toml") == {}
    p = tmp_path / "clusters.toml"
    p.write_text(
        '[hpc]\nkind = "slurm"\nhost = "login.example"\nfluxpype_root = "~/fluxpype"\npartition = "short"\n'
        '[box]\nhost = "box"\nfluxpype_root = "/opt/fluxpype"\nslots = 8\n'
    )
    got = load_clusters(p)
    assert isinstance(got["hpc"], SlurmBackend) and got["hpc"].config.partition == "short"
    assert type(got["box"]) is SSHBackend and got["box"].capabilities().slots == 8
    p.write_text('[x]\nhost = "h"\nfluxpype_root = "/r"\npartion = "typo"\n')
    with pytest.raises(ValueError, match="unknown keys"):
        load_clusters(p)
    p.write_text('[local]\nhost = "h"\nfluxpype_root = "/r"\n')
    with pytest.raises(ValueError, match="reserved"):
        load_clusters(p)
