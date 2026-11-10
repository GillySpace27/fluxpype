"""
Detached task launcher (stdlib only).

``python -m fluxpype.studio.launcher RUN_DIR`` reads ``RUN_DIR/launch.json``
(``argv``, ``env``, ``cwd``), starts the runner in its own session with stdout
and stderr going to ``RUN_DIR/stdout.log``, records ``pid`` (with the process
start time, so a reused pid is not mistaken for the task), waits, and then
writes ``exit_code`` atomically (temp file + rename).

Because the launcher is detached from the studio server, restarting the server
never kills a running job; the server re-attaches by reading these files.
"""

import json
import os
import signal
import subprocess
import sys
import time


def process_start_token(pid):
    """A string that identifies when `pid` started (None if the process is gone)."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            data = f.read().decode("utf-8", "replace")
        rest = data[data.rindex(")") + 2 :].split()
        return rest[19]  # field 22: start time in clock ticks since boot
    except (OSError, ValueError, IndexError):
        pass
    try:  # macOS and other systems without /proc
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True, timeout=5)
        text = out.stdout.strip()
        return text or None
    except Exception:
        return None


def process_alive(pid, token=None):
    """True if `pid` exists, is not a zombie, and (when given) started at `token`."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            data = f.read().decode("utf-8", "replace")
        if data[data.rindex(")") + 2 :].split()[0] == "Z":
            return False
    except (OSError, ValueError, IndexError):
        pass
    if token is not None:
        return process_start_token(pid) == token
    return True


def write_json_atomic(path, payload):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    run_dir = argv[0]
    with open(os.path.join(run_dir, "launch.json"), "r", encoding="utf-8") as f:
        spec = json.load(f)
    env = dict(os.environ)
    env.update(spec.get("env", {}))
    out = open(os.path.join(run_dir, "stdout.log"), "ab", buffering=0)
    proc = subprocess.Popen(
        spec["argv"],
        env=env,
        cwd=spec.get("cwd") or None,
        stdin=subprocess.DEVNULL,
        stdout=out,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    started = time.time()
    write_json_atomic(
        os.path.join(run_dir, "pid"),
        {"pid": proc.pid, "pid_start": process_start_token(proc.pid), "started": started, "launcher_pid": os.getpid()},
    )

    def forward(signum, frame):
        try:
            os.killpg(proc.pid, signum)
        except (ProcessLookupError, PermissionError):
            pass

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    rc = proc.wait()
    sig = -rc if rc < 0 else None
    write_json_atomic(
        os.path.join(run_dir, "exit_code"),
        {"exit_code": 128 + sig if sig else rc, "signal": sig, "ended": time.time()},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
