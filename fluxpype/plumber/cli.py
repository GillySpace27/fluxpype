"""``flux-plumb``: start the local web GUI."""

import argparse
import json
import os
import secrets
import signal
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

_LOCK_FD = None  # held for the life of the process; closing it would release the single-instance lock


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("fluxpype")
    except Exception:
        return "dev"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flux-plumb", description="Fluxpype Plumber: a web GUI for the fluxpype pipeline")
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to bind (default 127.0.0.1; use an SSH tunnel for remote machines)",
    )
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    p.add_argument("--demo", action="store_true", help="simulate runs with a fake pipeline (no FLUX needed)")
    p.add_argument("--config", help="config.ini to read profiles from (default: the one shipped with fluxpype)")
    p.add_argument("--home", help="plumber state directory (default ~/.fluxpype/plumber, or $FLUX_PLUMBER_HOME)")
    p.add_argument("--max-parallel", type=int, help="tasks to run at once (default 1; demo 2)")
    p.add_argument("--cpu-budget", type=int, help="CPU slots to share between tasks (default: all cores)")
    p.add_argument(
        "--headless",
        action="store_true",
        help="no browser and nothing on stdout; log to <home>/server.log (used by the MCP server's autostart)",
    )
    p.add_argument(
        "--stop",
        action="store_true",
        help="stop the Plumber running for --home (SIGTERM, waits up to 10 s). "
        "Jobs are detached, so running jobs keep running and are picked up by the next Plumber",
    )
    return p


def server_json_path(home: Path) -> Path:
    return Path(home) / "server.json"


def lock_path(home: Path) -> Path:
    return Path(home) / "plumber.lock"


def write_atomic(path: Path, text: str, mode: int = 0o600) -> None:
    """Write `text` to `path` through a temp file in the same directory, with permissions `mode`."""
    tmp = Path(f"{path}.tmp{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def read_server_json(home: Path):
    """The parsed server.json for `home`, or None when it is missing or unreadable."""
    try:
        data = json.loads(server_json_path(home).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def acquire_lock(home: Path) -> bool:
    """Take the single-instance lock for `home` (kept until the process exits). False if another holds it."""
    global _LOCK_FD
    try:
        import fcntl
    except ImportError:  # no flock on this platform: run without the guard
        return True
    fd = os.open(lock_path(home), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return False
    _LOCK_FD = fd
    return True


def lock_is_held(home: Path) -> bool:
    """True if some process holds the lock for `home` (probe with a non-blocking flock, released at once)."""
    try:
        import fcntl
    except ImportError:
        return False
    path = lock_path(home)
    if not path.exists():
        return False
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    finally:
        os.close(fd)  # closing also drops a lock we just took
    return False


def stop_running(home: Path, wait_s: float = 10.0) -> int:
    """Stop the Plumber recorded in server.json. Returns an exit code."""
    from .launcher import process_alive

    info = read_server_json(home)
    pid = info.get("pid") if info else None
    if not isinstance(pid, int) or not process_alive(pid, info.get("pid_start")):
        print(f"no Plumber is running for {home}", file=sys.stderr)
        return 1
    os.kill(pid, signal.SIGTERM)
    end = time.time() + wait_s
    while time.time() < end:
        if not process_alive(pid, info.get("pid_start")):
            print(f"stopped Plumber (pid {pid}); running jobs were left running")
            return 0
        time.sleep(0.1)
    print(f"Plumber (pid {pid}) did not exit within {wait_s:.0f} s", file=sys.stderr)
    return 1


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.stop:
        if args.home:
            os.environ["FLUX_PLUMBER_HOME"] = args.home
        from . import paths

        return stop_running(paths.plumber_home())
    try:
        import uvicorn
        from fastapi import FastAPI  # noqa: F401
    except ImportError:
        print('Fluxpype Plumber needs its extras: pip install -e ".[plumber]"', file=sys.stderr)
        return 2

    from . import demo, paths
    from .backends.fake import FakeBackend
    from .backends.local import LocalBackend
    from .scheduler import Scheduler
    from .server import create_app
    from .store import Store

    if args.home:
        os.environ["FLUX_PLUMBER_HOME"] = args.home
    config_path = None
    if args.demo and args.home:
        home = Path(args.home).expanduser()
        home.mkdir(parents=True, exist_ok=True)
    elif args.demo:
        home = demo.prepare_home()  # a fresh temp directory nobody else can hold
    else:
        home = paths.plumber_home()
        config_path = Path(args.config).expanduser() if args.config else paths.shipped_config()
        if not config_path.is_file():
            print(f"config file not found: {config_path}", file=sys.stderr)
            return 2

    if not acquire_lock(home):
        print(f"another Plumber is already running for {home}", file=sys.stderr)
        info = server_json_path(home)
        if info.is_file():
            print(info.read_text(encoding="utf-8"), file=sys.stderr)
        return 3
    if args.demo:
        if args.home:
            demo.prepare_home(home)  # only after the lock: this rewrites config.ini
        config_path = home / "config.ini"
        backends = {"fake": FakeBackend()}
    else:
        backends = {"local": LocalBackend()}

    bind_host = args.host if args.host not in ("0.0.0.0", "::") else "127.0.0.1"
    with socket.socket() as s:
        if s.connect_ex((bind_host, args.port)) == 0:
            print(f"port {args.port} is already in use; pick another with --port", file=sys.stderr)
            return 2

    if args.headless:
        args.no_browser = True
        log = open(home / "server.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = log
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)

    store = Store(paths.db_path(home))
    scheduler = Scheduler(
        store,
        backends,
        config_path,
        home=home,
        cpu_budget=args.cpu_budget or (64 if args.demo else None),
        stale_after=12.0 if args.demo else 120.0,
    )
    if args.max_parallel or args.demo:
        scheduler.set_queue(max_parallel=args.max_parallel or 2)

    token = secrets.token_urlsafe(24)
    hosts = ["localhost", "127.0.0.1", "::1"] + ([args.host] if args.host not in ("0.0.0.0", "::") else [])
    app = create_app(scheduler, token, allowed_hosts=hosts, demo=args.demo, version=_version())

    write_atomic(paths.token_path(home), token)
    started = time.time()
    from .launcher import process_start_token

    record = {
        "host": bind_host,
        "port": args.port,
        "pid": os.getpid(),
        "pid_start": process_start_token(os.getpid()),
        "version": _version(),
        "home": str(home),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "demo": bool(args.demo),
    }
    write_atomic(server_json_path(home), json.dumps(record))

    shown_host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0", "::") else args.host
    url = f"http://{shown_host}:{args.port}/?token={token}"
    if not args.headless:
        print(
            f"\nFluxpype Plumber {_version()}{' (demo)' if args.demo else ''}\n  {url}\n  state: {home}\n", flush=True
        )
    # uvicorn shuts down cleanly on SIGTERM and then re-raises it through the handler it replaced; make that
    # handler a normal exit so the cleanup below runs and `flux-plumb --stop` ends with status 0
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
    try:
        if args.demo:
            threading.Thread(target=demo.seed, args=(scheduler,), daemon=True).start()
        if not args.no_browser:
            threading.Timer(1.0, webbrowser.open, args=(url,)).start()
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        try:  # best effort: only remove the record if it is still ours
            current = read_server_json(home)
            if current and current.get("pid") == os.getpid():
                server_json_path(home).unlink()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
