"""``flux-studio``: start the local web GUI."""

import argparse
import os
import secrets
import socket
import sys
import threading
import webbrowser
from pathlib import Path


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("fluxpype")
    except Exception:
        return "dev"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flux-studio", description="FLUX Studio: a web GUI for the fluxpype pipeline")
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to bind (default 127.0.0.1; use an SSH tunnel for remote machines)",
    )
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    p.add_argument("--demo", action="store_true", help="simulate runs with a fake pipeline (no FLUX needed)")
    p.add_argument("--config", help="config.ini to read profiles from (default: the one shipped with fluxpype)")
    p.add_argument("--home", help="studio state directory (default ~/.fluxpype/studio, or $FLUX_STUDIO_HOME)")
    p.add_argument("--max-parallel", type=int, help="tasks to run at once (default 1; demo 2)")
    p.add_argument("--cpu-budget", type=int, help="CPU slots to share between tasks (default: all cores)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import uvicorn
        from fastapi import FastAPI  # noqa: F401
    except ImportError:
        print('FLUX Studio needs its extras: pip install -e ".[studio]"', file=sys.stderr)
        return 2

    from . import demo, paths
    from .backends.fake import FakeBackend
    from .backends.local import LocalBackend
    from .scheduler import Scheduler
    from .server import create_app
    from .store import Store

    if args.home:
        os.environ["FLUX_STUDIO_HOME"] = args.home
    if args.demo:
        home = demo.prepare_home()
        config_path = home / "config.ini"
        backends = {"fake": FakeBackend()}
    else:
        home = paths.studio_home()
        config_path = Path(args.config).expanduser() if args.config else paths.shipped_config()
        if not config_path.is_file():
            print(f"config file not found: {config_path}", file=sys.stderr)
            return 2
        backends = {"local": LocalBackend()}

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
    tpath = paths.token_path(home)
    tpath.write_text(token)
    os.chmod(tpath, 0o600)
    hosts = ["localhost", "127.0.0.1", "::1"] + ([args.host] if args.host not in ("0.0.0.0", "::") else [])
    app = create_app(scheduler, token, allowed_hosts=hosts, demo=args.demo, version=_version())

    shown_host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0", "::") else args.host
    url = f"http://{shown_host}:{args.port}/?token={token}"
    with socket.socket() as s:
        if s.connect_ex((args.host if args.host not in ("0.0.0.0", "::") else "127.0.0.1", args.port)) == 0:
            print(f"port {args.port} is already in use; pick another with --port", file=sys.stderr)
            return 2
    print(f"\nFLUX Studio {_version()}{' (demo)' if args.demo else ''}\n  {url}\n  state: {home}\n", flush=True)
    if args.demo:
        threading.Thread(target=demo.seed, args=(scheduler,), daemon=True).start()
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
