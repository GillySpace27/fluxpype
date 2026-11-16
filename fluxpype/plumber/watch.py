"""``python -m fluxpype.plumber.watch``: print one line per Plumber job event, for an agent's Monitor tool.

Stdlib only. It finds the Plumber for ``--home`` from ``server.json`` and ``token`` (as the MCP server does), prints
the current state of each watched job first, then long-polls ``GET /api/events`` and prints each new event as one
``flux ...`` line (notify.format_line: enums, ids and integers only, never log text). Under Claude Code's Monitor tool
each line wakes the session; with ``--quiet`` under a background Bash call only the final line is printed.

Exit codes: 0 every watched job ended ok; 1 one ended otherwise; 2 usage error; 3 unknown job; 4 Plumber unreachable
for 300 s. The token is read from the home and never printed or put on a command line.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import notify, paths
from .launcher import process_alive

RETRY_S = 5.0
GIVE_UP_S = 300.0
WAIT_S = 25
JOBLESS = {"backend_down", "backend_up", "plumber_started"}
OUTCOME_STATUS = {v: k for k, v in notify.JOB_OUTCOME.items()}  # job_ok -> ok


class Unreachable(Exception):
    pass


class UnknownJob(Exception):
    pass


class Conn:
    """The Plumber for one home; re-discovered after any failure (it may restart on another port or token)."""

    def __init__(self, home: Path):
        self.home = home
        self.found: Optional[Tuple[str, str]] = None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # localhost only, never a proxy

    def _discover(self) -> Tuple[str, str]:
        try:
            info = json.loads((self.home / "server.json").read_text(encoding="utf-8"))
            token = (self.home / "token").read_text(encoding="utf-8").strip()
        except (OSError, ValueError):
            raise Unreachable("no server.json or token") from None
        if not isinstance(info, dict) or not isinstance(info.get("pid"), int) or not isinstance(info.get("port"), int):
            raise Unreachable("server.json is incomplete")
        if not process_alive(info["pid"], info.get("pid_start")):
            raise Unreachable("the Plumber in server.json is not running")
        host = info.get("host") or "127.0.0.1"
        url = f"http://{'[' + host + ']' if ':' in host else host}:{info['port']}"
        reported = self._get(url, token, "/api/status", 5).get("plumber_home")
        if not reported or os.path.realpath(reported) != os.path.realpath(str(self.home)):
            raise Unreachable("server.json points at a Plumber for another home")
        return url, token

    def _get(self, url: str, token: str, path: str, timeout: float) -> Dict[str, Any]:
        req = urllib.request.Request(url + path, headers={"X-Flux-Token": token, "X-Flux-Client": "watch"})
        try:
            with self.opener.open(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise UnknownJob(path) from None
            raise Unreachable(f"HTTP {e.code}") from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise Unreachable(type(e).__name__) from None

    def get(self, path: str, timeout: float = 10) -> Dict[str, Any]:
        if self.found is None:
            self.found = self._discover()
        try:
            return self._get(*self.found, path, timeout)
        except Unreachable:
            self.found = None
            raise


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="flux-plumb-watch",
        description="Print one line per Plumber job event (for Claude Code's Monitor tool). "
        "Exit 0: every watched job ok; 1: one ended otherwise; 2: usage; 3: unknown job; 4: Plumber unreachable.",
    )
    p.add_argument("--job", action="append", default=[], help="job id to watch (repeatable)")
    p.add_argument("--session", help="only events of jobs started by this MCP session")
    p.add_argument("--all", action="store_true", help="every job's events")
    p.add_argument("--until", choices=("finished", "never"), help="finished (default with --job) or never")
    p.add_argument("--quiet", action="store_true", help="print only the final line")
    p.add_argument("--home", help="plumber state directory (default ~/.fluxpype/plumber, or $FLUX_PLUMBER_HOME)")
    return p


def run(args, out=sys.stdout, err=sys.stderr) -> int:
    jobs: List[str] = list(dict.fromkeys(args.job))
    until = args.until or ("finished" if jobs else "never")
    if not (jobs or args.session or args.all):
        print("give --job, --session or --all", file=err)
        return 2
    if until == "finished" and not jobs:
        print("--until finished needs --job", file=err)
        return 2
    if args.session and not notify.clean_id(args.session):
        print("--session must match ^[a-z0-9]{6,24}$", file=err)
        return 2
    home = Path(args.home).expanduser().resolve() if args.home else paths.plumber_home()
    conn = Conn(home)
    ended: Dict[str, str] = {}

    def say(line: str) -> None:
        if not args.quiet:
            out.write(line + "\n")

    def wanted(e: Dict[str, Any]) -> bool:
        if e.get("kind") not in notify.WATCH_KINDS:
            return False
        if e.get("kind") in JOBLESS or args.all:
            return True
        if jobs and e.get("job_id") in jobs:
            return True
        return bool(args.session) and (e.get("origin") or {}).get("session") == args.session

    def finish(cursor: int) -> int:
        ok = sum(1 for s in ended.values() if s == "ok")
        code = 0 if ok == len(jobs) else 1
        out.write(notify.format_done(code, ok, len(jobs) - ok, cursor, jobs[0] if len(jobs) == 1 else None) + "\n")
        out.flush()
        return code

    def snapshot() -> Tuple[str, int]:
        head = conn.get("/api/events?limit=1")
        for j in jobs:
            if not notify.JOB_RE.match(j):
                raise UnknownJob(j)
            view = conn.get(f"/api/jobs/{j}")
            say(notify.format_state(view, head["last_seq"]))
            if view.get("status") in notify.JOB_END:
                ended[j] = view["status"]
        out.flush()
        return head["log_id"], int(head["last_seq"])

    log_id, cursor, down_since = None, 0, None
    while True:
        try:
            if log_id is None:
                ended.clear()
                log_id, cursor = snapshot()
                if until == "finished" and len(ended) == len(jobs):
                    return finish(cursor)
            r = conn.get(f"/api/events?since={cursor}&limit=500&wait={WAIT_S}", timeout=WAIT_S + 15)
            down_since = None
            if r.get("log_id") != log_id:  # a new database: start over from its current state
                log_id = None
                continue
            for e in r.get("events", []):
                cursor = max(cursor, int(e.get("seq") or 0))
                if not wanted(e):
                    continue
                say(notify.format_line(e))
                if e.get("kind") in OUTCOME_STATUS and e.get("job_id") in jobs:
                    ended[e["job_id"]] = OUTCOME_STATUS[e["kind"]]
            out.flush()
            if until == "finished" and len(ended) == len(jobs):
                return finish(cursor)
        except UnknownJob as e:
            print(f"unknown job: {e}", file=err)
            return 3
        except Unreachable as e:
            now = time.monotonic()
            down_since = down_since or now
            if now - down_since >= GIVE_UP_S:
                print(f"Plumber for {home} unreachable for {GIVE_UP_S:.0f} s ({e})", file=err)
                return 4
            time.sleep(RETRY_S)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
