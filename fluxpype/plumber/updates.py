"""
Check for and apply updates to the fluxpype checkout Plumber runs from.

Plumber runs from a git checkout (``pip install -e``). ``status`` fetches the branch's upstream and
reports how far behind or ahead the checkout is, what the new commits are, and whether an update
can be applied safely. ``apply`` only fast-forwards: it refuses when tracked files have local edits,
when the checkout has commits of its own, or when there is no upstream. After an update the server
restarts itself in place (``restart_in_place``); detached FLUX jobs keep running, as with any restart.
"""

import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

FETCH_TIMEOUT = 30.0
CHECK_TTL = 600.0
# changes to these mean the Python environment may need a reinstall, which a restart does not do
DEP_FILES = ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt")


class UpdateError(RuntimeError):
    pass


def _git(repo: str, *args: str, timeout: float = 20.0) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "", "LC_ALL": "C"}
    try:
        r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, timeout=timeout, env=env)
    except FileNotFoundError:
        raise UpdateError("git is not installed")
    except subprocess.TimeoutExpired:
        raise UpdateError(f"git {args[0]} took longer than {timeout:.0f} s")
    if r.returncode != 0:
        msg = [ln for ln in (r.stderr or r.stdout or "").splitlines() if ln.strip()]
        raise UpdateError(f"git {args[0]}: {msg[-1] if msg else 'failed'}")
    return r.stdout.strip()


def head(repo: str) -> Optional[str]:
    try:
        return _git(repo, "rev-parse", "--short=10", "HEAD")
    except UpdateError:
        return None


def _log(repo: str, rng: str, limit: int = 30) -> List[Dict[str, Any]]:
    out = _git(repo, "log", f"--max-count={limit}", "--format=%h%x1f%s%x1f%an%x1f%ct", rng)
    rows = []
    for line in out.splitlines():
        parts = line.split("\x1f")
        if len(parts) == 4:
            rows.append({"sha": parts[0], "subject": parts[1], "author": parts[2], "time": int(parts[3])})
    return rows


class Updater:
    def __init__(self, repo: str):
        self.repo = str(repo)
        self.lock = threading.Lock()
        self._last: Optional[Dict[str, Any]] = None
        self._at = 0.0

    def status(self, fetch: bool = True, max_age: float = CHECK_TTL) -> Dict[str, Any]:
        """Where the checkout stands against its upstream. `fetch` contacts the remote (cached `max_age` s)."""
        with self.lock:
            if self._last and time.time() - self._at < max_age and not (fetch and self._last.get("fetched") is None):
                return self._last
            out = self._status(fetch)
            self._last, self._at = out, time.time()
            return out

    def _status(self, fetch: bool) -> Dict[str, Any]:
        base: Dict[str, Any] = {"repo": self.repo, "checked_at": time.time()}
        try:
            _git(self.repo, "rev-parse", "--is-inside-work-tree")
        except UpdateError:
            return {**base, "supported": False, "reason": "Plumber is not running from a git checkout"}
        base["supported"] = True
        base["head"] = head(self.repo)
        try:
            base["branch"] = _git(self.repo, "symbolic-ref", "--quiet", "--short", "HEAD")
        except UpdateError:
            base["branch"] = None
        try:
            upstream = _git(self.repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        except UpdateError:
            upstream = None
        base["upstream"] = upstream
        dirty = [
            ln.split(None, 1)[-1]
            for ln in _git(self.repo, "status", "--porcelain", "--untracked-files=no").splitlines()
        ]
        base["dirty"] = dirty[:20]
        if not upstream:
            return {
                **base,
                "behind": 0,
                "ahead": 0,
                "can_update": False,
                "reason": "this branch has no upstream to update from"
                + (f" (on {base['branch']})" if base["branch"] else " (detached HEAD)"),
                "fetched": None,
            }
        remote, _, rbranch = upstream.partition("/")
        fetched: Optional[bool] = None
        fetch_error = None
        if fetch:
            try:
                _git(self.repo, "fetch", "--quiet", remote, rbranch, timeout=FETCH_TIMEOUT)
                fetched = True
            except UpdateError as e:
                fetched, fetch_error = False, str(e)
        counts = _git(self.repo, "rev-list", "--left-right", "--count", f"HEAD...{upstream}").split()
        ahead, behind = (int(counts[0]), int(counts[1])) if len(counts) == 2 else (0, 0)
        new = _log(self.repo, f"HEAD..{upstream}") if behind else []
        deps = []
        if behind:
            changed = _git(self.repo, "diff", "--name-only", f"HEAD...{upstream}").splitlines()
            deps = [f for f in changed if os.path.basename(f) in DEP_FILES]
        reason = None
        if dirty:
            reason = "tracked files have local edits; commit or stash them first"
        elif ahead:
            reason = f"this checkout has {ahead} commit(s) not on {upstream}; update it by hand"
        elif not behind:
            reason = "already up to date"
        return {
            **base,
            "behind": behind,
            "ahead": ahead,
            "commits": new,
            "deps_changed": deps,
            "can_update": bool(behind and not dirty and not ahead),
            "reason": reason,
            "fetched": fetched,
            "fetch_error": fetch_error,
        }

    def apply(self) -> Dict[str, Any]:
        """Fast-forward to the upstream. Raises UpdateError when that is not safe."""
        st = self.status(fetch=True, max_age=0)
        if not st.get("supported"):
            raise UpdateError(st.get("reason") or "updates are not supported here")
        if not st.get("can_update"):
            raise UpdateError(st.get("reason") or "nothing to update")
        before = st["head"]
        with self.lock:
            _git(self.repo, "merge", "--ff-only", "--quiet", st["upstream"], timeout=60)
            self._last = None
        return {"from": before, "to": head(self.repo), "commits": st["commits"], "deps_changed": st["deps_changed"]}


def restart_argv(argv: List[str]) -> List[str]:
    """The command that starts this Plumber again: same flags, never a new token, never a new browser tab."""
    args = [a for a in argv[1:] if a not in ("--rotate-token",)]
    if "--headless" not in args and "--no-browser" not in args:
        args.append("--no-browser")
    return [sys.executable, "-m", "fluxpype.plumber.cli", *args]


def restart_in_place(delay: float = 0.8) -> None:
    """Replace this process with a fresh Plumber after `delay` s (same pid, so launchd sees no exit).

    File descriptors Python opened are not inherited, so the port and the single-instance lock are
    released by the exec and taken again by the new process."""

    def go():
        time.sleep(delay)
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:  # noqa: BLE001
                pass
        argv = restart_argv(sys.argv)
        os.execv(argv[0], argv)

    threading.Thread(target=go, name="plumber-restart", daemon=True).start()
