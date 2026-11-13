"""The Ask bar's backend: answers questions about FLUX and Plumber by running the Claude Code CLI headless.

It uses the CLI's own login (no key for Plumber to hold). The CLI is stripped down: no built-in tools, no user
settings, and only Plumber's MCP server with its read-only tools, so a question costs about a cent. The FLUX reference
and the Plumber guide (``ask/*.md``) are the system prompt. Nothing here changes Plumber: the MCP server's write tools
only store proposals (proposals.py), and a person runs one with a click after the answer. See docs/plumber-ask-bar-scope.md.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple

from . import paths, proposals

DOCS = Path(__file__).parent / "ask"
MODEL = "sonnet"
TIMEOUT_S = 120
MAX_BUDGET_USD = "1"
READ_TOOLS = ["status", "list_profiles", "get_profile", "list_jobs", "get_job", "get_task", "get_relax_series", "read_task_log", "cache_summary", "get_task_image", "list_proposals"]
PROPOSE_TOOLS = list(proposals.TOOLS) + ["withdraw_proposal"]  # only store or close a proposal (mcp_server)
NO_TOOLS = ["wait_for_task"]  # polls for up to 300 s, past the 120 s a question may take
# Repo documents added to the prompt when the checkout has them (a pip install has only the shipped ask/*.md). The
# rest of the repo is reachable through the read-only tools.
REPO_DOCS = [("docs/mcp/TOOLS.md", "Plumber's MCP server: tools and operation"), ("docs/PORTABILITY.md", "Install and environment notes")]
REPO_TOOLS = "Read,Grep,Glob"
USAGE_FILE, LIMITS_FILE = "ask_usage.jsonl", "ask_limits.json"  # in the Plumber home
_recent: Deque[Tuple[float, int]] = deque(maxlen=200)  # (time, new tokens) of answered questions, for the spend rate
_busy = threading.Lock()  # one question at a time: each one is a process and spends the user's quota
_question = ""  # id of the question running now; a proposal must carry it (server.py, POST /api/proposals)


def current_question() -> str:
    return _question

PREAMBLE = (
    "You are Plum, the assistant built into Fluxpype Plumber, a web app for running the fluxpype pipeline, used by solar "
    "physicists. Below are the FLUX reference, a guide to Plumber and the plan for the Scene Designer. Answer from them.\n\n"
)


def find_cli() -> Optional[str]:
    env = os.environ.get("FLUX_PLUMBER_CLAUDE")
    if env:
        return env
    for p in (shutil.which("claude"), *(os.path.expanduser(x) for x in ("~/.local/bin/claude", "/usr/local/bin/claude", "/opt/homebrew/bin/claude"))):
        if p and os.access(p, os.X_OK):
            return p
    return None


def availability() -> Dict[str, Any]:
    cli = find_cli()
    return {
        "available": cli is not None,
        "reason": None if cli else "The Claude Code CLI was not found. Install it and sign in, or set FLUX_PLUMBER_CLAUDE to its path.",
        "model": MODEL,
        "tools": importlib.util.find_spec("mcp") is not None,  # live state needs the MCP SDK in Plumber's Python
    }


def repo_root() -> Optional[Path]:
    root = paths.checkout_root()
    return root if (root / "fluxpype" / "plumber").is_dir() else None


def system_prompt() -> str:
    parts = [(DOCS / n).read_text(encoding="utf-8") for n in ("plumber_guide.md", "scene_designer.md")]
    root = repo_root()
    if root:
        parts.append(
            f"## The repository\n\nThe fluxpype checkout is at {root}. You can read it with Read, Grep and Glob (read-only). "
            "Use it for questions about how something is implemented or configured, and cite the file and line. Skip "
            ".venv*, node_modules and web/dist. config.ini is documented inline; its defaults are below."
        )
        for rel, title in REPO_DOCS:
            f = root / rel
            if f.is_file():
                parts.append(f"## {title} ({rel})\n\n{f.read_text(encoding='utf-8')}")
        ini = root / "fluxpype" / "config.ini"
        if ini.is_file():
            head = ini.read_text(encoding="utf-8").split("\n[quickstart]")[0]  # the [DEFAULT] section: each setting with its comment
            parts.append(f"## config.ini defaults (fluxpype/config.ini)\n\n```ini\n{head}\n```")
    parts.append((DOCS / "flux_reference.md").read_text(encoding="utf-8"))
    return PREAMBLE + "\n\n".join(parts)


def user_prompt(question: str, page: str, history: List[Dict[str, str]]) -> str:
    # the page and earlier turns go in the user message so the system prompt stays identical and caches
    earlier = "".join(f"Earlier question: {h.get('q', '')[:500]}\nEarlier answer: {h.get('a', '')[:1500]}\n\n" for h in history[-3:])
    return f"{earlier}The person is on this page: {page or 'unknown'}.\n\nQuestion: {question}"


def command(cli: str, home: str, tools: bool, question: str = "") -> List[str]:
    mcp: Dict[str, Any] = {"mcpServers": {}}
    if tools:
        mcp["mcpServers"]["fluxpype"] = {
            "command": sys.executable,
            "args": ["-m", "fluxpype.plumber.mcp_server", "--home", home, "--no-autostart"],
            "env": {"FLUX_MCP_CLIENT": "ask", "FLUX_ASK_QUESTION": question},  # "ask": its write tools only propose
        }
    cmd = [
        cli, "-p", "--model", MODEL, "--output-format", "stream-json", "--verbose", "--include-partial-messages",
        "--no-session-persistence", "--tools", REPO_TOOLS if repo_root() else "", "--strict-mcp-config", "--mcp-config", json.dumps(mcp),
        "--setting-sources", "", "--disable-slash-commands", "--max-budget-usd", MAX_BUDGET_USD,
        "--system-prompt", system_prompt(),
    ]
    # Read, Grep and Glob stay out of the allow list: a bare allow rule lets them read any path (the token in the
    # Plumber home included); unlisted, they reach only the cwd and --add-dir. The deny rule locks the home as well.
    allowed = [f"mcp__fluxpype__{t}" for t in READ_TOOLS + PROPOSE_TOOLS] if tools else []
    if repo_root():
        cmd += ["--add-dir", str(repo_root())]
    if allowed:
        cmd += ["--allowedTools", ",".join(allowed)]
    denied = [f"Read(/{os.path.abspath(home)}/**)"] + ([f"mcp__fluxpype__{t}" for t in NO_TOOLS] if tools else [])
    cmd += ["--disallowedTools", ",".join(denied)]
    return cmd


def parse(line: str) -> Optional[Dict[str, Any]]:
    """One line of the CLI's stream-json as a UI event ({type: text|tool|done|error}), or None to ignore it."""
    try:
        d = json.loads(line)
    except ValueError:
        return None
    if d.get("type") == "stream_event":
        e = d.get("event", {})
        if e.get("type") == "content_block_delta" and e.get("delta", {}).get("type") == "text_delta":
            return {"type": "text", "text": e["delta"]["text"]}
        b = e.get("content_block", {})
        if e.get("type") == "content_block_start" and b.get("type") == "tool_use":
            return {"type": "tool", "name": b.get("name", "").replace("mcp__fluxpype__", "")}
    if d.get("type") == "rate_limit_event":  # the plan's usage windows (five_hour, seven_day): the account's, not Plumber's
        w = (d.get("rate_limit_info") or {}).get("unifiedWindows") or {}
        return {"type": "limits", "windows": {k: {"utilization": v.get("utilization"), "resets_at": v.get("resetsAt")} for k, v in w.items()}}
    if d.get("type") == "result":
        if d.get("is_error"):
            return {"type": "error", "message": str(d.get("result") or "the model returned an error")[:500]}
        usage = d.get("modelUsage") or {}
        model = next(iter(usage), None)
        u = usage.get(model) or {}
        tokens = {"input": u.get("inputTokens", 0), "output": u.get("outputTokens", 0), "cache_read": u.get("cacheReadInputTokens", 0), "cache_write": u.get("cacheCreationInputTokens", 0)}
        return {"type": "done", "cost": d.get("total_cost_usd"), "ms": d.get("duration_ms"), "model": model, "tokens": tokens}
    return None


def _note(home: str, ev: Dict[str, Any]) -> None:
    """Keep what the CLI reports: one line per answered question, and the latest plan usage windows."""
    try:
        if ev["type"] == "done":
            t = ev["tokens"]
            _recent.append((time.time(), t["input"] + t["cache_write"] + t["output"]))
            with open(os.path.join(home, USAGE_FILE), "a") as f:
                f.write(json.dumps({"ts": time.time(), "model": ev["model"], "cost": ev["cost"] or 0, "ms": ev["ms"], **ev["tokens"]}) + "\n")
        elif ev["type"] == "limits" and ev["windows"]:
            with open(os.path.join(home, LIMITS_FILE), "w") as f:
                json.dump({"ts": time.time(), "windows": ev["windows"]}, f)
    except OSError:
        pass  # the readout just misses a point


def token_rate(window: float = 60.0) -> float:
    """New tokens (input, cache writes and output; cache reads excluded) the Ask bar spent per second over the last minute."""
    now = time.time()
    return sum(n for t, n in list(_recent) if now - t <= window) / window


def _limits(home: str) -> Optional[Dict[str, Any]]:
    try:
        with open(os.path.join(home, LIMITS_FILE)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def plan(home: str) -> Optional[Dict[str, Any]]:
    """Share of each plan usage window still left, from the CLI's last report; None for a window that has since rolled over."""
    lim = _limits(home)
    if not lim:
        return None
    out: Dict[str, Any] = {"ts": lim["ts"]}
    for k, w in lim["windows"].items():
        u, r = w.get("utilization"), w.get("resets_at")
        out[k] = {"left": None if u is None or (r and r < time.time()) else max(0.0, 1 - u), "resets_at": r}
    return out


def usage(home: str) -> Dict[str, Any]:
    """What the Ask bar has spent today (local time) and the last plan usage windows the CLI reported."""
    rows: List[Dict[str, Any]] = []
    try:
        with open(os.path.join(home, USAGE_FILE)) as f:
            for ln in f.read().splitlines()[-5000:]:
                try:
                    rows.append(json.loads(ln))
                except ValueError:
                    pass
    except OSError:
        pass
    t = time.localtime()
    midnight = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))
    today = [r for r in rows if r["ts"] >= midnight]
    keys = ("input", "output", "cache_read", "cache_write", "cost")
    limits = _limits(home)
    return {
        "model": rows[-1]["model"] if rows else None,
        "today": {"questions": len(today), **{k: sum(r.get(k, 0) or 0 for r in today) for k in keys}},
        "limits": limits,
    }


def stream(question: str, page: str, history: List[Dict[str, str]], home: str) -> Iterator[str]:
    """Server-sent events for one question."""

    def sse(ev: Dict[str, Any]) -> str:
        return f"data: {json.dumps(ev)}\n\n"

    cli = find_cli()
    if not cli:
        yield sse({"type": "error", "message": availability()["reason"]})
        return
    if not _busy.acquire(blocking=False):
        yield sse({"type": "error", "message": "Another question is still running; wait for it to finish."})
        return
    global _question
    _question = uuid.uuid4().hex[:12]
    proc: Optional[subprocess.Popen] = None
    timer: Optional[threading.Timer] = None
    errfile = tempfile.TemporaryFile()
    finished = False
    try:
        cwd = Path(home) / "ask"  # empty, so no project CLAUDE.md is picked up
        cwd.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            command(cli, home, availability()["tools"], _question), cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errfile, text=True, bufsize=1
        )
        timer = threading.Timer(TIMEOUT_S, proc.kill)
        timer.start()
        assert proc.stdin and proc.stdout
        proc.stdin.write(user_prompt(question, page, history))
        proc.stdin.close()  # the question travels on stdin, so nothing the person types can be read as a flag
        for line in proc.stdout:
            ev = parse(line)
            if ev:
                _note(home, ev)
                if ev["type"] == "limits":
                    continue  # kept for the dashboard, nothing for the answer panel
                finished = finished or ev["type"] in ("done", "error")
                ids = proposals.for_question(os.path.join(home, "proposals.json"), _question) if ev["type"] == "done" else []
                if ids:  # written by Plumber, not the model: the answer panel draws a card per id from GET /api/proposals
                    yield sse({"type": "text", "text": f"\n\n```flux-proposals\n{json.dumps(ids)}\n```\n"})
                yield sse(ev)
        proc.wait()
        if not finished:
            errfile.seek(0)
            tail = errfile.read().decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
            yield sse({"type": "error", "message": f"claude exited with code {proc.returncode}. {tail[0][:300]}"})
    finally:  # also runs when the browser goes away mid-answer
        if timer:
            timer.cancel()
        if proc and proc.poll() is None:
            proc.kill()
        errfile.close()
        _question = ""
        _busy.release()
