"""The Ask bar's backend: answers questions about FLUX and Plumber by running the Claude Code CLI headless.

It uses the CLI's own login (no key for Plumber to hold). The CLI is stripped down: no built-in tools, no user
settings, and only Plumber's MCP server with its read-only tools, so a question costs about a cent. The FLUX reference
and the Plumber guide (``ask/*.md``) are the system prompt. Nothing here changes Plumber: the model proposes changes as a
``flux-action`` block and the person clicks to run one. See docs/plumber-ask-bar-scope.md.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

DOCS = Path(__file__).parent / "ask"
MODEL = "sonnet"
TIMEOUT_S = 120
MAX_BUDGET_USD = "1"
READ_TOOLS = ["status", "list_profiles", "get_profile", "list_jobs", "get_job", "get_task", "get_relax_series", "read_task_log", "cache_summary"]
NO_TOOLS = ["submit_job", "cancel", "retry_failed", "set_queue", "wait_for_task", "get_task_image"]
_busy = threading.Lock()  # one question at a time: each one is a process and spends the user's quota

PREAMBLE = (
    "You are the assistant built into Fluxpype Plumber, a web app for running the fluxpype pipeline, used by solar "
    "physicists. Below are the FLUX reference and a guide to Plumber. Answer from them.\n\n"
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


def system_prompt() -> str:
    return PREAMBLE + "\n\n".join((DOCS / n).read_text(encoding="utf-8") for n in ("plumber_guide.md", "flux_reference.md"))


def user_prompt(question: str, page: str, history: List[Dict[str, str]]) -> str:
    # the page and earlier turns go in the user message so the system prompt stays identical and caches
    earlier = "".join(f"Earlier question: {h.get('q', '')[:500]}\nEarlier answer: {h.get('a', '')[:1500]}\n\n" for h in history[-3:])
    return f"{earlier}The person is on this page: {page or 'unknown'}.\n\nQuestion: {question}"


def command(cli: str, home: str, tools: bool) -> List[str]:
    mcp: Dict[str, Any] = {"mcpServers": {}}
    if tools:
        mcp["mcpServers"]["fluxpype"] = {
            "command": sys.executable,
            "args": ["-m", "fluxpype.plumber.mcp_server", "--home", home, "--no-autostart"],
            "env": {"FLUX_MCP_CLIENT": "ask"},
        }
    cmd = [
        cli, "-p", "--model", MODEL, "--output-format", "stream-json", "--verbose", "--include-partial-messages",
        "--no-session-persistence", "--tools", "", "--strict-mcp-config", "--mcp-config", json.dumps(mcp),
        "--setting-sources", "", "--disable-slash-commands", "--max-budget-usd", MAX_BUDGET_USD,
        "--system-prompt", system_prompt(),
    ]
    if tools:
        cmd += ["--allowedTools", ",".join(f"mcp__fluxpype__{t}" for t in READ_TOOLS)]
        cmd += ["--disallowedTools", ",".join(f"mcp__fluxpype__{t}" for t in NO_TOOLS)]
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
    if d.get("type") == "result":
        if d.get("is_error"):
            return {"type": "error", "message": str(d.get("result") or "the model returned an error")[:500]}
        return {"type": "done", "cost": d.get("total_cost_usd"), "ms": d.get("duration_ms")}
    return None


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
    proc: Optional[subprocess.Popen] = None
    timer: Optional[threading.Timer] = None
    errfile = tempfile.TemporaryFile()
    finished = False
    try:
        cwd = Path(home) / "ask"  # empty, so no project CLAUDE.md is picked up
        cwd.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            command(cli, home, availability()["tools"]), cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errfile, text=True, bufsize=1
        )
        timer = threading.Timer(TIMEOUT_S, proc.kill)
        timer.start()
        assert proc.stdin and proc.stdout
        proc.stdin.write(user_prompt(question, page, history))
        proc.stdin.close()  # the question travels on stdin, so nothing the person types can be read as a flag
        for line in proc.stdout:
            ev = parse(line)
            if ev:
                finished = finished or ev["type"] in ("done", "error")
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
        _busy.release()
