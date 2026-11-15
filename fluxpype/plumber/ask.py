"""The Ask bar's backend: answers questions about FLUX and Plumber by running the Claude Code CLI headless.

It uses the CLI's own login (no key for Plumber to hold). The CLI is stripped down: no built-in tools, no user
settings, and only Plumber's MCP server with its read-only tools, so a question costs about a cent. The FLUX reference
and the Plumber guide (``ask/*.md``) are the system prompt. Nothing here changes Plumber: the MCP server's write tools
only store proposals (proposals.py), and a person runs one with a click after the answer. See docs/plumber-ask-bar-scope.md.
"""

import importlib.util
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Tuple

from . import paths, proposals
from .conversations import ID_RE, Conversations, ConversationError

DOCS = Path(__file__).parent / "ask"
MODEL = "sonnet"
TIMEOUT_S = 120
MAX_BUDGET_USD = "1"
# Two modes. Explain is the Ask bar as it always was (sonnet, 120 s, $1, a stateless question with the last turns
# re-sent). Co-design is for designing scenes with Plum: a CLI session that resumes between turns, opus by default,
# 600 s and $3 a turn, the scene relax and hand-off proposals, and the FLUX source to read. The mode is fixed when a
# conversation starts (the CLI records the system prompt once per session), so the panel starts a new one to switch.
MODES: Dict[str, Dict[str, Any]] = {
    "explain": {"model": "sonnet", "timeout": 120, "budget": "1", "resume": False},
    "codesign": {"model": "opus", "timeout": 600, "budget": "3", "resume": True},
}
MODELS = ("sonnet", "opus")
CODESIGN_DAILY_CAP_USD = 20.0  # across all Co-design turns today (ask_usage.jsonl); Explain has no cap
SETTINGS_FILE = "ask_settings.json"  # in the Plumber home
SETTING_DEFAULTS: Dict[str, Any] = {"wake_on_results": True, "codesign_daily_cap_usd": CODESIGN_DAILY_CAP_USD, "flux_source": None}
CODESIGN_ONLY = ("relax_scene",)  # proposals that need Co-design
READ_TOOLS = ["status", "list_profiles", "get_profile", "list_jobs", "get_job", "list_external_runs", "get_external_run", "get_task", "get_relax_series", "read_task_log", "cache_summary", "get_task_image", "list_proposals", "list_scenes", "get_scene", "validate_scene", "preview_scene", "get_scene_result"]
PROPOSE_TOOLS = list(proposals.TOOLS) + ["withdraw_proposal"]  # only store or close a proposal (mcp_server)
NO_TOOLS = ["wait_for_task"]  # polls for up to 300 s, past the 120 s a question may take
# Repo documents added to the prompt when the checkout has them (a pip install has only the shipped ask/*.md). The
# rest of the repo is reachable through the read-only tools.
REPO_DOCS = [("docs/mcp/TOOLS.md", "Plumber's MCP server: tools and operation"), ("docs/PORTABILITY.md", "Install and environment notes")]
REPO_TOOLS = "Read,Grep,Glob"
USAGE_FILE, LIMITS_FILE = "ask_usage.jsonl", "ask_limits.json"  # in the Plumber home
TRANSCRIPT_FILE = "ask_transcripts.jsonl"  # in the Plumber home, 0600; never under ask/, which the CLI can read
TRANSCRIPT_MAX_BYTES = 5_000_000  # past this the file is kept as .1 and a new one starts
CONTINUITY_HOURS, CONTINUITY_TURNS = 12, 4  # earlier questions the server puts in front of a new one
_recent: Deque[Tuple[float, int]] = deque(maxlen=200)  # (time, new tokens) of answered questions, for the spend rate
_busy = threading.Lock()  # one question at a time: each one is a process and spends the user's quota
_question = ""  # id of the question running now; a proposal must carry it (server.py, POST /api/proposals)
_mode = "explain"  # the mode of that question: the Co-design proposals are refused in Explain


def current_question() -> str:
    return _question


def current_mode() -> str:
    return _mode


def settings(home: str) -> Dict[str, Any]:
    """The Ask settings (ask_settings.json in the Plumber home) over their defaults; a bad file reads as the defaults."""
    out = dict(SETTING_DEFAULTS)
    try:
        with open(os.path.join(home, SETTINGS_FILE), encoding="utf-8") as f:
            saved = json.load(f)
        if isinstance(saved, dict):
            out.update({k: saved[k] for k in SETTING_DEFAULTS if k in saved})
    except (OSError, ValueError):
        pass
    return out


def save_settings(home: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    """Merge `patch` into the saved settings after checking each value; unknown keys are refused."""
    bad = set(patch) - set(SETTING_DEFAULTS)
    if bad:
        raise ValueError("unknown setting: " + ", ".join(sorted(bad)))
    now = settings(home)
    for k, v in patch.items():
        if k == "wake_on_results" and not isinstance(v, bool):
            raise ValueError("wake_on_results is true or false")
        if k == "codesign_daily_cap_usd" and not (isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1000):
            raise ValueError("codesign_daily_cap_usd is a number of dollars, 0 to 1000")
        if k == "flux_source" and v is not None and not (isinstance(v, str) and os.path.isdir(os.path.expanduser(v))):
            raise ValueError("flux_source is the path of an existing folder, or null")
        now[k] = v
    tmp = os.path.join(home, SETTINGS_FILE + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(now, f)
    os.replace(tmp, os.path.join(home, SETTINGS_FILE))
    return now


def flux_source(home: str) -> Optional[Path]:
    """A fluxon-mhd checkout Co-design may read (never write), if one was named: $FLUX_PLUMBER_FLUX_SOURCE, else
    flux_source in ask_settings.json. Nothing is guessed: config.ini's fl_prefix and fl_mhdlib name installed
    libraries, not necessarily a source tree, so they are not used."""
    raw = os.environ.get("FLUX_PLUMBER_FLUX_SOURCE") or settings(home).get("flux_source")
    p = Path(os.path.expanduser(raw)).resolve() if raw else None
    return p if p and p.is_dir() else None

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


def availability(home: Optional[str] = None) -> Dict[str, Any]:
    cli = find_cli()
    out: Dict[str, Any] = {
        "available": cli is not None,
        "reason": None if cli else "The Claude Code CLI was not found. Install it and sign in, or set FLUX_PLUMBER_CLAUDE to its path.",
        "model": MODEL,
        "tools": importlib.util.find_spec("mcp") is not None,  # live state needs the MCP SDK in Plumber's Python
        "modes": {k: {"model": v["model"], "timeout_s": v["timeout"], "budget_usd": float(v["budget"])} for k, v in MODES.items()},
        "models": list(MODELS),
    }
    if home:
        cfg = settings(home)
        src = flux_source(home)
        out.update(
            settings=cfg,
            codesign={"cap_usd": cfg["codesign_daily_cap_usd"], "spent_today_usd": round(spent_today(home, "codesign"), 4)},
            flux_source=str(src) if src else None,
        )
    return out


def repo_root() -> Optional[Path]:
    root = paths.checkout_root()
    return root if (root / "fluxpype" / "plumber").is_dir() else None


def system_prompt(mode: str = "explain", flux_src: Optional[Path] = None) -> str:
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
    if mode == "codesign":
        parts.append(
            "## This conversation is in Co-design mode\n\nThe tools and proposals marked Co-design in the guide are yours here"
            + (f", and the FLUX source (the fluxon-mhd checkout) is readable at {flux_src} with Read, Grep and Glob. It is read-only and not part of fluxpype; never propose a change to it." if flux_src else ". No FLUX source checkout is configured, so you cannot read it.")
            + " A turn may be a message written by Plumber (not the person) that a job you proposed has finished: read its results and report briefly."
        )
    parts.append((DOCS / "flux_reference.md").read_text(encoding="utf-8"))
    return PREAMBLE + "\n\n".join(parts)


def user_prompt(question: str, page: str, history: List[Dict[str, str]]) -> str:
    # the page and earlier turns go in the user message so the system prompt stays identical and caches
    earlier = "".join(f"Earlier question: {h.get('q', '')[:500]}\nEarlier answer: {h.get('a', '')[:1500]}\n\n" for h in history[-3:])
    return f"{earlier}The person is on this page: {page or 'unknown'}.\n\nQuestion: {question}"


def command(cli: str, home: str, tools: bool, question: str = "", mode: str = "explain", model: Optional[str] = None, session: Optional[str] = None, resume: bool = False) -> List[str]:
    """The CLI's argv. Explain is stateless (`--no-session-persistence`). Co-design keeps a CLI session: the first turn
    names it (`--session-id`), later ones resume it (`--resume`)."""
    cfg = MODES[mode]
    flux_src = flux_source(home) if mode == "codesign" else None
    mcp: Dict[str, Any] = {"mcpServers": {}}
    if tools:
        mcp["mcpServers"]["fluxpype"] = {
            "command": sys.executable,
            "args": ["-m", "fluxpype.plumber.mcp_server", "--home", home, "--no-autostart"],
            "env": {"FLUX_MCP_CLIENT": "ask", "FLUX_ASK_QUESTION": question},  # "ask": its write tools only propose
        }
    cmd = [
        cli, "-p", "--model", model or cfg["model"], "--output-format", "stream-json", "--verbose", "--include-partial-messages",
        *(["--no-session-persistence"] if not session else ["--resume" if resume else "--session-id", session]),
        "--tools", REPO_TOOLS if repo_root() else "", "--strict-mcp-config", "--mcp-config", json.dumps(mcp),
        "--setting-sources", "", "--disable-slash-commands", "--max-budget-usd", cfg["budget"],
        "--system-prompt", system_prompt(mode, flux_src),
    ]
    # Read, Grep and Glob stay out of the allow list: a bare allow rule lets them read any path (the token in the
    # Plumber home included); unlisted, they reach only the cwd and --add-dir. The deny rule locks the home as well.
    offered = [t for t in READ_TOOLS + PROPOSE_TOOLS if mode == "codesign" or t not in CODESIGN_ONLY]
    allowed = [f"mcp__fluxpype__{t}" for t in offered] if tools else []
    if repo_root():
        cmd += ["--add-dir", str(repo_root())]
    if flux_src:  # Read, Grep and Glob only (--tools above), so nothing here can write to it
        cmd += ["--add-dir", str(flux_src)]
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
    if d.get("type") == "assistant":  # the finished message: tool calls with their arguments (the stream events carry none)
        calls = [b for b in (d.get("message") or {}).get("content") or [] if isinstance(b, dict) and b.get("type") == "tool_use"]
        if calls:
            return {"type": "calls", "calls": [{"name": b.get("name", "").replace("mcp__fluxpype__", ""), "input": b.get("input")} for b in calls]}
    if d.get("type") == "rate_limit_event":  # the plan's usage windows (five_hour, seven_day): the account's, not Plumber's
        w = (d.get("rate_limit_info") or {}).get("unifiedWindows") or {}
        return {"type": "limits", "windows": {k: {"utilization": v.get("utilization"), "resets_at": v.get("resetsAt")} for k, v in w.items()}}
    if d.get("type") == "result":
        if d.get("is_error"):
            return {"type": "error", "message": str(d.get("result") or "; ".join(map(str, d.get("errors") or [])) or "the model returned an error")[:500]}
        usage = d.get("modelUsage") or {}
        model = next(iter(usage), None)
        u = usage.get(model) or {}
        tokens = {"input": u.get("inputTokens", 0), "output": u.get("outputTokens", 0), "cache_read": u.get("cacheReadInputTokens", 0), "cache_write": u.get("cacheCreationInputTokens", 0)}
        return {"type": "done", "cost": d.get("total_cost_usd"), "ms": d.get("duration_ms"), "model": model, "tokens": tokens}
    return None


def _note(home: str, ev: Dict[str, Any], mode: str = "explain") -> None:
    """Keep what the CLI reports: one line per answered question, and the latest plan usage windows."""
    try:
        if ev["type"] == "done":
            t = ev["tokens"]
            _recent.append((time.time(), t["input"] + t["cache_write"] + t["output"]))
            with open(os.path.join(home, USAGE_FILE), "a") as f:
                f.write(json.dumps({"ts": time.time(), "model": ev["model"], "cost": ev["cost"] or 0, "ms": ev["ms"], "mode": mode, **ev["tokens"]}) + "\n")
        elif ev["type"] == "limits" and ev["windows"]:
            with open(os.path.join(home, LIMITS_FILE), "w") as f:
                json.dump({"ts": time.time(), "windows": ev["windows"]}, f)
    except OSError:
        pass  # the readout just misses a point


def _log_turn(home: str, rec: Dict[str, Any]) -> None:
    """Append one question to the transcript (0600). Tool results are left out: they are the untrusted bulk."""
    path = os.path.join(home, TRANSCRIPT_FILE)
    try:
        if os.path.getsize(path) > TRANSCRIPT_MAX_BYTES:
            os.replace(path, path + ".1")
    except OSError:
        pass
    try:
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass  # the log just misses a turn


def earlier_turns(home: str, now: Optional[float] = None) -> List[Dict[str, str]]:
    """The last few answered questions of the past hours, from the transcript, as the {q, a} pairs user_prompt takes."""
    now = time.time() if now is None else now
    try:
        with open(os.path.join(home, TRANSCRIPT_FILE)) as f:
            lines = f.read().splitlines()[-50:]
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("answer") and not r.get("error") and now - r.get("ts", 0) <= CONTINUITY_HOURS * 3600:
            out.append({"q": r.get("question", ""), "a": r["answer"]})
    return out[-CONTINUITY_TURNS:]


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


def _usage_rows(home: str) -> List[Dict[str, Any]]:
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
    return rows


def _midnight() -> float:
    t = time.localtime()
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))


def spent_today(home: str, mode: str) -> float:
    """Dollars the CLI reported for `mode` turns since local midnight (rows from before modes were kept count as explain)."""
    midnight = _midnight()
    return sum(float(r.get("cost") or 0) for r in _usage_rows(home) if r["ts"] >= midnight and r.get("mode", "explain") == mode)


def usage(home: str) -> Dict[str, Any]:
    """What the Ask bar has spent today (local time) and the last plan usage windows the CLI reported."""
    rows = _usage_rows(home)
    today = [r for r in rows if r["ts"] >= _midnight()]
    keys = ("input", "output", "cache_read", "cache_write", "cost")
    limits = _limits(home)
    return {
        "model": rows[-1]["model"] if rows else None,
        "today": {"questions": len(today), **{k: sum(r.get(k, 0) or 0 for r in today) for k in keys}},
        "limits": limits,
    }


RESUME_LOST = ("No conversation found", "already in use")  # what the CLI says when a session id cannot be resumed


def _run_cli(cmd: List[str], prompt: str, timeout: int, cwd: Path) -> Iterator[Any]:
    """Run the CLI once: yield its parsed events, then `("exit", returncode, last_words)`. The question goes in on stdin."""
    errfile = tempfile.TemporaryFile()
    proc: Optional[subprocess.Popen] = None
    timer: Optional[threading.Timer] = None
    plain = ""  # the last stdout line that was not an event: the CLI prints some refusals as plain text
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errfile, text=True, bufsize=1)
        timer = threading.Timer(timeout, proc.kill)
        timer.start()
        assert proc.stdin and proc.stdout
        proc.stdin.write(prompt)
        proc.stdin.close()  # the question travels on stdin, so nothing the person types can be read as a flag
        for line in proc.stdout:
            ev = parse(line)
            if ev:
                yield ev
            elif line.strip() and not line.lstrip().startswith("{"):
                plain = line.strip()
        proc.wait()
        errfile.seek(0)
        tail = errfile.read().decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
        yield ("exit", proc.returncode, (plain or tail[0])[:300])
    finally:  # also runs when the browser goes away mid-answer
        if timer:
            timer.cancel()
        if proc and proc.poll() is None:
            proc.kill()
        errfile.close()


def _pairs(conv: Dict[str, Any]) -> List[Dict[str, str]]:
    """The conversation's answered turns as the {q, a} pairs user_prompt re-sends when the CLI has no session to resume."""
    return [{"q": t["q"], "a": t.get("answer", "")} for t in conv["turns"] if t.get("answer") and not t.get("error")][-CONTINUITY_TURNS:]


def stream(
    question: str,
    page: str,
    history: List[Dict[str, str]],
    home: str,
    conversation: Optional[str] = None,
    mode: str = "explain",
    model: Optional[str] = None,
    wait_s: float = 0.0,
    system: bool = False,
    notify: Optional[Callable[[str], None]] = None,
) -> Iterator[str]:
    """Server-sent events for one question.

    `conversation` ties it to a stored conversation (conversations.py): its mode wins over `mode`. Co-design resumes the
    CLI session between turns and, when the CLI cannot resume it, starts a fresh session fed the stored turns. `wait_s`
    is how long to queue behind another question (a wake turn waits; a person's question does not). `system` marks a
    turn Plumber wrote itself (a job finished); `notify(conversation)` is called once the turn is stored."""

    def sse(ev: Dict[str, Any]) -> str:
        return f"data: {json.dumps(ev)}\n\n"

    cli = find_cli()
    if not cli:
        yield sse({"type": "error", "message": availability()["reason"]})
        return
    if model is not None and model not in MODELS:
        yield sse({"type": "error", "message": f"unknown model {model!r}"})
        return
    if mode not in MODES:
        yield sse({"type": "error", "message": f"unknown mode {mode!r}"})
        return
    convs = Conversations(home)
    conv: Optional[Dict[str, Any]] = None
    if conversation is not None:
        try:
            conv = convs.get_or_create(conversation, mode, model or MODES[mode]["model"])
        except ConversationError as e:
            yield sse({"type": "error", "message": str(e)})
            return
        mode = conv["mode"] if conv["mode"] in MODES else mode
    if mode == "codesign":
        cap, spent = float(settings(home)["codesign_daily_cap_usd"]), spent_today(home, "codesign")
        if spent >= cap:
            msg = f"Co-design has spent ${spent:.2f} today, at the ${cap:.2f} daily cap. Use Explain, or raise codesign_daily_cap_usd in {SETTINGS_FILE}."
            if conv and system:  # a wake turn the cap stopped: say so in the conversation instead of staying silent
                convs.add_turn(conv["id"], {"id": uuid.uuid4().hex[:12], "ts": time.time(), "q": question, "answer": "", "system": True, "error": msg})
                notify and notify(conv["id"])
            yield sse({"type": "error", "message": msg})
            return
    if not (_busy.acquire(timeout=wait_s) if wait_s else _busy.acquire(blocking=False)):
        yield sse({"type": "error", "message": "Another question is still running; wait for it to finish."})
        return
    global _question, _mode
    _question, _mode = uuid.uuid4().hex[:12], mode
    cfg = MODES[mode]
    rec: Dict[str, Any] = {"ts": time.time(), "id": _question, "page": page, "question": question, "answer": "", "tools": [], "proposals": [], "error": None, "mode": mode}
    if conv:
        rec["conversation"] = conv["id"]
    finished, turn_cost = False, 0.0
    try:
        cwd = Path(home) / "ask"  # empty, so no project CLAUDE.md is picked up
        cwd.mkdir(parents=True, exist_ok=True)
        tools = availability()["tools"]
        keep_session = bool(conv and cfg["resume"])
        if keep_session:
            attempts = [(conv["session"], bool(conv.get("started")))]
            if attempts[0][1]:
                attempts.append((str(uuid.uuid4()), False))  # the fallback if the CLI has lost the session
        else:
            attempts = [(None, False)]
        for n, (sess, resume) in enumerate(attempts):
            if keep_session:  # marked at launch: a turn that dies midway still leaves a session behind to resume
                conv = convs.update(conv["id"], session=sess, started=True) or conv
            earlier = _pairs(conv) if conv and not resume else [] if resume else (earlier_turns(home) or history)
            cmd = command(cli, home, tools, _question, mode, model, sess, resume)
            retry = False
            can_retry = resume and n + 1 < len(attempts)
            for ev in _run_cli(cmd, user_prompt(question, page, earlier), cfg["timeout"], cwd):
                if isinstance(ev, tuple):  # the process ended
                    if not finished and not retry:
                        if can_retry and any(m in ev[2] for m in RESUME_LOST):
                            retry = True
                        else:
                            rec["error"] = f"claude exited with code {ev[1]}. {ev[2]}"
                            yield sse({"type": "error", "message": rec["error"]})
                    continue
                _note(home, ev, mode)
                if ev["type"] == "text":
                    rec["answer"] += ev["text"]
                elif ev["type"] == "calls":
                    rec["tools"] += [{"name": c["name"], "input": json.dumps(c["input"])[:500]} for c in ev["calls"]]
                elif ev["type"] == "done":
                    rec.update(model=ev["model"], cost=ev["cost"])
                    turn_cost = float(ev["cost"] or 0)
                elif ev["type"] == "error":
                    if can_retry and not rec["answer"] and any(m in ev["message"] for m in RESUME_LOST):
                        retry = True  # the session is gone: show nothing of this, go again with the stored turns
                        continue
                    rec["error"] = ev["message"]
                if ev["type"] in ("limits", "calls"):
                    continue  # kept for the dashboard or the log, nothing for the answer panel
                finished = finished or ev["type"] in ("done", "error")
                ids = proposals.for_question(os.path.join(home, "proposals.json"), _question) if ev["type"] == "done" else []
                rec["proposals"] = ids
                if ids:  # written by Plumber, not the model: the answer panel draws a card per id from GET /api/proposals
                    block = f"\n\n```flux-proposals\n{json.dumps(ids)}\n```\n"
                    rec["answer"] += block
                    yield sse({"type": "text", "text": block})
                if ev["type"] == "done" and conv:
                    ev = {**ev, "turn_cost": turn_cost, "conversation_cost": round(float(conv.get("cost", 0)) + turn_cost, 6), "conversation": conv["id"], "mode": mode}
                yield sse(ev)
            if not retry:
                break
            rec["answer"], rec["error"] = "", None
    finally:  # also runs when the browser goes away mid-answer
        if not finished and not rec["error"]:
            rec["error"] = "stopped before the answer finished"  # timeout or the browser left
        _log_turn(home, rec)
        if conv:
            convs.add_turn(
                conv["id"],
                {k: rec.get(k) for k in ("id", "ts", "tools", "proposals", "error", "model")} | {"q": question, "answer": rec["answer"], "cost": turn_cost, "system": system, "page": page},
                turn_cost,
            )
            notify and notify(conv["id"])
        _question, _mode = "", "explain"
        _busy.release()


WAKE_STATUSES = ("ok", "failed", "partial")  # a cancelled job is the person's own doing: no report
WAKE_WAIT_S = 3600.0  # how long a wake turn queues behind a running question before giving up


def wake_text(view: Dict[str, Any]) -> str:
    """The message Plumber writes into a Co-design conversation when a job from one of its proposals ends. Ids,
    statuses and counts only: nothing from a log or a file, so there is nothing in it to follow."""
    tasks = view.get("tasks") or []
    rows = ", ".join(f"{t['id']} {t['status']}" for t in tasks[:12]) + (f", and {len(tasks) - 12} more" if len(tasks) > 12 else "")
    return (
        f"[Plumber, not the person] Job {view['id']} (preset {view.get('profile')}), which you proposed and the person ran, "
        f"has finished with status {view['status']}. Tasks: {rows}. Read the results with get_job, get_scene_result or "
        "read_task_log as needed and report to the person briefly: what ran, what it shows, and what you would do next. "
        "Do not propose anything unless the result calls for it."
    )


class Waker:
    """Turns a finished job into one Plum turn in the Co-design conversation that proposed it.

    Plumber, not the model, decides: the job's origin names the Ask question and proposal that started it, the question
    id finds the conversation, and the turn is queued behind any running question (the one-question lock) by a single
    worker. A job wakes its conversation once; the setting `wake_on_results` and the daily cap apply."""

    def __init__(self, home: str, notify: Optional[Callable[[str], None]] = None):
        self.home, self.notify = str(home), notify
        self.convs = Conversations(self.home)
        self.q: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def job_finished(self, view: Dict[str, Any]) -> bool:
        """Called on every job change; True when a wake turn was queued."""
        origin = view.get("origin") or {}
        if view.get("status") not in WAKE_STATUSES or origin.get("kind") != "ask" or not origin.get("question"):
            return False
        if not settings(self.home)["wake_on_results"]:
            return False
        cid = self.convs.find_by_turn(origin["question"])
        rec = self.convs.load(cid) if cid else None
        if not rec or rec["mode"] != "codesign" or not self.convs.mark_woken(rec["id"], view["id"]):
            return False
        self.q.put((rec["id"], wake_text(view)))
        with self.lock:
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._work, daemon=True, name="ask-wake")
                self.thread.start()
        return True

    def _work(self) -> None:
        while True:
            try:
                cid, text = self.q.get(timeout=30)
            except queue.Empty:
                return  # restarted on the next wake
            rec = self.convs.load(cid)
            if rec:
                for _ in stream(text, "", [], self.home, conversation=cid, mode=rec["mode"], model=rec.get("model"), wait_s=WAKE_WAIT_S, system=True, notify=self.notify):
                    pass
