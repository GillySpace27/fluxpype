"""Golden master for the shape of the event API: `GET /api/events`, one event record, and the WebSocket `event` message.

    python tests/events_api_golden.py --update    rewrite tests/golden/events_api.golden

The golden file lists every key path with its JSON type. Clients (the web app, `watch`, the MCP server) read these
keys, so test_notify_golden.py fails when a path disappears, is renamed or changes type; a new path fails until the
golden file is updated on purpose.
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOLDEN = os.path.join(HERE, "golden", "events_api.golden")
TOKEN = "golden-token"


def kind(v):
    return {dict: "object", list: "array", str: "string", bool: "bool", int: "int", float: "number"}.get(
        type(v), "null" if v is None else type(v).__name__
    )


def paths(obj, prefix):
    out = {prefix: kind(obj)}
    if isinstance(obj, dict) and not prefix.endswith(".f"):  # `f` holds per-kind fields, pinned by the line golden
        for k, v in obj.items():
            out.update(paths(v, f"{prefix}.{k}"))
    elif isinstance(obj, list) and obj:
        out.update(paths(obj[0], f"{prefix}[]"))
    return out


def shapes():
    sys.path.insert(0, ROOT)
    from fastapi.testclient import TestClient

    from fluxpype.plumber import config_io
    from fluxpype.plumber import paths as ppaths
    from fluxpype.plumber.backends.fake import FakeBackend
    from fluxpype.plumber.scheduler import Scheduler
    from fluxpype.plumber.server import create_app
    from fluxpype.plumber.store import Store

    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp) / "home"
        home.mkdir()
        cfg = Path(tmp) / "config.ini"
        cfg.write_text(config_io.set_value(ppaths.shipped_config().read_text(), "DEFAULT", "data_dir", tmp))
        store = Store(ppaths.db_path(home))
        sched = Scheduler(store, {"fake": FakeBackend()}, cfg, home=home)
        app = create_app(sched, TOKEN, tick_interval=0.1)
        try:
            with TestClient(app, base_url="http://localhost") as c:
                end = time.time() + 10
                while not store.last_seq() and time.time() < end:
                    time.sleep(0.05)
                sample = {
                    "kind": "task_failed",
                    "sev": "error",
                    "key": "task_end|j1a2b3c4d-01|1",
                    "job_id": "j1a2b3c4d",
                    "task_id": "j1a2b3c4d-01",
                    "origin": {"kind": "mcp", "session": "abc123def456", "tab": "a1b2c3d4e5f6"},
                    "f": {"attempt": 1, "exit": 1},
                }
                c.cookies.set("flux_plumber", TOKEN)
                with c.websocket_connect("/api/ws", headers={"Host": "localhost:8765"}) as ws:
                    ws.receive_json()  # hello
                    app.state.notifier.emit(sample)
                    msg = ws.receive_json()
                    while msg.get("type") != "event":
                        msg = ws.receive_json()
                r = c.get("/api/events?min_sev=error", headers={"X-Flux-Token": TOKEN}).json()
        finally:
            store.close()
    out = paths(r, "response")
    out.update(paths(r["events"][0], "event"))
    out.update(paths(msg, "ws"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args()
    rows = shapes()
    text = "".join(f"{k} {v}\n" for k, v in sorted(rows.items()))
    if a.update:
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as f:
            f.write(text)
        print("wrote", GOLDEN)
    else:
        print(text, end="")
