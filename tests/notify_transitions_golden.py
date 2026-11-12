"""Golden master for notify.task_transition and notify.job_transition: which store change makes which event.

    python tests/notify_transitions_golden.py --update    rewrite tests/golden/notify_transitions.golden

Each corpus entry is (notified, row, job) through task_transition, or (notified, job, tasks) through job_transition;
the golden value is one `kind|sev|key|sorted f` string per event, in order. test_notify_golden.py names every entry
whose events moved.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOLDEN = os.path.join(HERE, "golden", "notify_transitions.golden")
J, T = "j1a2b3c4d", "j1a2b3c4d-01"
WEB = {"origin": {"kind": "web", "tab": "a1b2c3d4e5f6"}}


def task(status, **kw):
    return {"id": T, "job_id": J, "status": status, "attempt": 1, "exit_code": None, "pid": 4242, "started": 10.0, **kw}


TASKS = {
    "new_pending": (None, task("pending", started=None, pid=None), WEB),
    "pending_to_running": ("pending", task("running"), WEB),
    "running_to_running": ("running", task("running"), WEB),
    "running_to_ok": ("running", task("ok", exit_code=0), WEB),
    "pending_to_ok_caught_up": ("pending", task("ok", exit_code=0), WEB),
    "never_seen_to_ok": (None, task("ok", exit_code=0), None),
    "running_to_failed_exit": ("running", task("failed", exit_code=1, stage="pfss"), WEB),
    "running_to_failed_swallowed": ("running", task("failed", exit_code=0, stage="pfss"), WEB),
    "pending_to_failed_submit": ("pending", task("failed", pid=None, started=10.0, ended=10.0), WEB),
    "running_to_timeout": ("running", task("timeout", exit_code=124), WEB),
    "running_to_crashed": ("running", task("crashed"), WEB),
    "running_to_ok_launcher_lost": ("running", task("ok", exit_code=0, launcher_lost=True), WEB),
    "pending_to_cancelled": ("pending", task("cancelled", started=None, pid=None), WEB),
    "running_to_cancelled": ("running", task("cancelled", exit_code=143), WEB),
    "failed_to_pending_retry": ("failed", task("pending", attempt=2, started=None, pid=None), WEB),
    "retry_to_running": ("pending", task("running", attempt=2), WEB),
    "forged_stage_dropped": ("running", task("failed", exit_code=1, stage="pfss\nkind=ok"), WEB),
    "mcp_origin_session": ("running", task("ok", exit_code=0), {"origin": {"kind": "mcp", "session": "abc123def456"}}),
    "bad_origin_dropped": ("running", task("ok", exit_code=0), {"origin": {"kind": "root", "session": "../x"}}),
}


def job(status, **kw):
    return {"id": J, "status": status, "backend": "fake", **WEB, **kw}


def tasks(*statuses, attempt=1):
    return [{"status": s, "attempt": attempt} for s in statuses]


JOBS = {
    "new_job": (None, job("pending"), tasks("pending", "pending")),
    "new_job_already_ok": (None, job("ok"), tasks("ok")),
    "pending_to_running": ("pending", job("running"), tasks("running", "pending")),
    "running_to_ok": ("running", job("ok"), tasks("ok", "ok")),
    "running_to_partial": ("running", job("partial"), tasks("ok", "failed", "timeout")),
    "running_to_failed": ("running", job("failed"), tasks("failed", "crashed")),
    "pending_to_cancelled": ("pending", job("cancelled"), tasks("cancelled")),
    "partial_to_pending_retry": ("partial", job("pending"), tasks("ok", "pending", attempt=2)),
    "retry_ends_ok": ("running", job("ok"), tasks("ok", "ok", attempt=2)),
    "ok_to_ok": ("ok", job("ok"), tasks("ok")),
}


def fmt(events):
    return [
        f"{e['kind']}|{e['sev']}|{e['key']}|{json.dumps(e['f'], sort_keys=True)}|{json.dumps(e['origin'], sort_keys=True)}"
        for e in events
    ]


def table():
    sys.path.insert(0, ROOT)
    from fluxpype.plumber import notify

    out = {f"task:{k}": fmt(notify.task_transition(*v)) for k, v in TASKS.items()}
    out.update({f"job:{k}": fmt(notify.job_transition(*v)) for k, v in JOBS.items()})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args()
    rows = table()
    if a.update:
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as f:
            json.dump(rows, f, indent=1, sort_keys=True, ensure_ascii=True)
            f.write("\n")
        print("wrote", GOLDEN)
    else:
        for k, v in sorted(rows.items()):
            print(k, v)
