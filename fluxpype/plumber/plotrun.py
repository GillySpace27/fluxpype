"""Remake plots from files already on disk (a post-hoc plot run).

``plan``/``run_plots`` turn registry entries (plotcatalog.PLOTS) into commands for one finished task;
``python -m fluxpype.plumber.plotrun RUN_DIR`` is what the scheduler launches. Nothing here edits science code:
a command only ever runs a script that already exists, with arguments filled from the task's own settings.
"""

import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from . import config_io, launcher, paths
from .backends.local import poll_run_dir
from .monitor import ProbeContext
from .plotcatalog import ALLOWED_PLACEHOLDERS, PLOTS

LATER_S = 60  # an input written this long after the task ended counts as changed since
MTIME_SLACK_S = 5.0
_CR, _NFLUX, _ADAPT, _RED = (re.compile(r) for r in (r"^\d{1,5}$", r"^\d{1,6}$", r"^\d$", r"^[\w.]{1,16}$"))


class PlotRunError(ValueError):
    pass


# Settings a remake may change. They land in config.ini text that pipe_helper.convert_value passes to eval(), so each is
# parsed and rewritten here (a number in range, or a name from a fixed list), never passed through as typed.
SETTINGS: Dict[str, List[Dict[str, Any]]] = {
    "light": [
        dict(
            key="light_method",
            label="Method",
            kind="choice",
            choices=["voronoi", "soft_voronoi", "fluxel", "vertex"],
            default="fluxel",
            hint="How the fluxons are turned into a density field (fluxlight.py).",
        ),
        dict(
            key="fov",
            label="Field of view (Rsun)",
            kind="float",
            min=0.5,
            max=300,
            default="215.0",
            hint="Image half-width in solar radii. Small values zoom on the low corona.",
        ),
        dict(
            key="nxy",
            label="Pixels per side",
            kind="int",
            min=16,
            max=2000,
            default="301",
            hint="Image size. Cost grows with the square.",
        ),
        dict(
            key="nz",
            label="Line-of-sight steps",
            kind="int",
            min=2,
            max=2000,
            default="301",
            hint="Integration steps along each line of sight. Cost grows linearly.",
        ),
        dict(
            key="influence_length",
            label="Influence length (Rsun)",
            kind="float",
            min=0.01,
            max=50,
            default="1.0",
            hint="Passed to the forward model as its influence length.",
        ),
        dict(
            key="observer_lon",
            label="Observer longitude (deg)",
            kind="float",
            min=-360,
            max=720,
            default="",
            hint="Carrington longitude of the viewer. Leave blank to keep the job's value (pole-on if it has none).",
        ),
    ],
    "wind": [
        dict(
            key="flow_method",
            label="Wind method",
            kind="choice",
            choices=["parker", "wsa", "cranmer", "ghosts", "tempest"],
            fmt="[{}]",
            default="",
            hint="The solar wind model. Each method writes its own wind files.",
        ),
    ],
}


def clean_settings(group: str, raw: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Validated {key: config text} for `group`; an empty value means 'keep the job's'. Raises PlotRunError."""
    spec = {f["key"]: f for f in SETTINGS.get(group, [])}
    out: Dict[str, str] = {}
    for k, v in (raw or {}).items():
        f = spec.get(k)
        v = str(v).strip()
        if f is None:
            raise PlotRunError(f"{k!r} is not a setting this step takes")
        if v == "":
            continue
        if f["kind"] == "choice":
            if v not in f["choices"]:
                raise PlotRunError(f"{k}: choose one of {', '.join(f['choices'])}")
            out[k] = f.get("fmt", '"{}"').format(v)
            continue
        if not re.match(r"^[-+]?\d+(\.\d*)?$" if f["kind"] == "float" else r"^\d+$", v):
            raise PlotRunError(f"{k}: {v!r} is not a {'whole ' if f['kind'] == 'int' else ''}number")
        x = float(v)
        if not f["min"] <= x <= f["max"]:
            raise PlotRunError(f"{k}: must be between {f['min']} and {f['max']}")
        out[k] = str(int(x)) if f["kind"] == "int" else repr(x)
    return out


def context_vars(ctx: ProbeContext, python: str) -> Dict[str, str]:
    """Placeholder values for one task, validated: they end up in file names and argv."""
    for name, rx in (("cr", _CR), ("nflux", _NFLUX), ("adapt", _ADAPT)):
        if not rx.match(str(getattr(ctx, name))):
            raise PlotRunError(f"bad {name} {getattr(ctx, name)!r}")
    if not config_io.BATCH_RE.match(ctx.batch or "") or ".." in ctx.batch:
        raise PlotRunError(f"bad batch {ctx.batch!r}")
    red = ctx.reduce if ctx.inst == "hmi" else "f" + ctx.adapt_select
    if not _RED.match(str(red)):
        raise PlotRunError(f"bad reduce/adapt_select {red!r}")
    root = paths.checkout_root() / "fluxpype"
    return {
        "python": python,
        "plotting": str(root / "plotting"),
        "science": str(root / "science"),
        "data_dir": ctx.data_dir,
        "batch": ctx.batch,
        "batch_dir": os.path.join(ctx.data_dir, "batches", ctx.batch),
        "cr": str(ctx.cr),
        "nflux": str(ctx.nflux),
        "adapt": str(ctx.adapt),
        "inst": ctx.inst,
        "red": str(red),
    }


def _natural(s: str):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", s)]


def _inside(path: str, data_dir: str) -> bool:
    return os.path.realpath(path).startswith(os.path.realpath(data_dir) + os.sep)


def _fill(template: str, v: Dict[str, str]) -> str:
    try:
        return template.format(**v)
    except (KeyError, IndexError, ValueError) as e:
        raise PlotRunError(f"cannot fill {template!r}: {e}") from e


def resolve_needs(entry: Dict[str, Any], ctx: ProbeContext) -> Tuple[Dict[str, str], List[str]]:
    """({need: absolute path of its last match in natural order}, [data_dir-relative globs with no usable match])."""
    v = context_vars(ctx, sys.executable)
    found, missing = {}, []
    for name, tmpl in (entry.get("needs") or {}).items():
        rel = _fill(tmpl, v)
        hits = [
            p
            for p in glob.glob(os.path.join(glob.escape(ctx.data_dir), rel))
            if os.path.isfile(p) and _inside(p, ctx.data_dir)
        ]
        if hits:
            found[name] = sorted(hits, key=_natural)[-1]
        else:
            missing.append(rel)
    return found, missing


def resolve_command(entry: Dict[str, Any], ctx: ProbeContext, python: str, inputs: Dict[str, str]) -> List[str]:
    if not entry.get("command"):
        raise PlotRunError(f"{entry['id']} has no command")
    v = {**context_vars(ctx, python), **{f"in_{k}": p for k, p in inputs.items()}}
    argv = [_fill(a, v) for a in entry["command"]]
    if any("{" in a or "}" in a for a in argv):
        raise PlotRunError(f"unresolved placeholder in {argv}")
    return argv


def _pattern(entry: Dict[str, Any], ctx: ProbeContext) -> str:
    return entry["pattern"].replace("{cr}", str(ctx.cr)).replace("{nflux}", str(ctx.nflux))


def targets(entry: Dict[str, Any], ctx: ProbeContext) -> List[str]:
    """Existing files (absolute) the plot is written to, confined to the data folder."""
    batch_dir = os.path.join(ctx.data_dir, "batches", ctx.batch)
    pat = os.path.normpath(os.path.join(glob.escape(batch_dir), _pattern(entry, ctx)))
    return sorted(p for p in glob.glob(pat) if os.path.isfile(p) and _inside(p, ctx.data_dir))


def _rel(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def generatable(entry: Dict[str, Any], ctx: ProbeContext, ended: Optional[float] = None) -> Dict[str, Any]:
    if not entry.get("post_hoc"):
        return {"can_generate": False, "missing_inputs": [], "inputs": [], "inputs_later": []}
    found, missing = resolve_needs(entry, ctx)
    inputs = {k: _rel(p, ctx.data_dir) for k, p in found.items()}
    later = []
    if ended:
        later = [inputs[k] for k, p in found.items() if os.path.getmtime(p) > ended + LATER_S]
    return {
        "can_generate": not missing,
        "missing_inputs": missing,
        "inputs": sorted(inputs.values()),
        "inputs_later": sorted(later),
    }


def plan(ctx: ProbeContext, ids: List[str], python: str, plots: List[Dict[str, Any]] = PLOTS) -> List[Dict[str, Any]]:
    by_id = {e["id"]: e for e in plots}
    items = []
    for i in ids:
        e = by_id.get(i)
        if e is None:
            raise PlotRunError(f"unknown plot {i!r}")
        if not e.get("post_hoc"):
            raise PlotRunError(f"{i} cannot be made on its own: {e.get('post_hoc_reason')}")
        found, missing = resolve_needs(e, ctx)
        items.append(
            {
                "id": i,
                "argv": None if missing else resolve_command(e, ctx, python, found),
                "missing": missing,
                "conflicts": [_rel(p, ctx.data_dir) for p in targets(e, ctx)],
                "group": e.get("group"),
            }
        )
    return items


def fake_name(pattern: str) -> str:
    """A concrete file name the glob `pattern` matches: each [...] becomes its first character, * becomes x."""
    return re.sub(r"\*", "x", re.sub(r"\[(.)[^\]]*\]", r"\1", pattern))


# ---------------------------------------------------------------------------------------------- the runner


def _event(path: str, **kw) -> None:
    line = (json.dumps({"ts": time.time(), **kw}) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def _kill_group(proc: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass


def _fake_kind(entry: Dict[str, Any]) -> str:
    return {"fieldmap": "fieldmap", "wind_plots": "wind", "fluxlight": "light"}.get(entry["stage"], "world")


def run_plots(
    run_dir: str,
    ctx: ProbeContext,
    ids: List[str],
    *,
    plots: List[Dict[str, Any]] = PLOTS,
    python: str = sys.executable,
    overwrite: bool = False,
    fake: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> int:
    """Run each group of plots once. Returns 0 if all made their file, 1 if any failed, 143 if cancelled."""
    from . import fake_images

    events = os.path.join(run_dir, "events.jsonl")
    by_id = {e["id"]: e for e in plots}
    items = plan(ctx, ids, python, plots)
    batch_dir = os.path.join(ctx.data_dir, "batches", ctx.batch)
    t_run = time.time()
    cancelled = {"flag": False, "proc": None}

    def on_term(signum, frame):
        cancelled["flag"] = True
        if cancelled["proc"] is not None:
            _kill_group(cancelled["proc"])

    old = None
    try:
        old = signal.signal(signal.SIGTERM, on_term)
    except ValueError:  # not the main thread (tests)
        pass
    _event(events, event="plotrun.start", plots=list(ids), overwrite=overwrite)
    results: Dict[str, Dict[str, Any]] = {}

    def end(i: str, status: str, **kw) -> None:
        rec = {
            "id": i,
            "status": status,
            "exit_code": None,
            "elapsed_s": 0.0,
            "detail": None,
            "outputs": [],
            "replaced": [],
        }
        rec.update(kw)
        results[i] = rec
        _event(events, event="plot.end", **rec)

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        groups.setdefault(it["group"] or it["id"], []).append(it)
    for members in groups.values():
        if cancelled["flag"]:
            for it in members:
                end(it["id"], "cancelled", detail="cancelled before it started")
            continue
        for it in members:
            _event(events, event="plot.start", id=it["id"])
        gone = [it for it in members if it["missing"]]
        if gone:
            for it in members:
                end(it["id"], "failed", detail="missing input: " + ", ".join(gone[0]["missing"]))
            continue
        existing = [rel for it in members for rel in it["conflicts"]]
        if existing and not overwrite:
            for it in members:
                end(it["id"], "failed", detail="target exists: " + ", ".join(existing))
            continue
        for rel in existing:
            dst = os.path.join(run_dir, "replaced", rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(os.path.join(ctx.data_dir, rel), dst)
        start = time.time()
        code, detail = 0, None
        if fake:
            for it in members:
                e = by_id[it["id"]]
                out = os.path.join(batch_dir, fake_name(_pattern(e, ctx)))
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "wb") as f:
                    f.write(fake_images.render(_fake_kind(e), seed=it["id"]))
            time.sleep(0.5)
        else:
            argv = members[0]["argv"]
            timeout = by_id[members[0]["id"]].get("timeout_s", 900)
            print(f"[plotrun] {' '.join(it['id'] for it in members)}: {' '.join(argv)}", flush=True)
            try:
                proc = subprocess.Popen(
                    argv,
                    cwd=str(paths.checkout_root()),
                    env={**os.environ, **(env or {})},
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except OSError as e:
                code, detail = None, f"could not start: {e}"
            else:
                cancelled["proc"] = proc
                while True:
                    try:
                        code = proc.wait(timeout=0.2)
                        break
                    except subprocess.TimeoutExpired:
                        if time.time() - start > timeout:
                            _kill_group(proc)
                            code, detail = None, f"timed out after {timeout} s"
                            break
                cancelled["proc"] = None
        elapsed = round(time.time() - start, 2)
        for it in members:
            e = by_id[it["id"]]
            outs = [p for p in targets(e, ctx) if os.path.getmtime(p) >= start - MTIME_SLACK_S]
            kw = dict(
                exit_code=code,
                elapsed_s=elapsed,
                outputs=[_rel(p, ctx.data_dir) for p in outs],
                replaced=[r for r in it["conflicts"]] if overwrite else [],
            )
            if cancelled["flag"]:
                end(it["id"], "cancelled", detail="cancelled", **kw)
            elif detail:
                end(it["id"], "failed", detail=detail, **kw)
            elif code != 0:
                end(it["id"], "failed", detail=f"exit code {code}", **kw)
            elif not outs:
                end(it["id"], "failed", detail=f"exited 0 but wrote no file matching {_pattern(e, ctx)}", **kw)
            else:
                end(it["id"], "ok", **kw)
    if old is not None:
        signal.signal(signal.SIGTERM, old)
    statuses = {r["status"] for r in results.values()}
    rc = 143 if cancelled["flag"] else (1 if "failed" in statuses else 0)
    status = "cancelled" if cancelled["flag"] else ("failed" if rc else "ok")
    _event(events, event="plotrun.end", status=status, exit_code=rc)
    launcher.write_json_atomic(
        os.path.join(run_dir, "result.json"),
        {
            "status": status,
            "exit_code": rc,
            "started": t_run,
            "ended": time.time(),
            "plots": [results[i] for i in ids if i in results],
        },
    )
    return rc


def _read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_view(run_dir: str) -> Dict[str, Any]:
    """The plot run as the API shows it (status from the launcher's files, plots from the event log)."""
    req = _read_json(os.path.join(run_dir, "request.json")) or {}
    titles = {e["id"]: e["title"] for e in PLOTS}
    plots: Dict[str, Dict[str, Any]] = {
        i: {
            "id": i,
            "title": titles.get(i, i),
            "status": "pending",
            "exit_code": None,
            "elapsed_s": None,
            "detail": None,
            "outputs": [],
            "replaced": [],
        }
        for i in req.get("plots", [])
    }
    try:
        with open(os.path.join(run_dir, "events.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                p = plots.get(ev.get("id"))
                if p is None:
                    continue
                if ev.get("event") == "plot.start":
                    p["status"] = "running"
                elif ev.get("event") == "plot.end":
                    p.update({k: ev[k] for k in p if k in ev})
    except OSError:
        pass
    result = _read_json(os.path.join(run_dir, "result.json"))
    st = poll_run_dir(run_dir)
    if st.state == "exited":
        status = (result or {}).get("status") or (
            "cancelled" if st.cancelled else "ok" if st.exit_code == 0 else "failed"
        )
        if st.cancelled and status != "ok":
            status = "cancelled"
    elif st.state == "lost":
        status = "cancelled" if st.cancelled else "crashed"
    else:
        status = st.state
    if status in ("cancelled", "crashed", "failed"):  # whatever never finished did not run
        for p in plots.values():
            if p["status"] in ("pending", "running"):
                p["status"] = "cancelled" if status == "cancelled" else "failed"
                p["detail"] = p["detail"] or ("cancelled" if status == "cancelled" else "the plot run ended early")
    try:
        log_size = os.path.getsize(os.path.join(run_dir, "stdout.log"))
    except OSError:
        log_size = 0
    return {
        "id": os.path.basename(run_dir),
        "task_id": req.get("task_id"),
        "job_id": req.get("job_id"),
        "status": status,
        "created": req.get("created"),
        "started": st.started,
        "ended": (result or {}).get("ended") or st.ended,
        "exit_code": st.exit_code,
        "overwrite": bool(req.get("overwrite")),
        "settings": req.get("settings") or {},
        "plots": list(plots.values()),
        "log_size": log_size,
    }


def main(argv: Optional[List[str]] = None) -> int:
    run_dir = (sys.argv[1:] if argv is None else argv)[0]
    with open(os.path.join(run_dir, "request.json"), "r", encoding="utf-8") as f:
        req = json.load(f)
    os.nice(10)
    ctx = ProbeContext(**req["ctx"])
    return run_plots(
        run_dir,
        ctx,
        req["plots"],
        python=req.get("python") or sys.executable,
        overwrite=bool(req.get("overwrite")),
        fake=bool(req.get("fake")),
    )


if __name__ == "__main__":
    sys.exit(main())
