"""What one task produced: its products by stage, read from the files on disk.

The monitor only reports images written while a task ran. A results view also
needs products the task reused from the cache, non-image products (FITS,
footpoints, worlds, wind), and products it should have made but did not.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Any, Dict, List, Optional

from .. import names
from .cache import classify
from . import plotrun
from .monitor import ProbeContext
from .plotcatalog import PLOTS

# stage id, label, key in ProbeContext.patterns(), most files to list
PRODUCTS = [
    ("magnetogram", "Magnetogram", "magnetogram", 1),
    ("footpoints", "Footpoints", "footpoints", 4),
    ("pfss", "PFSS model", "pfss", 4),
    ("world", "Initial world", "world", 2),
    ("relaxed_world", "Relaxed world", "relax", 6),
    ("wind", "Wind", "wind", 40),
    ("fluxlight", "FLUXlight", "fluxlight", 40),
]
PLOT_EXT = (".png", ".jpg", ".jpeg", ".gif", ".pdf", ".svg")
SLACK_S = 5.0  # file clocks and the task's start time are compared with this much tolerance


def plot_stage(path: str) -> Optional[str]:
    """The pipeline stage that writes a plot, judged from where it sits (a guess by folder, not a record)."""
    p = "/" + path.replace("\\", "/")
    if "fluxlight" in os.path.basename(p):
        return "fluxlight"
    if "/imgs/relax/" in p:
        return "relax"
    if "/imgs/footpoints/" in p:
        return "fieldmap"
    if "/imgs/world/" in p:
        return "world_plots"
    return "wind_plots" if "/imgs/" in p else None


def for_this_nflux(name: str, nflux: str, inst: Optional[str] = None) -> bool:
    """False only when the file name names a different fluxon count (``_f<N>_``) or, if given, a different instrument."""
    return names.is_this_task(name, nflux, inst)


def made_here(mtime: float, started: Optional[float], ended: Optional[float]) -> Optional[bool]:
    """True if written during the task, False if older (reused), None if the task never started."""
    if started is None:
        return None
    if mtime < started - SLACK_S:
        return False
    if ended is not None and mtime > ended + 60:
        return None  # written after the task by something else; don't claim it either way
    return True


def origin(mtime: float, started: Optional[float], ended: Optional[float]) -> Optional[str]:
    """'made' (written during the task), 'reused' (older), 'later' (rewritten after it ended), or None."""
    if started is None:
        return None
    if mtime < started - SLACK_S:
        return "reused"
    if ended is not None and mtime > ended + 60:
        return "later"
    return "made"


def _rel(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def _file(path: str, root: str, started, ended, posthoc=None) -> Optional[Dict[str, Any]]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    if not os.path.isfile(path):
        return None
    rel = _rel(path, root)
    info = classify(rel)
    extra = {"plot_run": posthoc[rel]} if posthoc and rel in posthoc else {}
    return {
        "path": rel,
        "name": os.path.basename(path),
        "size": st.st_size,
        "mtime": st.st_mtime,
        "made_here": made_here(st.st_mtime, started, ended),
        "origin": "posthoc" if extra else origin(st.st_mtime, started, ended),
        **extra,
        **{k: v for k, v in info.items() if k in ("kind", "cr", "batch", "n", "inst", "reduce", "sel", "steps")},
    }


def _expand(patterns: List[str], first: bool = False) -> List[str]:
    """Files matching any pattern; with ``first`` only those of the first pattern that matches (new name, else old)."""
    out: List[str] = []
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits and first:
            return sorted(set(hits))
        out += hits
    return sorted(set(out))


def plot_catalog(
    ctx: Optional[ProbeContext], started: Optional[float], ended: Optional[float], posthoc=None
) -> List[Dict[str, Any]]:
    """Every plot the pipeline defines, each with the files this task's folders hold for it (none: a placeholder)."""
    out = []
    for e in PLOTS:
        files, n = [], 0
        if ctx:
            batch = os.path.join(ctx.data_dir, "batches", ctx.batch)
            top = os.path.realpath(ctx.data_dir) + os.sep
            for pat in plotrun.pattern_candidates(e, ctx):  # new name first, the old one for an older batch
                hits = [
                    p
                    for p in glob.glob(os.path.normpath(os.path.join(batch, pat)))
                    if os.path.isfile(p) and os.path.realpath(p).startswith(top)
                ]
                if hits:
                    break
            hits.sort(key=os.path.getmtime, reverse=True)
            n = len(hits)
            files = [f for f in (_file(p, ctx.data_dir, started, ended, posthoc) for p in hits[:8]) if f]
        can = plotrun.generatable(e, ctx, ended) if ctx else plotrun.generatable({}, None)
        out.append({**e, **can, "files": files, "n_files": n})
    return out


IMAGE_EXT = (".png", ".jpg", ".jpeg")


def _relative(path: str, root: str) -> str:
    """``path`` relative to the data folder when it sits inside it (events carry absolute paths), else unchanged."""
    if os.path.isabs(path):
        rel = os.path.relpath(path, root)
        if not rel.startswith(".."):
            return rel.replace(os.sep, "/")
    return path


def event_artifacts(artifacts: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """The artifact events of one task (state ``artifacts`` rows with source ``event``), oldest first."""
    rows = [a for a in artifacts or [] if a.get("source") == "event" and a.get("path")]
    return sorted(rows, key=lambda a: a.get("t") or 0)


def pick_headline(artifacts: Optional[List[Dict[str, Any]]], root: Optional[str], fallback=None) -> Optional[str]:
    """The one headline rule (A3-07), used by Results, Overview and MCP.

    The newest image artifact event of the task wins: an event is a product of this task by construction. Only a run
    without such an event falls back to ``fallback()`` (the glob rule over its folders). Returns a path relative to
    ``root`` when the file sits inside it.
    """
    for a in reversed(event_artifacts(artifacts)):
        if a.get("kind", "image") == "image" and a["path"].lower().endswith(IMAGE_EXT):
            return _relative(a["path"], root) if root else a["path"]
    return fallback() if fallback else None


def _glob_headline(products: List[Dict[str, Any]], plots: List[Dict[str, Any]]) -> Optional[str]:
    """Fallback rule: the newest FLUXlight image, else the newest plot image."""
    light = next((p for p in products if p["id"] == "fluxlight"), None)
    images = [f for f in (light["files"] if light else []) if f["name"].lower().endswith(IMAGE_EXT)]
    images += [f for f in plots if f["name"].lower().endswith(IMAGE_EXT)]
    return images[0]["path"] if images else None


def task_headline(
    ctx: Optional[ProbeContext],
    started: Optional[float],
    ended: Optional[float],
    artifacts: Optional[List[Dict[str, Any]]] = None,
    posthoc=None,
) -> Optional[str]:
    """Headline of a task without building its catalog (cheap when an artifact event settles it)."""
    root = ctx.data_dir if ctx else None

    def fallback():
        if ctx is None:
            return None
        products, plots = _collect(ctx, started, ended, posthoc)
        return _glob_headline(products, plots)

    return pick_headline(artifacts, root, fallback)


def _collect(ctx: ProbeContext, started, ended, posthoc=None):
    """Products by stage and the other plots, from the files on disk."""
    root = ctx.data_dir
    pats = ctx.patterns()
    seen = set()
    products = []
    for sid, label, key, limit in PRODUCTS:
        hits = [
            p
            for p in _expand(pats[key], first=key in ("wind", "fluxlight"))
            if for_this_nflux(os.path.basename(p), ctx.nflux, ctx.inst)
        ]
        hits.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)
        files = [f for f in (_file(p, root, started, ended) for p in hits[:limit]) if f]
        seen.update(os.path.realpath(p) for p in hits)
        products.append({"id": sid, "label": label, "files": files, "more": max(0, len(hits) - limit)})

    plots = []
    # plots sit beside the data, and under batches/<batch>/imgs/ (all rotations mixed, so match the name)
    batch = os.path.join(root, "batches", ctx.batch)
    for pat, by_name in [
        (os.path.join(batch, "data", f"cr{ctx.cr}", "**", "*"), False),
        (os.path.join(batch, "imgs", "**", "*"), True),
    ]:
        for p in glob.glob(pat, recursive=True)[:2000]:
            if not p.lower().endswith(PLOT_EXT) or os.path.realpath(p) in seen:
                continue
            if not for_this_nflux(os.path.basename(p), ctx.nflux, ctx.inst):
                continue
            if by_name and f"cr{ctx.cr}" not in os.path.basename(p):
                continue
            f = _file(p, root, started, ended, posthoc)
            if f:
                plots.append({**f, "stage": plot_stage(f["path"])})
    plots.sort(key=lambda f: f["mtime"], reverse=True)
    return products, plots


def _add_reported(products, plots, ctx, started, ended, artifacts, posthoc=None):
    """Put artifact-event files the globs did not find into the product list (FITS under FLUXlight, rest apart)."""
    root = ctx.data_dir
    have = {os.path.realpath(os.path.join(root, f["path"])) for p in products for f in p["files"]}
    have |= {os.path.realpath(os.path.join(root, f["path"])) for f in plots}
    extra = []
    for a in event_artifacts(artifacts):
        full = a["path"] if os.path.isabs(a["path"]) else os.path.join(root, a["path"])
        real = os.path.realpath(full)
        if real in have or not real.startswith(os.path.realpath(root) + os.sep):
            continue
        f = _file(real, root, started, ended, posthoc)
        if not f:
            continue
        have.add(real)
        f["reported"] = a.get("kind", "image")
        extra.append(f)
    light = next((p for p in products if p["id"] == "fluxlight"), None)
    rest = []
    for f in sorted(extra, key=lambda f: f["mtime"], reverse=True):
        if light is not None and f["name"].lower().endswith(".fits") and for_this_nflux(f["name"], ctx.nflux, ctx.inst):
            light["files"].append(f)
        else:
            rest.append(f)
    if rest:
        products.append({"id": "reported", "label": "Reported by the run", "files": rest, "more": 0})


def task_results(
    ctx: Optional[ProbeContext],
    started: Optional[float],
    ended: Optional[float],
    posthoc=None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Products by stage plus the other plots for one task, with paths relative to ctx.data_dir."""
    if ctx is None:
        return {
            "data_dir": None,
            "products": [],
            "plots": [],
            "headline": None,
            "catalog": plot_catalog(None, None, None),
        }
    products, plots = _collect(ctx, started, ended, posthoc)
    _add_reported(products, plots, ctx, started, ended, artifacts, posthoc)
    headline = pick_headline(artifacts, ctx.data_dir, lambda: _glob_headline(products, plots))
    return {
        "data_dir": ctx.data_dir,
        "products": products,
        "plots": plots[:200],
        "headline": headline,
        "catalog": plot_catalog(ctx, started, ended, posthoc),
    }


def image_list(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The task's image files as one ordered list (A4-04): the headline first, then product images, then plots.

    Built from a ``task_results`` view so the web Results tab and the MCP tools name the same files. Each row has an
    ``index`` (position in this list), ``name``, ``rel`` (relative to the data folder), ``path`` (absolute, what
    ``/api/artifacts`` serves), ``product`` (stage id or ``plot``) and ``headline``.
    """
    root = results.get("data_dir") or ""
    head = results.get("headline")
    rows: List[Dict[str, Any]] = []
    seen = set()

    def add(rel: str, product: str) -> None:
        if rel in seen or not rel.lower().endswith(IMAGE_EXT):
            return
        seen.add(rel)
        full = rel if os.path.isabs(rel) else os.path.join(root, rel)
        rows.append(
            {
                "index": len(rows),
                "name": os.path.basename(rel),
                "rel": rel,
                "path": full,
                "product": product,
                "headline": rel == head,
            }
        )

    if head:
        add(head, "headline")
    for prod in results.get("products", []):
        for f in prod.get("files", []):
            add(f["path"], prod["id"])
    for f in results.get("plots", []):
        add(f["path"], "plot")
    return rows
