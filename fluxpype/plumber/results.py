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

from .cache import classify
from .monitor import ProbeContext

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
PLOT_EXT = (".png", ".jpg", ".jpeg", ".pdf", ".svg")
_NFLUX = re.compile(r"_f(\d+)(?:_|\.|$)")
SLACK_S = 5.0  # file clocks and the task's start time are compared with this much tolerance


def plot_stage(path: str) -> Optional[str]:
    """The pipeline stage that writes a plot, judged from where it sits (a guess by folder, not a record)."""
    p = "/" + path.replace("\\", "/")
    if "fluxlight" in os.path.basename(p):
        return "fluxlight"
    if "/imgs/footpoints/" in p:
        return "fieldmap"
    if "/imgs/world/" in p:
        return "world_plots"
    return "wind_plots" if "/imgs/" in p else None


def for_this_nflux(name: str, nflux: str) -> bool:
    """False only when the file name names a different fluxon count (``_f<N>_``)."""
    m = _NFLUX.search(name)
    return m is None or m.group(1) == str(nflux)


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


def _file(path: str, root: str, started, ended) -> Optional[Dict[str, Any]]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    if not os.path.isfile(path):
        return None
    rel = _rel(path, root)
    info = classify(rel)
    return {
        "path": rel,
        "name": os.path.basename(path),
        "size": st.st_size,
        "mtime": st.st_mtime,
        "made_here": made_here(st.st_mtime, started, ended),
        "origin": origin(st.st_mtime, started, ended),
        **{k: v for k, v in info.items() if k in ("kind", "cr", "batch", "n", "inst", "reduce", "sel", "steps")},
    }


def _expand(patterns: List[str]) -> List[str]:
    out: List[str] = []
    for pat in patterns:
        out += glob.glob(pat, recursive=True)
    return sorted(set(out))


def task_results(ctx: Optional[ProbeContext], started: Optional[float], ended: Optional[float]) -> Dict[str, Any]:
    """Products by stage plus the other plots for one task, with paths relative to ctx.data_dir."""
    if ctx is None:
        return {"data_dir": None, "products": [], "plots": [], "headline": None}
    root = ctx.data_dir
    pats = ctx.patterns()
    seen = set()
    products = []
    for sid, label, key, limit in PRODUCTS:
        hits = [p for p in _expand(pats[key]) if for_this_nflux(os.path.basename(p), ctx.nflux)]
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
            if not for_this_nflux(os.path.basename(p), ctx.nflux):
                continue
            if by_name and f"cr{ctx.cr}" not in os.path.basename(p):
                continue
            f = _file(p, root, started, ended)
            if f:
                plots.append({**f, "stage": plot_stage(f["path"])})
    plots.sort(key=lambda f: f["mtime"], reverse=True)

    headline = None
    light = next((p for p in products if p["id"] == "fluxlight"), None)
    images = [f for f in (light["files"] if light else []) if f["name"].lower().endswith((".png", ".jpg", ".jpeg"))]
    images += [f for f in plots if f["name"].lower().endswith((".png", ".jpg", ".jpeg"))]
    if images:
        headline = images[0]["path"]
    return {"data_dir": root, "products": products, "plots": plots[:200], "headline": headline}
