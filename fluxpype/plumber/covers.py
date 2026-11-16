"""Cover pictures for Plumber's lists: one image that stands for a task, a job, a preset or a scene.

A cover is chosen from files already on disk, never computed for the purpose (except that a FITS
magnetogram or a ``.flux`` world is drawn by the same cached renderers the file viewer uses).
Preference, best first:

1. a FLUXlight image (the showcase product);
2. a world or field-line plot under ``batches/<batch>/imgs/`` for this rotation and fluxon count;
3. any other PNG in the rotation's data folder;
4. the relaxed world, else the initial world (``.flux``, drawn as a thumbnail);
5. the magnetogram (FITS, drawn as an image).

Lookups glob the data folder, so results are kept for a short while per context.
"""

from __future__ import annotations

import glob
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from .monitor import ProbeContext
from .results import for_this_nflux

IMG = (".png", ".jpg", ".jpeg")
TTL_S = 20.0
_memo: Dict[Tuple, Tuple[float, Optional[Tuple[str, str]]]] = {}


def _newest(paths: List[str]) -> Optional[str]:
    files = [p for p in paths if os.path.isfile(p)]
    return max(files, key=os.path.getmtime) if files else None


def _glob(patterns: List[str], limit: int = 2000) -> List[str]:
    out: List[str] = []
    for pat in patterns:
        out += glob.glob(pat, recursive=True)[:limit]
    return out


def _rank_plot(path: str) -> int:
    """Lower is better among plots: world and field-line views read best at thumbnail size."""
    p = path.replace("\\", "/").lower()
    if "/imgs/world/" in p or "world" in os.path.basename(p):
        return 0
    if "/imgs/footpoints/" in p:
        return 1
    return 2


def find(ctx: Optional[ProbeContext]) -> Optional[Tuple[str, str]]:
    """(absolute path, kind) of the best cover for one task context; kind is image, flux or fits."""
    if ctx is None:
        return None
    key = (ctx.data_dir, ctx.batch, str(ctx.cr), str(ctx.nflux), ctx.inst, ctx.reduce, ctx.adapt_select)
    hit = _memo.get(key)
    if hit and time.time() - hit[0] < TTL_S:
        return hit[1]
    out = _find(ctx)
    if len(_memo) > 2000:
        _memo.clear()
    _memo[key] = (time.time(), out)
    return out


def _find(ctx: ProbeContext) -> Optional[Tuple[str, str]]:
    pats = ctx.patterns()
    n = str(ctx.nflux)
    light = [p for p in _glob(pats["fluxlight"]) if p.lower().endswith(IMG) and for_this_nflux(os.path.basename(p), n)]
    if light:
        return _newest(light), "image"
    batch = os.path.join(ctx.data_dir, "batches", ctx.batch)
    tag = f"cr{ctx.cr}"
    plots = [
        p
        for p in _glob([os.path.join(batch, "imgs", "**", "*")])
        if p.lower().endswith(IMG) and tag in os.path.basename(p).lower() and for_this_nflux(os.path.basename(p), n)
    ]
    if plots:
        best = min(_rank_plot(p) for p in plots)
        return _newest([p for p in plots if _rank_plot(p) == best]), "image"
    beside = [
        p
        for p in _glob([os.path.join(batch, "data", tag, "**", "*")])
        if p.lower().endswith(IMG) and for_this_nflux(os.path.basename(p), n)
    ]
    if beside:
        return _newest(beside), "image"
    for key in ("relax", "world"):
        world = _newest([p for p in _glob(pats[key]) if for_this_nflux(os.path.basename(p), n)])
        if world:
            return world, "flux"
    mag = _newest(_glob(pats["magnetogram"]))
    if mag:
        return mag, "fits"
    return None


def for_scene_task(run_dir: Optional[str], label: str, data_dir: str) -> Optional[Tuple[str, str]]:
    """A scene relaxation: its relaxed world if written, else the first scene's compiled world."""
    if run_dir:
        relaxed = _newest(glob.glob(os.path.join(run_dir, "relaxed-*.flux")))
        if relaxed:
            return relaxed, "flux"
    first = (label or "").split(" vs ")[0].strip()
    if first:
        from ..scenes import store as scene_store

        try:
            world = os.path.join(scene_store.folder(data_dir), f"{scene_store.check_id(first)}.flux")
        except Exception:  # noqa: BLE001 - a label that is not a scene id has no cover
            return None
        if os.path.isfile(world):
            return world, "flux"
    return None


def task_order(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Which of a job's tasks to try first for its cover: finished OK, newest first; then the rest."""
    rank = {"ok": 0, "running": 1}
    return sorted(tasks, key=lambda t: (rank.get(t["status"], 2), -(t.get("ended") or t.get("started") or 0)))
