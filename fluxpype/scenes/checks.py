"""Construction checks on a compiled planar scene, and a tiny reader for the `.flux` text it writes.

These are checks on how the world was built (labels, flux balance, geometry), not on the physics. They do not
tell you the world will relax well or that its energy means anything; that takes a FLUX run.
"""

import math
from typing import Dict, List, Tuple


def read(text: str) -> Dict:
    """Parse the subset of the format that `compile_flux` writes: concentrations, lines and vertices."""
    conc: Dict[int, Tuple[float, float, float, float]] = {}
    lines: Dict[int, Dict] = {}
    header: List[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        w = s.split()
        if w[0] == "NEW":
            conc[int(w[1])] = tuple(float(v) for v in w[2:6])
        elif w[0] == "LINE":
            lines[int(w[1])] = {"c0": int(w[2]), "c1": int(w[3]), "flux": float(w[4]), "vertices": {}}
        elif w[0] == "VERTEX":
            lines[int(w[1])]["vertices"][int(w[3])] = (int(w[2]), float(w[4]), float(w[5]), float(w[6]))
        else:
            header.append(s)
    return {"concentrations": conc, "lines": lines, "header": header}


def line_points(world: Dict, fl: int) -> List[Tuple[float, float, float]]:
    """Footpoint, interior vertices in order, footpoint."""
    ln = world["lines"][fl]
    c0, c1 = world["concentrations"][ln["c0"]], world["concentrations"][ln["c1"]]
    inner = [ln["vertices"][i][1:] for i in sorted(ln["vertices"])]
    return [c0[:3]] + inner + [c1[:3]]


def line_separation(text: str) -> float:
    """Smallest distance between points of two different lines, compared vertex to vertex.

    A rough guide to whether the starting lines nearly touch: it samples vertices, so two crossing lines can read
    larger than their true closest approach. Zero or very small means the start needs a second look.
    """
    world = read(text)
    pts = {fl: line_points(world, fl) for fl in world["lines"]}
    best = math.inf
    ids = sorted(pts)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            for p in pts[ids[a]][1:-1]:
                for q in pts[ids[b]][1:-1]:
                    best = min(best, math.dist(p, q))
    return best


def summarize(text: str) -> Dict:
    """Counts and the checks a person would otherwise do by eye. `problems` is empty when all pass."""
    world = read(text)
    conc, lines = world["concentrations"], world["lines"]
    problems: List[str] = []
    net = sum(c[3] for c in conc.values())
    if abs(net) > 1e-9:
        problems.append(f"net flux of the concentrations is {net:g}, not 0")
    used = [c for ln in lines.values() for c in (ln["c0"], ln["c1"])]
    if sorted(used) != sorted(conc):
        problems.append("a concentration is used by no line, or by more than one")
    labels = list(lines) + [v[0] for ln in lines.values() for v in ln["vertices"].values()]
    if len(set(labels)) != len(labels):
        problems.append("a fluxon or vertex label is repeated")
    for fl, ln in lines.items():
        c0, c1 = conc[ln["c0"]], conc[ln["c1"]]
        if c0[3] * c1[3] >= 0:
            problems.append(f"line {fl} joins two concentrations of the same sign")
        if abs(abs(c0[3]) - ln["flux"]) > 1e-9:
            problems.append(f"line {fl} flux does not match its concentrations")
    heights = [v[3] for ln in lines.values() for v in ln["vertices"].values()]
    return {
        "lines": len(lines),
        "vertices": sum(len(ln["vertices"]) for ln in lines.values()),
        "net_flux": net,
        "max_height": max(heights) if heights else 0.0,
        "min_line_separation": line_separation(text) if len(lines) > 1 else None,
        "photosphere": next((h for h in world["header"] if h.startswith("GLOBAL PHOTOSPHERE")), None),
        "lid": next((h for h in world["header"] if h.startswith("GLOBAL B2")), None),
        "problems": problems,
    }


def comparable(a: str, b: str) -> List[str]:
    """Why two compiled worlds are not a fair pair for an energy comparison (empty list when they are).

    Fair here means: the same footpoint positions and fluxes, the same photosphere and lid, and the same number of
    lines and vertices. Only the connectivity (which footpoint joins which) may differ.
    """
    wa, wb = read(a), read(b)
    why: List[str] = []
    fa = sorted(wa["concentrations"].values())
    fb = sorted(wb["concentrations"].values())
    if fa != fb:
        why.append("footpoint positions or fluxes differ")
    for key in ("GLOBAL PHOTOSPHERE", "GLOBAL B2"):
        ha = [h for h in wa["header"] if h.startswith(key)]
        hb = [h for h in wb["header"] if h.startswith(key)]
        if ha != hb:
            why.append(f"{key.lower()} differs")
    if len(wa["lines"]) != len(wb["lines"]):
        why.append("number of lines differs")
    na = sorted(len(l["vertices"]) for l in wa["lines"].values())
    nb = sorted(len(l["vertices"]) for l in wb["lines"].values())
    if na != nb:
        why.append("vertices per line differ")
    return why
