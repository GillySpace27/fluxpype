"""Parser for FLUX world files (``.flux``), for Plumber's 3-D viewer.

Row kinds, read off the FLUX C reader/writer (lib/io.c: ``footpoint_action``, ``fprint_fc_line``,
``fprint_fl_vertices``, ``fprint_v_nbors``) and checked against real files:

    FRAME <n>
    GLOBAL <KEY> <args...>                      (BOUNDARY SPHERE ox oy oz radius 0 0 is the photosphere)
    NEW <id> <x> <y> <z> <flux> <bound> <r>     a flux concentration (flux is +1 or -1 here)
    LINE <id> <v_start> <v_end> <fc0> <fc1> <flux> <sx> <sy> <sz> <ex> <ey> <ez>      a fluxon
    <tab>VERTEX <line id> <vertex id> <pos> <x> <y> <z>   interior vertex of a fluxon, pos = 1, 2, ...
    VNEIGHBOR <vertex id> <vertex id>           neighbour graph of the relaxation; not drawn, only counted

LINE carries the two end vertices; VERTEX rows are the interior ones only. So the vertex count FLUX
reports is VERTEX rows + 2 per LINE, and the fluxon count is the LINE rows (including the two
world-internal pseudo-fluxons with ids -9 and -10, which are not drawn). FLUX also creates 8 built-in
concentrations itself (lib/data.c: fc_im10/im11/im20/im21/ob/oe/pb/pe), which a file never lists.
"""

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

BUILTIN_CONCENTRATIONS = 8
DECIMALS = 5
MAX_VERTICES = 400_000  # past this the payload keeps every k-th fluxon


def _g(x: str) -> float:
    v = float(x)
    if v - v:  # NaN or inf would break the JSON
        raise ValueError(x)
    return v


def _f(x: str) -> float:
    return round(_g(x), DECIMALS)


def _plane(s: List[str]) -> Optional[Dict[str, Any]]:
    """A planar boundary from a GLOBAL row, as {role, origin, normal}: GLOBAL PHOTOSPHERE <point> <normal> (the
    convenience form of a boundary plane), GLOBAL BOUNDARY PLANE and GLOBAL B2 PLANE (a second plane, used as a lid)."""
    key = s[1].upper()
    if key == "PHOTOSPHERE" and len(s) >= 8:
        role, vals = "photosphere", s[2:8]
    elif key in ("BOUNDARY", "B2") and len(s) >= 9 and s[2].upper() == "PLANE":
        role, vals = ("boundary" if key == "BOUNDARY" else "b2"), s[3:9]
    else:
        return None
    v = [_g(x) for x in vals]
    if not any(v[3:]):
        return None
    return {"role": role, "origin": v[:3], "normal": v[3:]}


def parse_text(text: str) -> Dict[str, Any]:
    conc: Dict[str, list] = {"id": [], "xyz": [], "flux": []}
    cpos: Dict[int, List[float]] = {}
    frame = None
    boundary = None
    planes: List[Dict[str, Any]] = []
    globs: List[str] = []
    lines = []  # (id, flux, fc0, fc1, start xyz or None, end xyz or None)
    interior: Dict[int, list] = {}  # line id -> [(pos, x, y, z)]
    other: Dict[str, int] = {}
    n_vertex = n_vnb = bad = 0
    for ln in text.splitlines():
        s = ln.split()
        if not s or s[0][0] == "#":
            continue
        k = s[0].upper()
        try:
            if k == "VERTEX":
                p = int(s[3])
                interior.setdefault(int(s[1]), []).append((p if p >= 0 else 1 << 60, _f(s[4]), _f(s[5]), _f(s[6])))
                n_vertex += 1
            elif k == "VNEIGHBOR":
                n_vnb += 1
            elif k == "NEW":
                i, xyz = int(s[1]), [_f(v) for v in s[2:5]]
                conc["id"].append(i)
                conc["xyz"] += xyz
                conc["flux"].append(_g(s[5]))
                cpos[i] = xyz
            elif k == "LINE":
                if len(s) >= 13:
                    lines.append(
                        (int(s[1]), _g(s[6]), int(s[4]), int(s[5]), [_f(v) for v in s[7:10]], [_f(v) for v in s[10:13]])
                    )
                elif len(s) >= 7:  # 6-parameter form: ends sit on the concentrations
                    lines.append((int(s[1]), _g(s[6]), int(s[4]), int(s[5]), None, None))
                else:  # 4-parameter form: LINE id fc0 fc1 flux
                    lines.append((int(s[1]), _g(s[4]), int(s[2]), int(s[3]), None, None))
            elif k == "FRAME":
                frame = int(s[1])
            elif k == "GLOBAL":
                globs.append(" ".join(s[1:]))
                if s[1].upper() == "BOUNDARY":
                    boundary = {"type": s[2].upper(), "args": [_g(v) for v in s[3:]]}
                plane = _plane(s)
                if plane:
                    planes.append(plane)
            else:
                other[k] = other.get(k, 0) + 1
        except (ValueError, IndexError):
            bad += 1
    fl: Dict[str, list] = {"id": [], "flux": [], "fc0": [], "fc1": [], "offsets": [0], "xyz": []}
    special = 0
    for lid, flux, fc0, fc1, a, b in lines:
        if -10 <= lid <= 0:  # FLUX's own pseudo-fluxons: no geometry
            special += 1
            continue
        a = a or cpos.get(fc0)
        b = b or cpos.get(fc1)
        if a is None or b is None:
            bad += 1
            continue
        fl["id"].append(lid)
        fl["flux"].append(flux)
        fl["fc0"].append(fc0)
        fl["fc1"].append(fc1)
        fl["xyz"] += a
        for _, x, y, z in sorted(interior.get(lid, ())):
            fl["xyz"] += (x, y, z)
        fl["xyz"] += b
        fl["offsets"].append(len(fl["xyz"]) // 3)
    return {
        "frame": frame,
        "boundary": boundary,
        "planes": planes,
        "globals": globs,
        "counts": {
            # what the Flux library reports on reading the file
            "vertices": n_vertex + 2 * len(lines),
            "fluxons": len(lines),
            "concentrations": len(conc["id"]) + BUILTIN_CONCENTRATIONS,
            # extra detail
            "vertex_rows": n_vertex,
            "concentration_rows": len(conc["id"]),
            "pseudo_fluxons": special,
            "vneighbor_rows": n_vnb,
            "unknown_rows": other,
            "bad_rows": bad,
        },
        "concentrations": conc,
        "fluxons": fl,
    }


@lru_cache(maxsize=4)
def _parse_cached(path: str, mtime_ns: int, size: int) -> Dict[str, Any]:
    with open(path, "r", errors="replace") as f:
        return parse_text(f.read())


def parse_file(path: str) -> Dict[str, Any]:
    """Parsed file, cached by (path, mtime, size). Treat the result as read-only."""
    st = os.stat(path)
    return _parse_cached(path, st.st_mtime_ns, st.st_size)


def payload(path: str, max_vertices: int = MAX_VERTICES) -> Dict[str, Any]:
    """The parse as compact JSON data; above ``max_vertices`` keeps every k-th fluxon and says so in ``decimated``."""
    w = parse_file(path)
    fl = w["fluxons"]
    n = len(fl["id"])
    nv = len(fl["xyz"]) // 3
    out = {**w, "decimated": None}
    if nv > max_vertices and n:
        step = -(-nv // max_vertices)
        keep = range(0, n, step)
        off, xyz = [0], []
        for i in keep:
            xyz += fl["xyz"][3 * fl["offsets"][i] : 3 * fl["offsets"][i + 1]]
            off.append(len(xyz) // 3)
        out["fluxons"] = {
            "offsets": off,
            "xyz": xyz,
            **{k: [fl[k][i] for i in keep] for k in ("id", "flux", "fc0", "fc1")},
        }
        out["decimated"] = {"step": step, "fluxons_kept": len(keep), "fluxons_total": n, "vertices_total": nv}
    return out
