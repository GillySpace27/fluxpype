"""Reader for the pipeline's numeric ``.dat`` text tables, for Plumber's data viewer.

The files are plain tables: whitespace or comma separated numbers, optionally one header line, no fixed schema.
``payload`` returns the columns as compact JSON plus per-column statistics; ``KINDS`` (below) names the kinds the
pipeline is known to write, so the viewer can pick a plot. A file that matches no kind is still shown as a table.
"""

import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np

MAX_ROWS = 20_000  # past this the payload keeps every k-th row and says so

# The kinds the pipeline writes, read off its writers (paths under fluxpype/science/) and checked against the real
# cr2150 files. Columns are (name, unit); unit "?" means the code does not say. `plot` is the default chart:
# scatter or lines, x and y column indices, `group` (lines are drawn per value of this column), `color` (scatter).
# A "(code)" unit is stated by a writer comment or formula; the rest is inferred and says so in `notes`.
_LON_LAT = [("line", ""), ("polarity", "+1/-1"), ("lat", "deg"), ("lon", "deg"), ("r", "R_sun")]
_PROFILE = {"type": "lines", "x": 4, "group": 0}
KINDS: List[Dict[str, Any]] = [
    {
        "id": "footpoints",
        "label": "Footpoints",
        "pattern": r"^floc_cr\d+_(?:r|rf)\d+_f\d+_\w+\.dat$",
        "writer": "get_hilbert_footpoints.pm:194 (wcols); points from fluxon_placement_hilbert.pm",
        "columns": [("x", "px"), ("y", "px"), ("sign", "+1/-1")],
        "plot": {"type": "scatter", "x": 0, "y": 1, "color": 2},
        "notes": ["Pixel indices on the (reduced) magnetogram, row order follows a space-filling curve."],
    },
    {
        "id": "open_lines",
        "label": "Open PFSS lines",
        "pattern": r"^floc_open_cr\d+_.*\.dat$",
        "writer": "pfss_funcs.py:688 (np.savetxt), column order in the comment at :893",
        "columns": _LON_LAT,
        "plot": {"type": "lines", "x": 3, "y": 2, "group": 0, "color": 1},
        "notes": ["Several rows per line, grouped by the first column."],
    },
    {
        "id": "closed_lines",
        "label": "Closed PFSS lines",
        "pattern": r"^floc_closed_cr\d+_.*\.dat$",
        "writer": "pfss_funcs.py:689 (np.savetxt), column order in the comment at :893",
        "columns": _LON_LAT,
        "plot": {"type": "lines", "x": 3, "y": 2, "group": 0, "color": 1},
        "notes": ["Only even-numbered closed lines are written."],
    },
    {
        "id": "bmag",
        "label": "Open-fluxon end field",
        "pattern": r"^cr\d+_f\d+_radial_bmag\.dat$",
        "writer": "map_fluxon_b.pm:114 (wcols); column names as unpacked in plot_bmag.py:57",
        "columns": [
            ("fluxon", ""),
            ("phi_begin", "rad"),
            ("theta_begin", "rad"),
            ("phi_end", "rad"),
            ("theta_end", "rad"),
            ("B_base", "?"),
            ("B_outer", "?"),
            ("area_begin", "?"),
            ("area_end", "?"),
        ],
        "plot": {"type": "scatter", "x": 3, "y": 4, "color": 6},
        "notes": ["One row per open fluxon. The units of B are not stated in the code."],
    },
    {
        "id": "bmag_all",
        "label": "Radial profiles of B",
        "pattern": r"^cr\d+_f\d+(?:_stiff)?_radial_bmag_all\.dat$",
        "writer": "map_fluxon_b_all.pm:31 (header), :92 (rows)",
        "columns": None,  # the file has a header line
        "plot": {**_PROFILE, "y": 9, "logy": True},
        "notes": [
            "b_mag and b_x, b_y, b_z are the first three rows of the library's bfield array, which hold vertex coordinates, "
            "not the field (b_x, b_y, b_z equal x, y, z on most rows; written at map_fluxon_b_all.pm:96). Only b_mag2 is the field magnitude, in unknown units."
        ],
    },
    {
        "id": "fr",
        "label": "Expansion factor profiles",
        "pattern": r"^cr\d+_f\d+_radial_fr\.dat$",
        "writer": "map_fluxon_fr.pm:79",
        "columns": [
            ("fluxon", ""),
            ("x", "?"),
            ("y", "?"),
            ("z", "?"),
            ("r", "m"),
            ("theta", "rad"),
            ("phi", "rad"),
            ("A", "m^2"),
            ("fr", ""),
        ],
        "plot": {**_PROFILE, "y": 8, "logy": True},
        "notes": ["Radius is in metres (r0 = 696e6 in the writer); x, y, z units are not stated."],
    },
    {
        "id": "wind",
        "label": "Wind solution per fluxon",
        "pattern": r"^cr\d+_f\d+_radial_wind_(?:parker|wsa|ghosts|tempest|cranmer|schonfeld)\.dat$",
        "writer": "map_fluxon_flow_parallel_master.pm:455 (wcols); names as read in plot_wind_map_square.py:398",
        "columns": [
            ("fluxon", ""),
            ("phi_base", "rad"),
            ("theta_base", "rad"),
            ("phi_end", "rad"),
            ("theta_end", "rad"),
            ("v_base", "km/s"),
            ("v_end", "km/s"),
            ("fr_base", ""),
            ("fr_end", ""),
        ],
        "plot": {"type": "scatter", "x": 3, "y": 4, "color": 6},
        "notes": ["One row per fluxon that converged. Some methods write NaN speeds."],
    },
    {
        "id": "velocity",
        "label": "Full velocity profiles",
        "pattern": r"^results_\w+_full_velocity\.dat$",
        "writer": "map_fluxon_flow_parallel_master.pm:116 (header), :155 (rows)",
        "columns": None,
        "plot": {"type": "lines", "x": 1, "y": 2, "group": 0},
        "notes": ["Comma separated despite the extension. Radius in R_sun, velocity in km/s."],
    },
]


def _is_number(t: str) -> bool:
    try:
        float(t)
        return True
    except ValueError:
        return False


def parse_text(text: str) -> Dict[str, Any]:
    """Columns of a numeric table. A first line that is not all numbers is the header; `#` lines are skipped; rows
    whose column count differs from the first data row are counted in `bad_rows` and dropped."""
    lines = [s for s in (ln.strip() for ln in text.splitlines()) if s and not s.startswith("#")]
    comma = bool(lines) and "," in lines[0]
    split = (lambda s: [t.strip() for t in s.split(",")]) if comma else str.split
    header: Optional[List[str]] = None
    if lines and not all(_is_number(t) for t in split(lines[0])):
        header, lines = split(lines[0]), lines[1:]
    rows: List[List[float]] = []
    bad = 0
    ncol = len(split(lines[0])) if lines else 0
    for ln in lines:
        t = split(ln)
        try:
            if len(t) != ncol:
                raise ValueError
            rows.append([float(x) for x in t])
        except ValueError:
            bad += 1
    a = np.array(rows, dtype=float).reshape(len(rows), ncol)
    names = header if header and len(header) == ncol else [f"col{i + 1}" for i in range(ncol)]
    cols = []
    for i in range(ncol):
        v = a[:, i]
        ok = np.isfinite(v)
        g = v[ok]
        cols.append(
            {
                "name": names[i],
                "min": float(g.min()) if g.size else None,
                "max": float(g.max()) if g.size else None,
                "mean": float(g.mean()) if g.size else None,
                "nonfinite": int((~ok).sum()),
                "integer": bool(g.size and np.all(g == np.round(g))),
                "sorted": bool(g.size > 1 and np.all(np.diff(g) >= 0)),
            }
        )
    return {"rows": len(rows), "bad_rows": bad, "header": header is not None, "columns": cols, "_a": a}


@lru_cache(maxsize=4)
def _parse_cached(path: str, mtime_ns: int, size: int) -> Dict[str, Any]:
    with open(path, "r", errors="replace") as f:
        return parse_text(f.read())


def parse_file(path: str) -> Dict[str, Any]:
    """Parsed file, cached by (path, mtime, size). Treat the result as read-only."""
    st = os.stat(path)
    return _parse_cached(path, st.st_mtime_ns, st.st_size)


def kind_of(path: str, ncol: int) -> Optional[Dict[str, Any]]:
    """The KINDS entry for a file name, if its column count agrees; else None (shown as a plain table)."""
    name = os.path.basename(path)
    for k in KINDS:
        if re.match(k["pattern"], name) and (k["columns"] is None or len(k["columns"]) == ncol):
            return {
                "id": k["id"],
                "label": k["label"],
                "writer": k["writer"],
                "columns": [{"name": n, "unit": u} for n, u in k["columns"]] if k["columns"] else None,
                "plot": k["plot"],
                "notes": k["notes"],
            }
    return None


def payload(path: str, max_rows: int = MAX_ROWS) -> Dict[str, Any]:
    """The parse as JSON data: `data` is one list per column (null for NaN or inf). Above `max_rows` it keeps every
    k-th row and says so in `decimated`."""
    p = parse_file(path)
    a = p["_a"]
    n = a.shape[0]
    step = -(-n // max_rows) if n > max_rows else 1
    a = a[::step]
    out = {k: v for k, v in p.items() if k != "_a"}
    out["data"] = [
        [x if x == x and abs(x) != float("inf") else None for x in a[:, i].tolist()] for i in range(a.shape[1])
    ]
    out["kind"] = kind_of(path, a.shape[1])
    out["decimated"] = {"step": step, "rows_kept": a.shape[0], "rows_total": n} if step > 1 else None
    return out
