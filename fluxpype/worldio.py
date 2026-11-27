"""One parser for the `.flux` world text, shared by the Python science readers (audit A2-06).

Two dialects of the LINE record exist and both are accepted (io.c:257-259 reads the 12, 6 and 4 parameter forms):

    input form (make_world_sphere.pm, scenes/planar.py):  LINE <id> <fc0> <fc1> <flux>
    core form (Flux write_world):  LINE <id> <vstart> <vend> <fc0> <fc1> <flux> <x0> <y0> <z0> <x1> <y1> <z1> ...

In core form the end points are on the LINE record. In input form there are none, so an end point is taken from
its flux concentration (NEW record); an open end (fc -1 or -2, which has no concentration) has no end point and is
left out of the line. Reading stops at the first VNEIGHBOR record, as the older readers did. GLOBAL (and other
header) records are kept verbatim in `World.header`.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

Point = Tuple[float, float, float]


@dataclass
class Line:
    id: int
    fc0: int
    fc1: int
    flux: float
    start: Optional[Point]
    end: Optional[Point]
    x: List[float] = field(default_factory=list)  # interior vertices, in file order
    y: List[float] = field(default_factory=list)
    z: List[float] = field(default_factory=list)


@dataclass
class World:
    concentrations: List[Tuple[int, float, float, float, float]] = field(default_factory=list)  # id, x, y, z, flux
    lines: List[Line] = field(default_factory=list)  # fluxons only (id > 0), in file order
    header: List[str] = field(default_factory=list)  # GLOBAL, FRAME and any other non-geometry records
    dialect: str = "input"  # "input" or "core", from the first LINE record


def read(path: Union[str, "os.PathLike"]) -> World:
    """Parse the world file at `path`. A missing file raises FileNotFoundError; for text use `parse_text`."""
    with open(path, "r") as fh:
        return parse_text(fh.read())


def parse_text(text: str) -> World:
    w = World()
    by_id = {}
    pending = []  # (line, id of fc0, id of fc1) for input-form lines, resolved once all NEW records are in
    conc = {}
    verts = {}  # line id -> ([x], [y], [z])
    seen_line = False
    for raw in text.splitlines():
        t = raw.split()
        if not t or t[0].startswith("#"):
            continue
        k = t[0]
        if k == "NEW":
            c = (int(t[1]), float(t[2]), float(t[3]), float(t[4]), float(t[5]))
            w.concentrations.append(c)
            conc[c[0]] = c
        elif k == "LINE":
            lid = int(t[1])
            if len(t) >= 13:  # core form, 12 parameters (what Flux writes)
                if not seen_line:
                    w.dialect = "core"
                ln = Line(
                    lid,
                    int(t[4]),
                    int(t[5]),
                    float(t[6]),
                    (float(t[7]), float(t[8]), float(t[9])),
                    (float(t[10]), float(t[11]), float(t[12])),
                )
            elif len(t) >= 7:  # core form, 6 parameters (io.c:258): no end points on the record
                if not seen_line:
                    w.dialect = "core"
                ln = Line(lid, int(t[4]), int(t[5]), float(t[6]), None, None)
                pending.append(ln)
            elif len(t) == 5:  # input form, 4 parameters (io.c:259)
                ln = Line(lid, int(t[2]), int(t[3]), float(t[4]), None, None)
                pending.append(ln)
            else:
                raise ValueError(f"unrecognised LINE record (expected 5, or 7 or more tokens): {raw.strip()!r}")
            seen_line = True
            if lid > 0:
                w.lines.append(ln)
                by_id.setdefault(lid, ln)
        elif k == "VERTEX":
            v = verts.setdefault(int(t[1]), ([], [], []))
            v[0].append(float(t[4]))
            v[1].append(float(t[5]))
            v[2].append(float(t[6]))
        elif "VNEIGHBOR" in k:
            break
        else:
            w.header.append(raw.strip())
    for ln in pending:
        c0, c1 = conc.get(ln.fc0), conc.get(ln.fc1)
        ln.start = c0[1:4] if c0 else None
        ln.end = c1[1:4] if c1 else None
    for ln in w.lines:
        v = verts.get(ln.id)
        if v:
            ln.x, ln.y, ln.z = list(v[0]), list(v[1]), list(v[2])
    return w


def line_arrays(ln: Line):
    """(x, y, z) lists for a line: start point, interior vertices, end point (a missing end point is skipped)."""
    out = []
    for i, inner in enumerate((ln.x, ln.y, ln.z)):
        out.append(([ln.start[i]] if ln.start else []) + list(inner) + ([ln.end[i]] if ln.end else []))
    return out
