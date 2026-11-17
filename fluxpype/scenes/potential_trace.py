"""A potential-field seed for a planar scene: field lines of the point sources themselves, traced in pure Python.

The arc seed (`planar._arc`) draws untwisted semicircles between hand-paired footpoints; that is a drawing, not a
field. This module computes the current-free field of the scene's own footpoints and traces one line from each
positive footpoint, so both the starting shapes and the connectivity are derived, not set by hand.

The field. Each footpoint is a point source of strength +-flux_per_line on the plane z = 0, and the field above
the plane is the sum of their monopole fields, B = sum q (r - r_i) / |r - r_i|^3 (the constant factor does not
change a field line). With a lid, FLUX's B2 plane acts as a mirror wall (no flux through it), so the field between
the two planes is the same sum plus image sources at z = 2 m lid_height, m = +-1 .. +-IMAGES. The series is
truncated; the sideways field converges as 1 / IMAGES^2 and the truncation leaves a small normal field at the lid.

What this is and is not:
- It is the potential field of point sources, the same idealisation FLUX's flux concentrations make. It is not a
  potential field of a smooth magnetogram.
- One fluxon stands for all the flux of its concentration. A source sends flux out in every upward direction, so
  one line has to be picked: this module starts it 45 degrees up from the plane, leaning along the sideways field
  the other sources make there (straight up when that field vanishes). That is a stated choice, not a theorem.
- Two traced lines can land on the same negative footpoint. A fluxon world needs one line per concentration, so
  the landings are then matched one to one with the smallest total distance moved, and `trace` reports how many
  lines were moved. A moved line is no longer a field line of the potential field.
- A relaxed FLUX run started from this seed is still a FLUX equilibrium with FLUX's lid behaviour (energies
  depend on lid height). The seed makes the reference state a computed one; it does not remove that dependence.
"""

import math
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

IMAGES = 6  # image sources each side of the slab when there is a lid
START_ELEVATION = math.pi / 4  # the starting direction of a line above the plane
MAX_STEPS = 4000

Vec = Tuple[float, float, float]


def field(r: Sequence[float], sources: Sequence[Tuple[float, float, float]], lid: Optional[float] = None) -> Vec:
    """B at r from point sources (x, y, q) on z = 0, with the mirror images of a lid at z = `lid` if given."""
    x, y, z = r
    bx = by = bz = 0.0
    zs = [0.0] if lid is None else [2.0 * m * lid for m in range(-IMAGES, IMAGES + 1)]
    for sx, sy, q in sources:
        dx, dy = x - sx, y - sy
        rho2 = dx * dx + dy * dy
        for z0 in zs:
            dz = z - z0
            d2 = rho2 + dz * dz
            if d2 < 1e-24:
                continue
            w = q / (d2 * math.sqrt(d2))
            bx += w * dx
            by += w * dy
            bz += w * dz
    return bx, by, bz


def _unit(v: Vec) -> Optional[Vec]:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return None if n < 1e-300 else (v[0] / n, v[1] / n, v[2] / n)


def _trace_one(start: Tuple[float, float], sources, lid, own: int, scale: float, reach: float) -> Dict:
    """Follow B from just above positive source `own` until it comes down to the plane, or gives up."""
    others = [s for i, s in enumerate(sources) if i != own]
    eps = 0.02 * scale
    ext = field((start[0], start[1], 0.0), others, lid)
    h = math.hypot(ext[0], ext[1])
    if h > 1e-12:
        c, s = math.cos(START_ELEVATION), math.sin(START_ELEVATION)
        d0 = (c * ext[0] / h, c * ext[1] / h, s)
    else:
        d0 = (0.0, 0.0, 1.0)
    p = (start[0] + eps * d0[0], start[1] + eps * d0[1], eps * d0[2])
    pts = [(start[0], start[1], 0.0), p]
    negs = [(sx, sy) for sx, sy, q in sources if q < 0]
    landed = False
    for _ in range(MAX_STEPS):
        near = min(math.hypot(p[0] - sx, p[1] - sy) ** 2 + p[2] ** 2 for sx, sy, _q in sources) ** 0.5
        ds = max(min(0.05 * scale, 0.25 * near), 1e-4 * scale)

        def f(q):
            return _unit(field(q, sources, lid))

        k1 = f(p)
        if k1 is None:
            break
        k2 = f((p[0] + 0.5 * ds * k1[0], p[1] + 0.5 * ds * k1[1], p[2] + 0.5 * ds * k1[2])) or k1
        k3 = f((p[0] + 0.5 * ds * k2[0], p[1] + 0.5 * ds * k2[1], p[2] + 0.5 * ds * k2[2])) or k2
        k4 = f((p[0] + ds * k3[0], p[1] + ds * k3[1], p[2] + ds * k3[2])) or k3
        q = tuple(p[i] + ds / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(3))
        if q[2] <= 0.0:  # came down: stop on the plane
            t = p[2] / (p[2] - q[2]) if p[2] != q[2] else 0.0
            pts.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]), 0.0))
            landed = True
            break
        pts.append(q)
        p = q
        if min(math.hypot(p[0] - sx, p[1] - sy) ** 2 + p[2] ** 2 for sx, sy in negs) < (0.5 * eps) ** 2:
            landed = True  # close enough to a negative source that the rest is a straight drop into it
            break
        if math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) > reach:
            break
    return {"points": pts, "landed": landed}


def _resample(pts: List[Vec], n: int) -> List[Vec]:
    """n points evenly spaced in arc length strictly inside the polyline."""
    seg = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    total = sum(seg)
    out, i, acc = [], 0, 0.0
    for k in range(1, n + 1):
        want = total * k / (n + 1)
        while i < len(seg) - 1 and acc + seg[i] < want:
            acc += seg[i]
            i += 1
        t = (want - acc) / seg[i] if seg[i] > 0 else 0.0
        a, b = pts[i], pts[i + 1]
        out.append(tuple(a[j] + t * (b[j] - a[j]) for j in range(3)))
    return out


def _assign(cost: List[List[float]]) -> List[int]:
    """One-to-one matching of rows to columns with the smallest total cost (Hungarian method, O(n^3))."""
    n = len(cost)
    inf = float("inf")
    u, v, match, way = [0.0] * (n + 1), [0.0] * (n + 1), [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        match[0] = i
        j0 = 0
        minv, used = [inf] * (n + 1), [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = match[j0], inf, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[match[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if match[j0] == 0:
                break
        while j0:
            match[j0] = match[way[j0]]
            j0 = way[j0]
    out = [0] * n
    for j in range(1, n + 1):
        out[match[j] - 1] = j - 1
    return out


@lru_cache(maxsize=64)
def _trace(pos: tuple, neg: tuple, flux: float, lid: Optional[float], n_vertices: int) -> Dict:
    sources = [(x, y, flux) for x, y in pos] + [(x, y, -flux) for x, y in neg]
    xs = [s[0] for s in sources]
    ys = [s[1] for s in sources]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    near = min(math.dist(a, b) for a in pos for b in neg)
    scale = (
        near
        if len(sources) == 2
        else min(near, min(math.dist(pos[i], pos[j]) for i in range(len(pos)) for j in range(i)))
    )
    reach = 30.0 * max(span, near)
    raw = [_trace_one(p, sources, lid, i, scale, reach) for i, p in enumerate(pos)]
    # which negative footpoint each line came down nearest to; if that is not one to one, the cheapest matching
    ends = [r["points"][-1] for r in raw]
    dist = [[math.hypot(e[0] - q[0], e[1] - q[1]) for q in neg] for e in ends]
    traced = [min(range(len(neg)), key=lambda k: d[k]) for d in dist]
    pairing = traced if sorted(traced) == list(range(len(neg))) else _assign(dist)
    lines = []
    for j, r in enumerate(raw):
        inner = _resample(r["points"], n_vertices)
        end, target = r["points"][-1], neg[pairing[j]]
        shift = (target[0] - end[0], target[1] - end[1])
        fixed = []
        for i, (x, y, z) in enumerate(inner, start=1):  # slide the far end onto its concentration, the near end stays
            t = i / (n_vertices + 1)
            fixed.append((x + t * shift[0], y + t * shift[1], max(z, 1e-6 * scale)))
        lines.append(fixed)
    return {
        "pairing": pairing,
        "traced": traced,
        "moved": sum(1 for j in range(len(pos)) if traced[j] != pairing[j]),
        "not_landed": sum(1 for r in raw if not r["landed"]),
        "top_z": max(p[2] for ln in lines for p in ln),
        "lines": lines,
    }


def trace(spec) -> Dict:
    """The potential seed of a spec: `pairing`, the interior `lines`, and how honest the result is.

    `moved` counts lines whose traced landing was already taken and that were matched to another footpoint;
    `not_landed` counts lines that did not come back to the plane within reach (their far end is then forced onto a
    footpoint, which is not a field line). Both are 0 for a clean seed.
    """
    pos, neg = spec.footpoints()
    return _trace(tuple(pos), tuple(neg), float(spec.flux_per_line), spec.lid_height, int(spec.vertices_per_line))
