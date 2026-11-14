"""Compile a PlanarArcadeSpec to the text `.flux` world format.

The format follows the menagerie worlds in fluxon-mhd (`pdl/menagerie/potential.flux`): a GLOBAL header, then per
line two `NEW <conc> x y z flux` flux concentrations at the footpoints, a `LINE <fluxon> <conc0> <conc1> <flux>`,
and its interior `VERTEX <fluxon> <vertex> <index> x y z` rows. Fluxon and vertex labels share one running counter
as the menagerie generators do. These files have been written to match that format but loading one in FLUX has to
be confirmed on a machine that has it (the unit tests only check the text against a small reader).
"""

import math
from typing import List

from .spec import PlanarArcadeSpec


def _arc(p, q, n: int, height: float) -> List[tuple]:
    """n interior points of an arc from p to q (x, y, 0) rising to `height` over the chord midpoint."""
    pts = []
    for i in range(1, n + 1):
        t = i / (n + 1)
        pts.append((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t, height * math.sin(math.pi * t)))
    return pts


def arc_height(spec: PlanarArcadeSpec, p, q) -> float:
    chord = math.hypot(q[0] - p[0], q[1] - p[1])
    return min(spec.height_ratio * chord / 2, spec.max_height)


def compile_flux(spec: PlanarArcadeSpec) -> str:
    spec = spec.validate()
    pos, neg = spec.footpoints()
    pair = spec.pairs()
    n = spec.vertices_per_line
    out = [
        f"# scene {spec.name}  hash {spec.content_hash()}  (written by fluxpype.scenes, schema {spec.schema})",
        "# connectivity is hand-set: pairing = " + ",".join(str(k) for k in pair),
        "FRAME 0",
        "GLOBAL FORCES f_pressure_equi f_curvature f_vertex",
        f"GLOBAL PHOTOSPHERE 0 0 {spec.photosphere_z:g} 0 0 1",
    ]
    if spec.lid_height is not None:
        out.append(f"GLOBAL B2 PLANE 0 0 {spec.lid_height:g} 0 0 -1")
    out.append("")
    label = 2  # running label for concentrations (own namespace) and fluxon + vertices (shared counter)
    conc = 2
    flux = f"{spec.flux_per_line:g}"
    for j in range(spec.n_lines):
        p, q = pos[j], neg[pair[j]]
        c0, c1 = conc, conc + 1
        conc += 2
        out.append(f"NEW {c0} {p[0]:.6g} {p[1]:.6g} 0 {flux}")
        out.append(f"NEW {c1} {q[0]:.6g} {q[1]:.6g} 0 -{flux}")
        out.append(f"LINE {label} {c0} {c1} {flux}")
        for i, (x, y, z) in enumerate(_arc(p, q, n, arc_height(spec, p, q)), start=1):
            out.append(f"VERTEX {label} {label + i} {i} {x:.6g} {y:.6g} {z:.6g}")
        label += n + 1
        out.append("")
    return "\n".join(out)
