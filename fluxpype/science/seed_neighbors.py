"""Seed a FLUX world file with KD-tree neighbour candidates, so relaxation can skip FLUX's O(V^2) start-up search.

On its first step ``simple_relaxer`` calls ``update_neighbors(-1)``, a global search that compares every vertex with
every other vertex (lib/model.c, ``fast_world_gather_neighbors``). Measured in the cloud on 2026-11-24 (unmodified
fluxon-mhd, 4 cores): 7 s, 68 s and 861 s at 8k, 25k and 79k vertices. FLUX's world reader already accepts
``VNEIGHBOR <vertex> <vertex>`` rows (lib/io.c), and a vertex that has neighbours is then updated locally, so this
module appends, for every interior vertex, its ``k`` nearest vertices on *other* fluxons. The relaxation then does one
local ``update_neighbors(0)`` instead of the global one (see relax_pfss_world.pm). At 1,000 seeds that took the
start-up from 68 s to 8 s with the same stiffness after two cycles (18.29 vs 18.19).

The seeds are only candidates: FLUX's local update still builds the real hull neighbours from them and from their
neighbours. Vertices the file does not list (the two ends of each fluxon, which FLUX creates from the LINE row) get
no seeds, and FLUX falls back to a global search for those vertices alone.

    python -m fluxpype.science.seed_neighbors IN.flux OUT.flux [K]
"""

import sys

import numpy as np
from scipy.spatial import cKDTree

DEFAULT_K = 12


def read_vertices(lines):
    """(line id, vertex id, xyz) arrays for every VERTEX row; FLUX writes them as
    ``VERTEX <line> <vertex> <pos> <x> <y> <z>``, sometimes indented by a tab."""
    fl, lab, xyz = [], [], []
    for s in lines:
        p = s.split()
        if len(p) >= 7 and p[0] == "VERTEX":
            fl.append(int(p[1]))
            lab.append(int(p[2]))
            xyz.append((float(p[4]), float(p[5]), float(p[6])))
    return np.array(fl, dtype=np.int64), np.array(lab, dtype=np.int64), np.array(xyz, dtype=float).reshape(-1, 3)


def neighbor_pairs(fl, lab, xyz, k=DEFAULT_K):
    """For each vertex, the labels of its ``k`` nearest vertices that lie on other fluxons, one per fluxon."""
    n = len(lab)
    if n < 2 or k <= 0:
        return []
    q = min(n, 4 * k)
    _, idx = cKDTree(xyz).query(xyz, k=q)
    idx = np.atleast_2d(idx)
    pairs = []
    for i in range(n):
        seen = {fl[i]}
        got = 0
        for j in idx[i]:
            if j >= n or fl[j] in seen:
                continue
            seen.add(fl[j])
            pairs.append((lab[i], lab[j]))
            got += 1
            if got >= k:
                break
    return pairs


def seed_world_file(src, dst, k=DEFAULT_K):
    """Write ``src`` to ``dst`` with VNEIGHBOR seeds appended. A file that already carries VNEIGHBOR rows (a relaxed
    world FLUX wrote) is copied unchanged. Returns (vertices, seeds written)."""
    with open(src) as f:
        txt = f.read().splitlines()
    if any(s.split()[:1] == ["VNEIGHBOR"] for s in txt):
        with open(dst, "w") as f:
            f.write("\n".join(txt) + "\n")
        return 0, 0
    fl, lab, xyz = read_vertices(txt)
    pairs = neighbor_pairs(fl, lab, xyz, k)
    with open(dst, "w") as f:
        f.write("\n".join(txt) + "\n\n")
        f.write("".join(f"VNEIGHBOR {a} {b}\n" for a, b in pairs))
    return len(lab), len(pairs)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[-1].strip())
        return 2
    k = int(argv[2]) if len(argv) > 2 else DEFAULT_K
    nv, ns = seed_world_file(argv[0], argv[1], k)
    print(f"\tseed_neighbors: {nv} vertices, {ns} neighbour seeds (k={k})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
