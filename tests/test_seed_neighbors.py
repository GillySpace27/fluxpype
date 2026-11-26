"""seed_neighbors: KD-tree VNEIGHBOR seeds appended to a world file."""

import numpy as np

from fluxpype.science import seed_neighbors as sn

WORLD = """GLOBAL FORCES f_pressure_equi2b f_curvature f_vertex4
GLOBAL BOUNDARY SPHERE 0 0 0 1

NEW 100 1 0 0 1
NEW 101 0 1 0 -1
LINE 102 100 101 1
VERTEX 102 103 1 1.1 0.1 0
VERTEX 102 104 2 0.8 0.8 0
\tVERTEX 102 105 3 0.1 1.1 0

NEW 106 1 0 0.1 1
NEW 107 0 1 0.1 -1
LINE 108 106 107 1
VERTEX 108 109 1 1.1 0.1 0.1
VERTEX 108 110 2 0.8 0.8 0.1
VERTEX 108 111 3 0.1 1.1 0.1
"""


def test_pairs_only_cross_fluxon(tmp_path):
    src, dst = tmp_path / "w.flux", tmp_path / "s.flux"
    src.write_text(WORLD)
    nv, ns = sn.seed_world_file(str(src), str(dst), k=3)
    assert nv == 6
    rows = [ln.split() for ln in dst.read_text().splitlines() if ln.startswith("VNEIGHBOR")]
    assert len(rows) == ns == 6  # only one other fluxon, so one seed per vertex
    owner = {103: 102, 104: 102, 105: 102, 109: 108, 110: 108, 111: 108}
    for _, a, b in rows:
        assert owner[int(a)] != owner[int(b)]
    # nearest on the other fluxon is the vertex at the same position index
    pairs = {int(a): int(b) for _, a, b in rows}
    assert pairs[103] == 109 and pairs[110] == 104
    assert WORLD.strip() in dst.read_text()  # original rows untouched


def test_already_seeded_is_copied(tmp_path):
    src, dst = tmp_path / "w.flux", tmp_path / "s.flux"
    src.write_text(WORLD + "VNEIGHBOR 103 109\n")
    assert sn.seed_world_file(str(src), str(dst)) == (0, 0)
    assert dst.read_text().count("VNEIGHBOR") == 1


def test_k_limits_seeds():
    fl = np.repeat(np.arange(10), 2)
    lab = np.arange(20)
    xyz = np.random.default_rng(1).random((20, 3))
    pairs = sn.neighbor_pairs(fl, lab, xyz, k=3)
    assert len(pairs) == 20 * 3
