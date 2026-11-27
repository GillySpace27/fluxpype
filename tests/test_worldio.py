"""One shared .flux parser (audit A2-06): both LINE dialects, and relaxed-world output unchanged."""

import hashlib
import os

import numpy as np
import pytest

from fluxpype import worldio
from fluxpype.science.open_world_python import read_flux_world
from fluxpype.science.read_world import rdworld as science_rdworld

HERE = os.path.dirname(__file__)
RELAXED = [
    os.path.join(HERE, "data", "realflux", "relaxed", f"{n}.relaxed.flux")
    for n in ("simple-4", "planar_shear2", "twisted")
] + [os.path.join(HERE, "golden", "audit", "simple-4.relaxed.flux")]

INPUT_FORM = """GLOBAL BOUNDARY SPHERE 0 0 0 1 0 0
NEW 2 1 0 0 1
NEW 3 0 1 0 -1
LINE 4 2 3 1
VERTEX 4 5 1 0.9 0.5 0.1
VERTEX 4 6 2 0.5 0.9 0.1
NEW 7 0.5 0.5 1.2 1
LINE 8 7 -2 1
VERTEX 8 9 1 0.6 0.6 1.6
VERTEX 8 10 2 0.7 0.7 2.0
"""


def _digest(w):
    """Hash of everything the science reader exposes; recorded from the pre-A2-06 reader (origin/dev d01e7b9)."""
    h = hashlib.sha256()
    h.update(repr((w.fc.id, w.fc.x, w.fc.y, w.fc.z, w.fc.fl)).encode())
    h.update(
        repr(
            (
                [int(i) for i in w.fx.id],
                [int(i) for i in w.fx.fc0],
                [int(i) for i in w.fx.fc1],
                [float(f) for f in w.fx.fl],
            )
        ).encode()
    )
    for a in "xyz":
        for v in getattr(w.fx, a):
            h.update(np.asarray(v, dtype=float).tobytes())
    return h.hexdigest()


# (fluxons, total points, digest) recorded by running the old reader on each fixture
OLD_READER = {
    "simple-4.relaxed.flux": (4, 63, "82cd705ae6015e2c4dcf3d69b591a71dedd317c94b11d203b5c24dab7b36d106"),
    "planar_shear2.relaxed.flux": (9, 189, "41c4c9e1d6f1bd95689dddde2f6132bb159c1e977f1fd59a2f946095ef5e23d7"),
    "twisted.relaxed.flux": (25, 1201, "2e7d5f1387a59ac949e0f74447f6c19368f73279def59716381e163749d1fbc8"),
}


def test_input_form_parses():
    w = worldio.parse_text(INPUT_FORM)
    assert w.dialect == "input" and w.header == ["GLOBAL BOUNDARY SPHERE 0 0 0 1 0 0"]
    assert [c[0] for c in w.concentrations] == [2, 3, 7]
    closed, open_ = w.lines
    assert (closed.id, closed.fc0, closed.fc1, closed.flux) == (4, 2, 3, 1.0)
    assert closed.start == (1.0, 0.0, 0.0) and closed.end == (0.0, 1.0, 0.0)
    xs, ys, zs = worldio.line_arrays(closed)
    assert xs == [1.0, 0.9, 0.5, 0.0] and ys == [0.0, 0.5, 0.9, 1.0] and zs == [0.0, 0.1, 0.1, 0.0]
    # open end: no concentration, so no end point
    assert open_.fc1 == -2 and open_.end is None
    assert worldio.line_arrays(open_) == [[0.5, 0.6, 0.7], [0.5, 0.6, 0.7], [1.2, 1.6, 2.0]]


def test_input_form_with_core_style_negative_pseudo_line_is_skipped():
    w = worldio.parse_text("NEW 2 1 0 0 1\nNEW 3 0 1 0 -1\nLINE -10 2 3 0\nLINE 4 2 3 1\n")
    assert [ln.id for ln in w.lines] == [4]


@pytest.mark.parametrize("path", RELAXED)
def test_core_form_parses(path):
    w = worldio.read(path)
    assert w.dialect == "core" and w.lines and all(ln.start and ln.end for ln in w.lines)


def test_read_accepts_text_and_path(tmp_path):
    p = tmp_path / "w.flux"
    p.write_text(INPUT_FORM)
    assert worldio.read(str(p)) == worldio.parse_text(INPUT_FORM) == worldio.read(p)


def test_three_input_forms_load_in_all_readers(tmp_path):
    """make_world_sphere style (text above), a planar scene, and an unrelaxed scene-like world all load."""
    from fluxpype.scenes import PlanarArcadeSpec, compile_flux

    texts = {"sphere": INPUT_FORM, "planar": compile_flux(PlanarArcadeSpec()), "unrelaxed": _unrelaxed_fixture()}
    for name, text in texts.items():
        p = tmp_path / name / "world" / "w.flux"
        p.parent.mkdir(parents=True)
        p.write_text(text)
        fw = read_flux_world(str(p))  # FLUXlight's reader; used to raise IndexError
        assert len(fw.fluxons) == len(worldio.parse_text(text).lines) > 0
        rw = science_rdworld(str(p))
        assert len(rw.fx.id) == len(fw.fluxons)
        from fluxpype.pipe_helper import rdworld

        assert len(rdworld(str(p)).fx.id) == len(fw.fluxons)


def _unrelaxed_fixture():
    return open(os.path.join(HERE, "data", "realflux", "fixtures", "twisted.flux")).read()


@pytest.mark.parametrize("path", RELAXED)
def test_relaxed_parse_identical_to_pre_change_reader(path):
    w = science_rdworld(path)
    n, pts, digest = OLD_READER[os.path.basename(path)]
    assert (len(w.fx.id), sum(len(v) for v in w.fx.x)) == (n, pts)
    assert _digest(w) == digest


@pytest.mark.parametrize("path", RELAXED)
def test_read_flux_world_agrees_with_science_reader_on_relaxed(path):
    fw = read_flux_world(path)
    old = science_rdworld(path)
    assert sorted(fw.fluxons) == sorted(int(i) for i in old.fx.id)
    for i, fid in enumerate(old.fx.id):
        o = np.column_stack((old.fx.x[i], old.fx.y[i], old.fx.z[i]))
        n = fw.fluxons[int(fid)].get_vertices()  # oriented smallest radius first
        assert np.array_equal(n, o) or np.array_equal(n, o[::-1])


def test_missing_file_raises_not_empty_world():
    with pytest.raises(FileNotFoundError):
        worldio.read("/no/such/dir/missing.flux")
    with pytest.raises(FileNotFoundError):
        worldio.read("missing.flux")


def test_core_form_exact_endpoints_and_six_parameter_line():
    text = (
        "NEW 100 0.6 0 0.8 -1\nNEW 200 0 0.6 0.8 1\n"
        "LINE 101 -302 -301 200 100 1 0.0 0.6 0.8 0.6 0.0 0.8\n"
        "VERTEX 101 102 1 0.0 0.7 1.2\nVERTEX 101 103 2 0.4 0.4 1.5\n"
        "LINE 105 -312 -311 200 100 1\nVERTEX 105 106 1 0.1 0.2 0.3\n"
    )
    full, six = worldio.parse_text(text).lines
    assert (full.fc0, full.fc1, full.start, full.end) == (200, 100, (0.0, 0.6, 0.8), (0.6, 0.0, 0.8))
    assert worldio.line_arrays(full) == [
        [0.0, 0.0, 0.4, 0.6],
        [0.6, 0.7, 0.4, 0.0],
        [0.8, 1.2, 1.5, 0.8],
    ]
    # io.c reads the 6 parameter form too; its end points come from the concentrations
    assert (six.fc0, six.fc1, six.start, six.end) == (200, 100, (0.0, 0.6, 0.8), (0.6, 0.0, 0.8))


def test_add_fluxon_accepts_numpy_endpoints():
    from fluxpype.science.open_world_python import FluxWorld

    fw = FluxWorld("x")
    fw.add_fluxon(1, 2, 3, 1.0, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), [0.9], [0.5], [0.1])
    assert fw.fluxons[1].x_coords.tolist() == [1.0, 0.9, 0.0] or fw.fluxons[1].x_coords.tolist() == [0.0, 0.9, 1.0]
    fw.add_fluxon(2, 2, -2, 1.0, np.array([1.0, 0, 0]), None, [0.9], [0.5], [0.1])
    assert len(fw.fluxons[2].x_coords) == 2
