"""One shared .flux parser (audit A2-06): both LINE dialects, and relaxed-world output unchanged."""

import importlib.util
import os
import subprocess
import sys

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


def _git_show(ref_path):
    root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True).strip()
    return subprocess.check_output(["git", "show", ref_path], cwd=root, text=True)


def _legacy_rdworld():
    """The reader as it was before A2-06 (origin/dev at the time); skipped if the history is not available."""
    try:
        src = _git_show("d01e7b9:fluxpype/science/read_world.py")
    except Exception:
        pytest.skip("pre-A2-06 reader not available from git history")
    spec = importlib.util.spec_from_loader("legacy_read_world", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, "legacy_read_world", "exec"), mod.__dict__)
    return mod.rdworld


def test_input_form_parses():
    w = worldio.parse_text(INPUT_FORM)
    assert w.dialect == "input" and w.header == ["GLOBAL BOUNDARY SPHERE 0 0 0 1 0 0"]
    assert [c[0] for c in w.concentrations] == [2, 3, 7]
    closed, open_ = w.lines
    assert (closed.id, closed.fc0, closed.fc1, closed.flux) == (4, 2, 3, 1.0)
    assert closed.start == (1.0, 0.0, 0.0) and closed.end == (0.0, 1.0, 0.0)
    xs, ys, zs = worldio.line_arrays(closed)
    assert xs == [1.0, 0.9, 0.5, 0.0] and zs == [0.0, 0.1, 0.1, 0.0]
    # open end: no concentration, so no end point
    assert open_.fc1 == -2 and open_.end is None
    assert worldio.line_arrays(open_)[2] == [1.2, 1.6, 2.0]


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
    assert worldio.read(str(p)) == worldio.read(INPUT_FORM) == worldio.read(p)


def test_three_input_forms_load_in_all_readers(tmp_path):
    """make_world_sphere style (text above), a planar scene, and an unrelaxed scene-like world all load."""
    from fluxpype.scenes import PlanarArcadeSpec, compile_flux

    texts = {"sphere": INPUT_FORM, "planar": compile_flux(PlanarArcadeSpec()), "unrelaxed": _unrelaxed_fixture()}
    for name, text in texts.items():
        p = tmp_path / name / "world" / "w.flux"
        p.parent.mkdir(parents=True)
        p.write_text(text)
        fw = read_flux_world(str(p))  # FLUXlight's reader; used to raise IndexError
        assert len(fw.fluxons) == len(worldio.read(text).lines) > 0
        rw = science_rdworld(str(p))
        assert len(rw.fx.id) == len(fw.fluxons)
        from fluxpype.pipe_helper import rdworld

        assert len(rdworld(str(p)).fx.id) == len(fw.fluxons)


def _unrelaxed_fixture():
    return open(os.path.join(HERE, "data", "realflux", "fixtures", "twisted.flux")).read()


@pytest.mark.parametrize("path", RELAXED)
def test_relaxed_parse_identical_to_pre_change_reader(path):
    old, new = _legacy_rdworld()(path), science_rdworld(path)
    assert old.fc.id == new.fc.id
    for a in ("x", "y", "z", "fl"):
        assert getattr(old.fc, a) == getattr(new.fc, a)
    assert [int(i) for i in old.fx.id] == [int(i) for i in new.fx.id]
    assert [int(i) for i in old.fx.fc0] == [int(i) for i in new.fx.fc0]
    assert [int(i) for i in old.fx.fc1] == [int(i) for i in new.fx.fc1]
    assert old.fx.fl == new.fx.fl
    for a in ("x", "y", "z"):
        for u, v in zip(getattr(old.fx, a), getattr(new.fx, a)):
            assert np.array_equal(u, v)


@pytest.mark.parametrize("path", RELAXED)
def test_read_flux_world_unchanged_on_relaxed(path):
    fw = read_flux_world(path)
    old = _legacy_rdworld()(path)
    assert sorted(fw.fluxons) == sorted(int(i) for i in old.fx.id)
    # read_flux_world orients each fluxon smallest radius first; compare as sets of vertices per fluxon
    for i, fid in enumerate(old.fx.id):
        f = fw.fluxons[int(fid)]
        o = np.column_stack((old.fx.x[i], old.fx.y[i], old.fx.z[i]))
        n = f.get_vertices()
        assert np.array_equal(n, o) or np.array_equal(n, o[::-1])
