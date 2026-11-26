"""Fixes from the FLUX stack audit that unblock magnetogram -> FLUX -> FLUXlight (A5-06, A2-01, A2-02)."""

import shutil
import types
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from fluxpype.science import open_world_python as owp
from fluxpype.science import pfss_funcs

# Written by FLUX core write_world: the Python reader does not take the hand-written input form (audit A2-06).
FIXTURE = Path(__file__).parent / "golden" / "audit" / "simple-4.relaxed.flux"


class _FakeLine:
    def __init__(self, rads, is_open=False):
        n = len(rads)
        self.is_open = is_open
        self.polarity = 1
        self.coords = SkyCoord(
            np.linspace(0.1, 0.2, n) * u.rad,
            np.linspace(0.0, 0.1, n) * u.rad,
            np.asarray(rads, float) * u.R_sun,
            frame="icrs",
            representation_type="spherical",
        )


def test_magex_tracer_skips_an_empty_field_line(monkeypatch, tmp_path):
    # A5-06: a real ADAPT map (CR2225) gave one zero-point closed line and np.max raised ValueError.
    lines = [_FakeLine([1.01, 1.3, 1.01]), _FakeLine([]), _FakeLine([1.01, 1.2, 1.5, 1.2, 1.01])]

    class _Tracer:
        def __init__(self, max_steps):
            pass

        def trace(self, seeds, output):
            return lines

    fake = types.SimpleNamespace(tracing=types.SimpleNamespace(PerformanceTracer=_Tracer))
    monkeypatch.setattr(pfss_funcs, "_magex_pfss", fake)
    output = types.SimpleNamespace(coordinate_frame="icrs")
    f_vars = (np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]), np.array([1, -1, 1]))
    fl_open, fl_closed, skipped, _, n_open, n_closed = pfss_funcs._trace_lines_magex(
        output, f_vars, tmp_path / "open.dat", tmp_path / "closed.dat", adapt=False
    )
    assert skipped == 1
    assert n_closed == 2 and n_open == 0


def test_world_loads_without_a_wind_folder(tmp_path):
    # A2-01: FluxWorld used to require <world dir>/../wind/*_radial_fr.dat even for light_method = fluxel.
    world_dir = tmp_path / "data" / "cr2150" / "world"
    world_dir.mkdir(parents=True)
    dst = world_dir / "cr2150_f4_hmi_relaxed_s0.flux"
    shutil.copy(FIXTURE, dst)
    w = owp.read_flux_world(str(dst))
    assert w.flux_area_file is None
    assert len(w.fluxons) > 0
    with pytest.raises(FileNotFoundError, match="radial_fr.dat"):
        w.compute_cross_sectional_areas(method="file")


def test_vectorized_segment_distance_matches_the_per_point_loop():
    # A2-02: the batched distance must equal calling the per-point helper and taking nanmin.
    rng = np.random.default_rng(3)
    seg_a = rng.normal(size=(200, 3))
    seg_b = seg_a + rng.normal(scale=0.2, size=(200, 3))
    seg_b[5] = seg_a[5]  # a zero-length segment takes the t = 0 branch
    pts = rng.normal(scale=2.0, size=(1000, 3))
    idx = rng.integers(0, 200, size=(1000, 24))
    idx[:, 0] = 5
    loop = np.array(
        [
            np.nanmin(owp._point_to_segments_distance_and_t(p, seg_a[np.unique(i)], seg_b[np.unique(i)])[0])
            for p, i in zip(pts, idx)
        ]
    )
    vec = owp._min_distance_to_candidate_segments(pts, idx, seg_a, seg_b, chunk=97)
    assert np.array_equal(loop, vec)
