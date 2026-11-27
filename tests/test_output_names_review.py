"""A7-01 review fixes: readers find the qualified wind names (and the old ones), FluxWorld never takes the sibling
instrument's expansion file, the Cranmer/Tempest subprocess is told the instrument, cached FLUXlight results get their
instrument back, and the Plumber legacy wind glob also matches the old stiff spelling."""

import glob
import os
import pickle
import re
import sys
import types
from pathlib import Path

import pytest

from fluxpype import names
from fluxpype.plumber.monitor import ProbeContext

SCIENCE = Path(__file__).resolve().parent.parent / "fluxpype" / "science"


def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x")


class Stop(Exception):
    pass


# ------------------------------------------------------------------------------------------------ Perl (text level)


def test_cranmer_and_tempest_subprocess_get_the_instrument():
    src = (SCIENCE / "map_fluxon_flow_parallel_master.pm").read_text()
    assert re.search(r"\$n_want, \$max_processes_arg, \$inst\) = @_;", src)
    assert re.search(r'\$adapt_arg = \( defined \$inst and \$inst eq \'adapt\' \) \? " --adapt 1" : "";', src)
    for script in ("tempest.py", "cranmer_wind.py"):
        assert f'python3 fluxpype/science/{script} --cr $CR --nwant $n_want$adapt_arg"' in src
    wind = (SCIENCE / "get_wind.pm").read_text()
    assert "$CR, $n_want, undef, $inst );" in wind  # get_wind hands its instrument to the wind map master


def test_get_wind_direct_execution_passes_the_instrument():
    wind = (SCIENCE / "get_wind.pm").read_text()
    direct = wind.split("if ($0 eq __FILE__)")[1]
    assert re.search(r'get_wind\(\s*\$this_world,[^;]*"hmi"\s*\);', direct)  # the world searched for is the _hmi one


def test_cranmer_argv_flows_through_the_precompute_name(tmp_path, monkeypatch):
    """parse_args gets the ADAPT file name only when --adapt 1 is on the command line, which the master now adds."""
    pytest.importorskip("scipy")
    from fluxpype.science import cranmer_wind

    wind = tmp_path / "batches" / "B" / "data" / "cr2310" / "wind"
    touch(str(wind / "cr2310_f100_adapt_radial_bmag_all.dat"))
    touch(str(wind / "cr2310_f100_hmi_radial_bmag_all.dat"))
    base = ["x", "--cr", "2310", "--nwant", "100", "--batch", "B", "--dat_dir", str(tmp_path)]
    cfg = {"rotations": [2310], "data_dir": str(tmp_path), "batch_name": "B", "fluxon_count": [100]}
    monkeypatch.setattr(cranmer_wind, "configurations", lambda args=None: cfg)
    for extra, inst in (([], "hmi"), (["--adapt", "1"], "adapt")):
        monkeypatch.setattr(sys, "argv", base + extra)
        filename, _ = cranmer_wind.parse_args()
        assert os.path.basename(filename) == f"cr2310_f100_{inst}_radial_bmag_all.dat"


# ------------------------------------------------------------------------------------------------ readers


def _batch_with(tmp_path, wind_files):
    for f in wind_files:
        touch(str(tmp_path / "batches" / "B" / "data" / "cr2310" / "wind" / f))
    return {"data_dir": str(tmp_path), "batch_name": "B", "cr": 2310, "nwant": 100}


@pytest.mark.parametrize("module", ["plot_wind_map_latitude", "plot_wind_map_detailed"])
def test_wind_map_plotters_default_to_the_qualified_name_then_the_old_one(module, tmp_path, monkeypatch):
    mod = pytest.importorskip(f"fluxpype.plotting.{module}")
    fn = getattr(mod, module)

    def seen_path(configs):
        def fake_loadtxt(path, *a, **k):
            raise Stop(path)

        monkeypatch.setattr(mod.np, "loadtxt", fake_loadtxt)
        with pytest.raises(Stop) as e:
            fn(configs)
        return os.path.basename(e.value.args[0])

    kind = "wind_parker" if module == "plot_wind_map_detailed" else "wind"
    cfg = _batch_with(tmp_path, [])
    cfg["adapt"] = 1
    assert seen_path(dict(cfg)) == names.wind_dat(2310, 100, "adapt", kind)  # nothing yet: the new name
    cfg = _batch_with(tmp_path, [names.wind_dat(2310, 100, None, kind)])
    assert seen_path(dict(cfg)) == names.wind_dat(2310, 100, None, kind)  # old batch
    cfg = _batch_with(tmp_path, [names.wind_dat(2310, 100, "adapt", kind), names.wind_dat(2310, 100, "hmi", kind)])
    cfg["adapt"] = 0
    assert seen_path(dict(cfg)) == names.wind_dat(2310, 100, "hmi", kind)
    cfg["adapt"] = 1
    assert seen_path(dict(cfg)) == names.wind_dat(2310, 100, "adapt", kind)


@pytest.mark.parametrize("module", ["tempest", "tempest_bkk"])
def test_tempest_readers_find_qualified_and_old_names(module, tmp_path, monkeypatch):
    pytest.importorskip("scipy")
    mod = pytest.importorskip(f"fluxpype.science.{module}")
    monkeypatch.setattr(
        mod,
        "configurations",
        lambda args=None: {"rotations": [2310], "data_dir": str(tmp_path), "batch_name": "B", "fluxon_count": [100]},
    )
    base = ["x", "--cr", "2310", "--nwant", "100", "--batch", "B", "--dat_dir", str(tmp_path)]
    _batch_with(tmp_path, ["cr2310_f100_radial_bmag_all.dat"])
    monkeypatch.setattr(sys, "argv", base)
    assert os.path.basename(mod.parse_args()[0]) == "cr2310_f100_radial_bmag_all.dat"  # old batch
    _batch_with(tmp_path, ["cr2310_f100_hmi_radial_bmag_all.dat", "cr2310_f100_adapt_radial_bmag_all.dat"])
    assert os.path.basename(mod.parse_args()[0]) == "cr2310_f100_hmi_radial_bmag_all.dat"
    monkeypatch.setattr(sys, "argv", base + ["--adapt", "1"])
    assert os.path.basename(mod.parse_args()[0]) == "cr2310_f100_adapt_radial_bmag_all.dat"


# ------------------------------------------------------------------------------------------------ FluxWorld


def _world_file(tmp_path, inst):
    return str(tmp_path / "batches" / "B" / "data" / "cr2310" / "world" / f"cr2310_f100_{inst}_relaxed_s40.flux")


def test_flux_world_picks_its_own_instruments_area_file(tmp_path):
    from fluxpype.science.open_world_python import FluxWorld

    wind = tmp_path / "batches" / "B" / "data" / "cr2310" / "wind"
    for f in ("cr2310_f100_hmi_radial_fr.dat", "cr2310_f100_adapt_radial_fr.dat"):
        touch(str(wind / f))
    assert os.path.basename(FluxWorld(_world_file(tmp_path, "hmi")).flux_area_file) == "cr2310_f100_hmi_radial_fr.dat"
    w = FluxWorld(_world_file(tmp_path, "adapt"))
    assert w.inst == "adapt" and os.path.basename(w.flux_area_file) == "cr2310_f100_adapt_radial_fr.dat"


def test_flux_world_never_takes_the_sibling_and_falls_back_to_the_old_name(tmp_path):
    from fluxpype.science.open_world_python import FluxWorld

    wind = tmp_path / "batches" / "B" / "data" / "cr2310" / "wind"
    touch(str(wind / "cr2310_f100_adapt_radial_fr.dat"))
    assert FluxWorld(_world_file(tmp_path, "hmi")).flux_area_file is None  # only the sibling's file exists
    touch(str(wind / "cr2310_f100_radial_fr.dat"))  # an old batch's table
    assert os.path.basename(FluxWorld(_world_file(tmp_path, "hmi")).flux_area_file) == "cr2310_f100_radial_fr.dat"
    touch(str(wind / "cr2310_f100_hmi_radial_fr.dat"))  # the qualified one wins over the old one
    assert os.path.basename(FluxWorld(_world_file(tmp_path, "hmi")).flux_area_file) == "cr2310_f100_hmi_radial_fr.dat"


# ------------------------------------------------------------------------------------------------ cached FLUXlight


def test_cached_light_results_get_their_instrument_back(tmp_path):
    pytest.importorskip("sunkit_image")  # fluxlight imports the rendering stack at module level
    from fluxpype.science import fluxlight

    world_file = _world_file(tmp_path, "adapt")
    old_world = types.SimpleNamespace(cr="2310", nflx="100")  # pickled before the world carried .inst
    touch(world_file)
    results = {
        "world": old_world,
        "light_path": str(tmp_path / "light"),
        "nz": 1,
        "cache_key": __import__("fluxpype.science.light_fits", fromlist=["x"]).cache_key(
            world_file, fluxlight.render_params()
        ),  # A7-03: a cache needs its key
    }
    with open(world_file.replace(".flux", ".light"), "wb") as f:
        pickle.dump(results, f)
    got = fluxlight.do_fluxlight(world_file)
    assert got["world"].inst == "adapt"
    assert os.path.basename(fluxlight._light_path(got, "light_x_1")) == "cr2310_f100_adapt_fluxlight_light_x_1.png"


# ------------------------------------------------------------------------------------------------ Plumber legacy glob


def test_monitor_wind_glob_matches_old_stiff_spelling_but_not_a_sibling(tmp_path):
    ctx = ProbeContext(data_dir=str(tmp_path), batch="B", cr="2310", nflux="100", adapt="0", inst="hmi")
    wind = tmp_path / "batches" / "B" / "data" / "cr2310" / "wind"
    touch(str(wind / "cr2310_f100_adapt_radial_bmag.dat"))  # the sibling's new file
    assert not any(glob.glob(p) for p in ctx.patterns()["wind"])
    touch(str(wind / "cr2310_f100_stiff_radial_bmag_all.dat"))
    assert any(glob.glob(p) for p in ctx.patterns()["wind"])


def test_old_batch_with_plain_and_stiff_tables_shows_both(tmp_path):
    from fluxpype.plumber.results import task_results

    for f in ("cr2310_f100_radial_bmag.dat", "cr2310_f100_stiff_radial_bmag_all.dat"):
        touch(str(tmp_path / "batches" / "B" / "data" / "cr2310" / "wind" / f))
    ctx = ProbeContext(data_dir=str(tmp_path), batch="B", cr="2310", nflux="100", adapt="0", inst="hmi")
    by = {p["id"]: p for p in task_results(ctx, None, None)["products"]}
    assert {f["name"] for f in by["wind"]["files"]} == {
        "cr2310_f100_radial_bmag.dat",
        "cr2310_f100_stiff_radial_bmag_all.dat",
    }
