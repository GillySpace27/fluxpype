"""The stage manifest (A2-09): its templates reproduce today's file names, and the other modules read it."""

import os
import re
import types

import pytest

from fluxpype import stages
from fluxpype.plumber import fake_flux, monitor, results, server

D = "/d"
B = "/d/batches/B/data/cr2310"


def _ctx(inst):
    return monitor.ProbeContext(
        D, "B", "2310", "100", "0" if inst == "hmi" else "1", inst, reduce="4", adapt_select="2"
    )


def test_probe_patterns_are_todays_paths():
    """Literal spellings written by pipe_helper.pm / names.py / the pdl, for HMI and ADAPT (A7-01 names first)."""
    hmi = _ctx("hmi").patterns()
    assert hmi == {
        "magnetogram": [f"{D}/magnetograms/CR2310_r4_hmi.fits"],
        "footpoints": [f"{B}/floc/floc_cr2310_*_f100_hmi.dat", "/d/batches/B/cr2310/floc/floc_cr2310_*_f100_hmi.dat"],
        "pfss": [f"{D}/pfss/pfss_cr2310_*hmi*.pkl"],
        "world": [f"{B}/world/cr2310_f100_hmi.flux"],
        "relax": [f"{B}/world/cr2310_f100_hmi_relaxed_s*.flux"],
        "wind": [f"{B}/wind/cr2310_f100_hmi_*", f"{B}/wind/cr2310_f100_[!ah]*"],
        "fluxlight": ["/d/batches/B/**/cr2310_f100_hmi_fluxlight_*.png", "/d/batches/B/**/cr2310_fluxlight_*.png"],
    }
    assert _ctx("adapt").patterns()["magnetogram"] == [f"{D}/magnetograms/CR2310_rf2_adapt.fits"]
    assert _ctx("adapt").patterns()["footpoints"][0] == f"{B}/floc/floc_cr2310_*_f100_adapt.dat"


@pytest.mark.parametrize("inst, rtag", [("hmi", "r4"), ("adapt", "rf2")])
def test_fake_backend_writes_todays_names(inst, rtag):
    fk = fake_flux.Fake.__new__(fake_flux.Fake)
    fk.a = types.SimpleNamespace(cr="2310", nflux="100")
    fk.data_dir, fk.batch, fk.reduce, fk.adapt_sel, fk.inst = D, "B", "4", "2", inst
    assert fk.out("magnetogram", "fits") == f"{D}/magnetograms/CR2310_{rtag}_{inst}.fits"
    assert fk.out("footpoints", "floc") == f"{B}/floc/floc_cr2310_{rtag}_f100_{inst}.dat"
    assert fk.out("footpoints", "open") == f"{B}/floc/floc_open_cr2310_{rtag}_f100_{inst}.dat"
    assert fk.out("footpoints", "closed") == f"{B}/floc/floc_closed_cr2310_{rtag}_f100_{inst}.dat"
    assert fk.out("pfss", "pkl") == f"{D}/pfss/pfss_cr2310_{rtag}_{inst}.pkl"
    assert fk.out("world", "flux") == f"{B}/world/cr2310_f100_{inst}.flux"
    assert fk.out("relax", "flux", steps=840) == f"{B}/world/cr2310_f100_{inst}_relaxed_s840.flux"
    assert fk.out("world_plots", "png") == f"{B}/world/cr2310_f100_{inst}_relaxed.png"
    assert fk.out("fieldmap", "png") == f"{B}/fieldmap/cr2310_f100_{inst}_fieldmap.png"
    assert fk.out("wind", "speed", method="parker") == f"{B}/wind/cr2310_f100_{inst}_radial_wind_parker.dat"
    assert fk.out("wind", "table", kind="bmag_all") == f"{B}/wind/cr2310_f100_{inst}_radial_bmag_all.dat"
    assert (
        fk.out("wind", "full_velocity", method="parker")
        == f"{B}/wind/full_velocity_profiles/results_parker_full_velocity.dat"
    )
    assert fk.out("wind_plots", "png", method="parker") == f"{B}/wind/cr2310_f100_{inst}_parker_wind.png"
    assert fk.out("fluxlight", "png", method="parker") == f"{B}/light/cr2310_f100_{inst}_fluxlight_light_parker_1.png"
    assert fk.out("euv", "model_fits", channel=193) == f"{B}/light/cr2310_f100_{inst}_aia193.fits"


def test_fake_probes_find_what_the_fake_writes():
    """Every probe pattern matches the fake's own file of that stage (the two read one table)."""
    import fnmatch

    fk = fake_flux.Fake.__new__(fake_flux.Fake)
    fk.a = types.SimpleNamespace(cr="2310", nflux="100")
    fk.data_dir, fk.batch, fk.reduce, fk.adapt_sel, fk.inst = D, "B", "4", "2", "hmi"
    ctx = _ctx("hmi").patterns()
    written = {
        "magnetogram": fk.out("magnetogram", "fits"),
        "footpoints": fk.out("footpoints", "floc"),
        "pfss": fk.out("pfss", "pkl"),
        "world": fk.out("world", "flux"),
        "relax": fk.out("relax", "flux", steps=840),
        "wind": fk.out("wind", "speed", method="parker"),
        "fluxlight": fk.out("fluxlight", "png", method="parker"),
    }
    for sid, path in written.items():
        assert any(fnmatch.fnmatch(path, pat.replace("**/", "*/")) for pat in ctx[sid]), sid


def test_other_modules_read_the_manifest():
    ids = [s.id for s in stages.STAGES]
    assert len(ids) == len(set(ids))
    assert monitor.STAGE_IDS == ids and [s for s, _ in monitor.STAGES] == ids
    assert [p[2] for p in results.PRODUCTS] == [s.id for s in stages.STAGES if s.product]
    assert stages.public() == [{"id": s.id, "label": s.label} for s in stages.STAGES]
    assert server.stage_manifest is stages
    # every log marker names a stage
    for attr in ("log_start", "log_reused", "log_failure"):
        assert all(sid in ids for sid, _ in stages.log_markers(attr))


def test_every_stage_id_emitted_by_the_pdl_and_the_fake_is_in_the_manifest():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ids = set(stages.STAGE_IDS)
    pdl = open(os.path.join(root, "fluxpype", "science", "magnetogram2wind.pdl"), encoding="utf-8").read()
    pdl_ids = set(re.findall(r'stage_start\("(\w+)"\)', pdl))
    assert pdl_ids and pdl_ids <= ids, pdl_ids - ids
    fake = open(os.path.join(root, "fluxpype", "plumber", "fake_flux.py"), encoding="utf-8").read()
    fake_ids = set(re.findall(r'self\.stage\("(\w+)"\)', fake)) | set(re.findall(r'self\.end\("(\w+)"', fake))
    assert fake_ids and fake_ids <= ids, fake_ids - ids
