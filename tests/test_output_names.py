"""Audit A7-01: wind tables and FLUXlight images carry ``cr{CR}_f{N}_{inst}``; batches with the old names still read.

Two runs of one preset (same rotation, count and instrument) still write the same names. That needs a run id and is
not covered here."""

import glob
import os
import time

from fluxpype import names
from fluxpype.plumber import plotrun
from fluxpype.plumber.covers import _find
from fluxpype.plumber.datfile import KINDS
from fluxpype.plumber.monitor import ProbeContext, _first_match
from fluxpype.plumber.plotcatalog import PLOTS
from fluxpype.plumber.results import for_this_nflux, plot_catalog, task_results


def touch(path, mtime=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x")
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def ctx_for(d, n, inst, cr="2310", batch="B"):
    adapt = "1" if inst == "adapt" else "0"
    return ProbeContext(data_dir=str(d), batch=batch, cr=cr, nflux=str(n), adapt=adapt, inst=inst)


# ------------------------------------------------------------------------------------------------ names module


def test_names_carry_count_and_instrument():
    assert names.inst_name(0) == "hmi" and names.inst_name("0") == "hmi" and names.inst_name(1) == "adapt"
    assert names.wind_dat(2150, 100, "adapt", "bmag") == "cr2150_f100_adapt_radial_bmag.dat"
    assert names.wind_dat(2150, 100, "hmi", "bmag_all", stiff=True) == "cr2150_f100_hmi_stiff_radial_bmag_all.dat"
    assert names.wind_dat(2150, 100, "hmi", "wind_parker") == "cr2150_f100_hmi_radial_wind_parker.dat"
    assert names.light_png(2150, 100, "hmi", "light_fluxel_301") == "cr2150_f100_hmi_fluxlight_light_fluxel_301.png"
    # a world with no count or instrument in its name keeps the old spelling
    assert names.light_png(2150, "unknown", "unknown", "light_x_1") == "cr2150_fluxlight_light_x_1.png"


def test_is_this_task_filters_count_and_instrument():
    assert names.is_this_task("cr1_f100_hmi_fluxlight_light_x_1.png", 100, "hmi")
    assert not names.is_this_task("cr1_f100_adapt_fluxlight_light_x_1.png", 100, "hmi")
    assert not names.is_this_task("cr1_f200_hmi_fluxlight_light_x_1.png", 100, "hmi")
    assert names.is_this_task("cr1_f100_radial_bmag.dat", 100, "adapt")  # old name, no instrument: kept
    assert names.is_this_task("cr1_fluxlight_light_x_1.png", 100, "adapt")  # old name, no count: kept
    assert names.is_this_task("cr1_f100_hmi_relaxed_s40.flux", 100, "hmi")
    assert for_this_nflux("cr1_f100_adapt_radial_fr.dat", "100")
    assert not for_this_nflux("cr1_f100_adapt_x.dat", "100", "hmi")


def test_wind_path_falls_back_to_the_old_name(tmp_path):
    w = str(tmp_path)
    new, old = os.path.join(w, "cr1_f5_hmi_radial_fr.dat"), os.path.join(w, "cr1_f5_radial_fr.dat")
    assert names.wind_dat_path(w, 1, 5, "hmi", "fr") == new  # neither: the new name
    touch(old)
    assert names.wind_dat_path(w, 1, 5, "hmi", "fr") == old
    touch(new)
    assert names.wind_dat_path(w, 1, 5, "hmi", "fr") == new


def test_legacy_template():
    light = "imgs/light/cr{cr}_f{nflux}_{inst}_fluxlight_light_*.png"
    assert names.legacy_template(light) == "imgs/light/cr{cr}_fluxlight_light_*.png"
    assert names.legacy_template("w/cr{cr}_f{nflux}_{inst}_radial_bmag.dat") == "w/cr{cr}_f{nflux}_radial_bmag.dat"
    assert names.legacy_template("w/cr{cr}_f{nflux}_{inst}_stiff_radial_x") == "w/cr{cr}_f{nflux}_stiff_radial_x"
    assert names.legacy_template("world/cr{cr}_f{nflux}_{inst}_relaxed_s*.flux") is None


def test_datfile_kinds_accept_both_spellings():
    import re

    pats = {k["id"]: k["pattern"] for k in KINDS}
    for kind, new, old in [
        ("bmag", "cr1_f5_adapt_radial_bmag.dat", "cr1_f5_radial_bmag.dat"),
        ("bmag_all", "cr1_f5_hmi_stiff_radial_bmag_all.dat", "cr1_f5_stiff_radial_bmag_all.dat"),
        ("fr", "cr1_f5_hmi_radial_fr.dat", "cr1_f5_radial_fr.dat"),
        ("wind", "cr1_f5_hmi_radial_wind_parker.dat", "cr1_f5_radial_wind_parker.dat"),
    ]:
        assert re.match(pats[kind], new) and re.match(pats[kind], old)


# ------------------------------------------------------------------------------------------------ four tasks, one batch

TASKS = [(100, "hmi"), (100, "adapt"), (200, "hmi"), (200, "adapt")]


def write_batch(d, legacy=False):
    """Four tasks of one batch. Old style: all four write the one shared light file (the overwrite this item removes)."""
    t0 = time.time()
    b = os.path.join(str(d), "batches", "B")
    for n, inst in TASKS:
        for kind in ("bmag", "fr"):
            f = names.wind_dat(2310, n, None if legacy else inst, kind)
            touch(os.path.join(b, "data", "cr2310", "wind", f), t0 + 5)
        f = names.light_png(2310, "unknown" if legacy else n, "unknown" if legacy else inst, "light_x_1")
        touch(os.path.join(b, "imgs", "light", f), t0 + 10)
        touch(os.path.join(b, "data", "cr2310", "world", f"cr2310_f{n}_{inst}_relaxed_s40.flux"), t0 + 2)
    return t0


def test_four_tasks_get_four_files_and_each_headline_is_its_own(tmp_path):
    t0 = write_batch(tmp_path)
    light = sorted(os.listdir(tmp_path / "batches/B/imgs/light"))
    assert len(light) == 4 and len(set(light)) == 4
    for n, inst in TASKS:
        r = task_results(ctx_for(tmp_path, n, inst), t0, t0 + 60)
        assert os.path.basename(r["headline"]) == f"cr2310_f{n}_{inst}_fluxlight_light_x_1.png"
        by = {p["id"]: p for p in r["products"]}
        assert [f["name"] for f in by["fluxlight"]["files"]] == [f"cr2310_f{n}_{inst}_fluxlight_light_x_1.png"]
        assert sorted(f["name"] for f in by["wind"]["files"]) == [
            f"cr2310_f{n}_{inst}_radial_bmag.dat",
            f"cr2310_f{n}_{inst}_radial_fr.dat",
        ]


def test_cover_follows_the_new_names_and_never_shows_a_sibling(tmp_path):
    write_batch(tmp_path)
    for n, inst in TASKS:
        path, kind = _find(ctx_for(tmp_path, n, inst))
        assert kind == "image" and os.path.basename(path) == f"cr2310_f{n}_{inst}_fluxlight_light_x_1.png"
    only = tmp_path / "other"
    touch(str(only / "batches/B/imgs/light/cr2310_f100_hmi_fluxlight_light_x_1.png"))
    found = _find(ctx_for(only, 100, "adapt"))
    assert found is None or found[1] != "image"


# ------------------------------------------------------------------------------------------------ old batches


def test_old_batch_is_still_found_by_every_reader(tmp_path):
    t0 = write_batch(tmp_path, legacy=True)
    ctx = ctx_for(tmp_path, 100, "hmi")
    r = task_results(ctx, t0, t0 + 60)
    by = {p["id"]: p for p in r["products"]}
    assert [f["name"] for f in by["fluxlight"]["files"]] == ["cr2310_fluxlight_light_x_1.png"]
    assert r["headline"].endswith("cr2310_fluxlight_light_x_1.png")
    assert {f["name"] for f in by["wind"]["files"]} == {"cr2310_f100_radial_bmag.dat", "cr2310_f100_radial_fr.dat"}
    assert os.path.basename(_find(ctx)[0]) == "cr2310_fluxlight_light_x_1.png"
    assert any(glob.glob(p, recursive=True) for p in ctx.patterns()["wind"])


def test_monitor_probe_sees_old_and_new_fluxlight(tmp_path):
    ctx = ctx_for(tmp_path, 100, "hmi")
    assert _first_match(ctx.patterns()["fluxlight"]) is None
    touch(str(tmp_path / "batches/B/imgs/light/cr2310_fluxlight_light_x_1.png"))
    assert _first_match(ctx.patterns()["fluxlight"]).endswith("cr2310_fluxlight_light_x_1.png")
    touch(str(tmp_path / "batches/B/imgs/light/cr2310_f100_hmi_fluxlight_light_x_1.png"))
    assert _first_match(ctx.patterns()["fluxlight"]).endswith("cr2310_f100_hmi_fluxlight_light_x_1.png")
    # a sibling instrument's new file does not mark this task's FLUXlight stage found
    new_only = tmp_path / "x"
    touch(str(new_only / "batches/B/imgs/light/cr2310_f100_hmi_fluxlight_light_x_1.png"))
    assert _first_match(ctx_for(new_only, 100, "adapt").patterns()["fluxlight"]) is None


def test_post_hoc_plot_inputs_and_catalog_fall_back(tmp_path):
    ctx = ctx_for(tmp_path, 100, "hmi")
    bm = next(e for e in PLOTS if e["id"] == "bmag_scatter")
    found, missing = plotrun.resolve_needs(bm, ctx)
    assert missing == ["batches/B/data/cr2310/wind/cr2310_f100_hmi_radial_bmag.dat"]
    old = tmp_path / "batches/B/data/cr2310/wind/cr2310_f100_radial_bmag.dat"
    touch(str(old))
    found, missing = plotrun.resolve_needs(bm, ctx)
    assert missing == [] and found["bmag"] == str(old)
    new = tmp_path / "batches/B/data/cr2310/wind/cr2310_f100_hmi_radial_bmag.dat"
    touch(str(new))
    assert plotrun.resolve_needs(bm, ctx)[0]["bmag"] == str(new)
    # the plot it makes: an old batch's image name, then the new one once it exists
    touch(str(tmp_path / "batches/B/imgs/bmag/cr2310_f100_radial_bmag.png"))
    cat = {e["id"]: e for e in plot_catalog(ctx, None, None)}
    assert [f["name"] for f in cat["bmag_scatter"]["files"]] == ["cr2310_f100_radial_bmag.png"]
    touch(str(tmp_path / "batches/B/imgs/bmag/cr2310_f100_hmi_radial_bmag.png"))
    cat = {e["id"]: e for e in plot_catalog(ctx, None, None)}
    assert [f["name"] for f in cat["bmag_scatter"]["files"]] == ["cr2310_f100_hmi_radial_bmag.png"]
    assert [os.path.basename(p) for p in plotrun.targets(bm, ctx)] == ["cr2310_f100_hmi_radial_bmag.png"]


# ------------------------------------------------------------------------------------------------ fake_flux end to end


def test_fake_flux_batch_of_four_writes_four_distinct_light_files(make_scheduler):
    from tests.test_plumber_scheduler import FAST, finish

    sched = make_scheduler()
    over = {"fluxon_count": "[100, 200]", "adapts": "[0, 1]"}
    job = sched.create_job("t_light", overrides=over, backend="fake", extra=FAST)
    assert len(job["tasks"]) == 4
    finish(sched, job["id"], 120)
    seen = set()
    for t in sched.store.tasks_for_job(job["id"]):
        assert t["status"] == "ok"
        ctx = sched.probe_context(t)
        r = task_results(ctx, t.get("started"), t.get("ended"))
        head = os.path.basename(r["headline"])
        assert f"_f{t['nflux']}_{ctx.inst}_" in head, head
        seen.add(r["headline"])
    assert len(seen) == 4
