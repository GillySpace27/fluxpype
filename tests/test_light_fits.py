"""FLUXlight FITS output and cache key (audit A7-03)."""

import filecmp
import os
import pickle
import types

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits

pytest.importorskip("sunpy")
pytest.importorskip("sunkit_image")  # fluxlight imports the rendering stack at module level

from fluxpype import names  # noqa: E402
from fluxpype.science import fluxlight, light_fits  # noqa: E402

NPIX, FOV = 21, 3.0


def _results(tmp_path, inst="hmi"):
    rng = np.random.default_rng(1)
    unit = u.erg / u.cm**2 / u.s / u.sr
    tb = rng.random((NPIX, NPIX)) + 1.0
    world = types.SimpleNamespace(cr="2150", nflx="100", inst=inst)
    return {
        "B_total": tb * unit,
        "B_polarized": 0.5 * tb * unit,
        "Polarization_fraction": np.full((NPIX, NPIX), 0.5),
        "Polarization_angle": rng.random((NPIX, NPIX)),
        "Column_Density": tb * u.cm**-2,
        "X": None,
        "Y": None,
        "fov": FOV * u.R_sun,
        "influence_length": 1.0 * u.R_sun,
        "nz": 5,
        "nxy": NPIX,
        "world": world,
        "light_path": str(tmp_path / "light"),
    }


def test_fits_name_and_header(tmp_path):
    res = _results(tmp_path)
    paths = light_fits.write_thomson_fits(res, date="2014-05-18T20:00:00", observer_lon_deg=172.0)
    assert sorted(os.path.basename(p) for p in paths) == sorted(
        f"cr2150_f100_hmi_{c}.fits" for c in ("tb", "pb", "pfrac", "pangle", "coldens")
    )
    assert os.path.basename(paths[0]) == names.light_fits(2150, 100, "hmi", "tb")
    with fits.open(paths[0]) as hl:
        h, d = hl[0].header, hl[0].data
    assert (h["CTYPE1"], h["CTYPE2"]) == ("HPLN-TAN", "HPLT-TAN")
    assert h["CRVAL1"] == 0.0 and h["CRVAL2"] == 0.0
    from sunpy.coordinates import sun

    rsun = sun.angular_radius("2014-05-18T20:00:00").to_value(u.arcsec)
    assert h["CDELT1"] == pytest.approx(2 * FOV / (NPIX - 1) * rsun)
    assert h["CRPIX1"] == (NPIX + 1) / 2
    assert h["DATE-OBS"].startswith("2014-05-18T20:00")
    assert h["HGLT_OBS"] == 0.0 and h["CRLN_OBS"] == 172.0 and h["DSUN_OBS"] > 1e11 and h["RSUN_REF"] == 695_700_000.0
    assert h["BUNIT"] == "erg s-1 sr-1 cm-2" and h["CHANNEL"] == "tb"
    # pixel values are the renderer's numbers
    np.testing.assert_array_equal(d, res["B_total"].value)


def test_pole_on_header_and_default_date(tmp_path):
    res = _results(tmp_path)
    p, *_ = light_fits.write_thomson_fits(res)
    h = fits.getheader(p)
    assert h["HGLT_OBS"] == 90.0 and "CRLN_OBS" not in h
    assert h["DATE-OBS"][:4].isdigit()  # middle of CR 2150, not "now"
    assert h["DATE-OBS"].startswith("2014-")


def test_sunpy_map_round_trip(tmp_path):
    from sunpy.map import Map

    p, *_ = light_fits.write_thomson_fits(_results(tmp_path), date="2014-05-18T20:00:00", observer_lon_deg=172.0)
    m = Map(p)
    c = m.pixel_to_world(((NPIX - 1) / 2) * u.pix, ((NPIX - 1) / 2) * u.pix)
    assert abs(c.Tx.to_value(u.arcsec)) < 1e-6 and abs(c.Ty.to_value(u.arcsec)) < 1e-6


def test_euv_channel_files(tmp_path):
    img = np.arange(16, dtype=np.float32).reshape(4, 4)
    hdr = fits.Header()
    for k, v in dict(CTYPE1="HPLN-TAN", CTYPE2="HPLT-TAN", CRPIX1=2.5, CRPIX2=2.5, CDELT1=2.4, CDELT2=2.4).items():
        hdr[k] = v
    hdr["DATE-OBS"] = "2014-05-18T20:00:00"
    hdr["BUNIT"] = "DN s-1 pix-1"
    hdr["AIATABLE"] = "k.fits"
    hdus = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(img, header=hdr, name="SYN_A193")])
    hdus.append(fits.ImageHDU(img, header=fits.Header({"BUNIT": "x"}), name="SYN_Acustom"))  # not AIA: skipped
    (p,) = light_fits.write_euv_fits(hdus, str(tmp_path), 2150, 100, "hmi")
    assert os.path.basename(p) == "cr2150_f100_hmi_aia193.fits"
    with fits.open(p) as hl:
        assert hl[0].header["BUNIT"] == "DN s-1 pix-1" and hl[0].header["CTYPE1"] == "HPLN-TAN"
        np.testing.assert_array_equal(hl[0].data, img)


def _world(tmp_path, text="world one"):
    p = tmp_path / "cr2150_f100_hmi_relaxed_s10.flux"
    p.write_text(text)
    return str(p)


def test_cache_key_changes_with_geometry_resolution_channel_and_world(tmp_path, monkeypatch):
    w = _world(tmp_path)
    base = light_fits.cache_key(w, fluxlight.render_params())
    assert base == light_fits.cache_key(w, fluxlight.render_params())  # stable
    for attr, val in (("observer_lon", 10.0), ("nxy", 99), ("fov", 5.0), ("nz", 7), ("method", "vertex")):
        monkeypatch.setattr(fluxlight, attr, val)
        assert light_fits.cache_key(w, fluxlight.render_params()) != base, attr
        monkeypatch.undo()
    assert light_fits.cache_key(w, {**fluxlight.render_params(), "channels": ["tb"]}) != base
    assert light_fits.cache_key(_world(tmp_path, "world two"), fluxlight.render_params()) != base


def test_changed_geometry_rerenders_and_unchanged_reuses(tmp_path, monkeypatch):
    w = _world(tmp_path)
    calls = []

    def fake_sim(**kw):
        calls.append(kw)
        return {"B_total": np.ones((2, 2)) * u.erg / u.cm**2 / u.s / u.sr}

    monkeypatch.setattr(fluxlight, "simulate_thomson_scattering", fake_sim)
    monkeypatch.setattr(
        fluxlight,
        "read_flux_world",
        lambda f: types.SimpleNamespace(out_dir=str(tmp_path), cr="2150", nflx="100", inst="hmi"),
    )
    fluxlight.do_fluxlight(w)
    fluxlight.do_fluxlight(w)
    assert len(calls) == 1  # same world and parameters: cache reused
    monkeypatch.setattr(fluxlight, "observer_lon", 33.0)
    fluxlight.do_fluxlight(w)
    assert len(calls) == 2 and calls[-1]["observer_lon"] == pytest.approx(np.deg2rad(33.0))
    monkeypatch.setattr(fluxlight, "nxy", 50)
    fluxlight.do_fluxlight(w)
    assert len(calls) == 3
    open(w, "w").write("changed world")
    fluxlight.do_fluxlight(w)
    assert len(calls) == 4


def test_cache_without_key_is_not_reused(tmp_path, monkeypatch):
    w = _world(tmp_path)
    with open(w.replace(".flux", ".light"), "wb") as f:
        pickle.dump({"world": types.SimpleNamespace(cr="1", nflx="1", inst="hmi")}, f)  # written before A7-03
    calls = []
    monkeypatch.setattr(fluxlight, "simulate_thomson_scattering", lambda **kw: calls.append(1) or {})
    monkeypatch.setattr(fluxlight, "read_flux_world", lambda f: types.SimpleNamespace(out_dir=str(tmp_path)))
    fluxlight.do_fluxlight(w)
    assert calls == [1]


def test_png_is_unchanged_by_fits_output(tmp_path):
    res = _results(tmp_path)
    res["B_total"] = res["B_total"] + 0.0 * res["B_total"].unit
    out = []
    for tag in ("before", "after"):
        d = tmp_path / tag
        d.mkdir()
        res["light_path"] = str(d)
        if tag == "after":
            light_fits.write_thomson_fits(res, date="2014-05-18T20:00:00")
        fluxlight.triplot(res, 2.0)
        out.append(next(p for p in d.iterdir() if p.suffix == ".png"))
    assert filecmp.cmp(out[0], out[1], shallow=False)


def test_observer_geometry_in_header(tmp_path):
    from sunpy.coordinates import sun
    from sunpy.map import Map

    date = "2014-05-18T20:00:00"
    l0 = float(sun.L0(date).to_value(u.deg))
    for off, want in ((0.0, 0.0), (90.0, 90.0), (-90.0, -90.0)):
        p, *_ = light_fits.write_thomson_fits(_results(tmp_path), date=date, observer_lon_deg=l0 + off)
        h = fits.getheader(p)
        assert h["HGLN_OBS"] == pytest.approx(want, abs=0.5) and h["CRVAL1"] == 0.0 and h["CRLN_OBS"] == l0 + off
        oc = Map(p).observer_coordinate
        assert oc.lon.to_value(u.deg) == pytest.approx(want, abs=0.5)


def test_unknown_rotation_skips_fits_with_warning(tmp_path, capsys):
    res = _results(tmp_path)
    res["world"].cr = "unknown"
    assert light_fits.write_thomson_fits(res) == []
    assert "FITS export skipped" in capsys.readouterr().out
    assert not os.path.exists(res["light_path"]) or not os.listdir(res["light_path"])


def test_dtype_kept_and_no_temp_left(tmp_path):
    paths = light_fits.write_thomson_fits(_results(tmp_path), date="2014-05-18T20:00:00")
    assert fits.getdata(paths[0]).dtype.newbyteorder("=") == np.float64
    assert not [f for f in os.listdir(os.path.dirname(paths[0])) if f.endswith(".tmp")]


def test_render_version_in_key(tmp_path, monkeypatch):
    w = _world(tmp_path)
    k = light_fits.cache_key(w, {})
    monkeypatch.setattr(light_fits, "RENDER_VERSION", light_fits.RENDER_VERSION + 1)
    assert light_fits.cache_key(w, {}) != k


def test_corrupt_cache_is_a_miss(tmp_path, monkeypatch):
    w = _world(tmp_path)
    open(w.replace(".flux", ".light"), "wb").write(b"\x80\x04truncated")
    calls = []
    monkeypatch.setattr(fluxlight, "simulate_thomson_scattering", lambda **kw: calls.append(1) or {})
    monkeypatch.setattr(fluxlight, "read_flux_world", lambda f: types.SimpleNamespace(out_dir=str(tmp_path)))
    fluxlight.do_fluxlight(w)
    assert calls == [1] and not os.path.exists(w.replace(".flux", ".light.tmp"))
