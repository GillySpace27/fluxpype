"""FLUXlight EUV renderer: projection, occultation, loop model and EM bookkeeping."""

import numpy as np
import pytest

from fluxpype.science import fluxlight_euv as fe


def _semicircle(lon_deg, lat_deg, sep, h, n=40, flux=1.0):
    """Closed loop footpoints at (lon +- sep/2, lat), apex height h (R_sun), in Carrington Cartesian."""
    lo = np.radians(lon_deg) + np.linspace(-sep / 2, sep / 2, n)
    s = np.linspace(0, np.pi, n)
    r = 1.0 + h * np.sin(s)
    la = np.radians(lat_deg)
    return np.c_[r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la) * np.ones(n)]


def _world_text(loops):
    out, j = ["GLOBAL BOUNDARY SPHERE 0 0 0 1", ""], 100
    for p in loops:
        a, b = p[0] / np.linalg.norm(p[0]), p[-1] / np.linalg.norm(p[-1])
        out += [f"NEW {j} {a[0]:.7g} {a[1]:.7g} {a[2]:.7g} 1", f"NEW {j+1} {b[0]:.7g} {b[1]:.7g} {b[2]:.7g} -1"]
        line = j + 2
        out.append(f"LINE {line} {j} {j+1} 1")
        j += 3
        for k, v in enumerate(p[1:-1], 1):
            out.append(f"VERTEX {line} {j} {k} {v[0]:.7g} {v[1]:.7g} {v[2]:.7g}")
            j += 1
        out.append("")
    return "\n".join(out) + "\n"


def test_helioprojective_matches_sunpy():
    pytest.importorskip("sunpy")
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import Helioprojective

    obstime = "2020-06-01T00:00"
    rot, dsun, earth = fe.observer_frame(obstime)
    pts = np.random.default_rng(0).normal(size=(50, 3))
    pts *= (1.05 + 0.5 * np.random.default_rng(1).random(50))[:, None] / np.linalg.norm(pts, axis=1)[:, None]
    tx, ty = fe.to_helioprojective(pts @ rot, dsun)
    c = SkyCoord(
        pts[:, 0] * u.R_sun,
        pts[:, 1] * u.R_sun,
        pts[:, 2] * u.R_sun,
        frame="heliographic_carrington",
        observer=earth,
        obstime=obstime,
        representation_type="cartesian",
    ).transform_to(Helioprojective(observer=earth, obstime=obstime))
    assert np.allclose(tx, c.Tx.to_value(u.arcsec), atol=0.05)
    assert np.allclose(ty, c.Ty.to_value(u.arcsec), atol=0.05)


def test_occultation():
    d = 215.0
    pts = np.array([[0, 0, 1.2], [0, 0, -1.2], [1.3, 0, -0.5], [0.5, 0, -1.0]])
    assert fe.occulted(pts, d).tolist() == [False, True, False, True]


def test_loop_plasma_uses_loop_model(tmp_path):
    from fluxpype.science.loop_model import LoopModel

    loops = [_semicircle(0, lat, 0.1, h) for lat, h in ((0, 0.02), (3, 0.05), (6, 0.1))]
    f = tmp_path / "w.flux"
    f.write_text(_world_text(loops))
    fl = fe.read_world(str(f))
    geom = fe.tube_geometry(fl)
    pl = fe.loop_plasma(fl, geom, flux_per_fluxon=1e20)
    assert pl["closed"].all()
    lm = LoopModel()
    for k in range(3):
        a = lm.apex(pl["half_length_cm"][k], pl["b_sun"][k])
        assert pl["T_max"][k] == pytest.approx(a["T_max"], rel=1e-9)
        assert pl["T_grid"][k].max() == pytest.approx(a["T_max"], rel=1e-6)
    assert np.all(pl["b_sun"] > 0)
    T, ne = fe.sample_plasma(pl, np.zeros(3, int), np.array([0.0, pl["half_length_cm"][0] / fe.RSUN_CM, 1e9]))
    assert T[1] == pytest.approx(pl["T_max"][0], rel=1e-6)
    assert T[0] == pytest.approx(1e4, rel=1e-6) and T[2] == pytest.approx(1e4, rel=1e-6)


def _em_quadrature(pl, geom, nh_ne, logt_edges, n=20000):
    tot = 0.0
    for k in range(len(geom["len"])):
        s = geom["s0"][k] + (np.arange(n) + 0.5) / n * geom["len"][k]
        T, ne = fe.sample_plasma(pl, np.full(n, geom["owner"][k]), s)
        lt = np.log10(T)
        ok = (lt >= logt_edges[0]) & (lt < logt_edges[-1])
        tot += nh_ne * np.sum(ne[ok] ** 2) * geom["len"][k] / n * np.pi * geom["radius"][k] ** 2
    return tot * fe.RSUN_CM**3


def test_em_total_conserved_for_unocculted_loop(tmp_path):
    """A loop facing the observer, fully in the field of view: the image holds the loop's EM volume integral."""
    pytest.importorskip("sunpy")
    import sunpy.coordinates.sun as sun

    obstime = "2020-06-01T00:00"
    l0 = sun.L0(obstime).deg
    loop = _semicircle(l0, 0, 0.1, 0.05)
    f = tmp_path / "w.flux"
    f.write_text(_world_text([loop]))
    hdus, info = fe.render(str(f), obstime=obstime, npix=600, scale_arcsec=4.0, n_disc=8, r_tube_max=0.005)
    em = hdus["EM_TOTAL"].data.astype(float)
    fl = fe.read_world(str(f))
    geom = fe.tube_geometry(fl, r_tube_max=0.005)
    pl = fe.loop_plasma(fl, geom)
    _, dsun, _ = fe.observer_frame(obstime)
    pix_cm2 = (np.radians(4.0 / 3600) * dsun * fe.RSUN_CM) ** 2
    expect = _em_quadrature(pl, geom, fe.DEFAULTS["nh_ne"], hdus["LOGT"].data["LOGT_EDGE"]) / pix_cm2
    assert em.sum() == pytest.approx(expect, rel=0.01)
    assert hdus[0].data.shape[0] == len(hdus["LOGT"].data) - 1
    assert hdus[0].header["CTYPE1"] == "HPLN-TAN"


def test_aia_table_images(tmp_path):
    pytest.importorskip("sunpy")
    import sunpy.coordinates.sun as sun
    from astropy.io import fits

    logt, logne = np.array([4.0, 8.0]), np.array([6.0, 12.0])
    hdus = [fits.PrimaryHDU(), fits.ImageHDU(np.full((2, 2), 3.0), name="A193")]
    for name, arr in (("LOGT", logt), ("LOGNE", logne)):
        hdus.append(fits.BinTableHDU.from_columns([fits.Column(name=name, format="D", array=arr)], name=name))
    tab = tmp_path / "k.fits"
    fits.HDUList(hdus).writeto(tab)
    obstime = "2020-06-01T00:00"
    loop = _semicircle(sun.L0(obstime).deg, 10, 0.1, 0.05)
    f = tmp_path / "w.flux"
    f.write_text(_world_text([loop]))
    out, _ = fe.render(str(f), obstime=obstime, npix=300, scale_arcsec=8.0, aia_table=str(tab))
    assert np.allclose(out["SYN_A193"].data, 3.0 * out["EM_TOTAL"].data, rtol=1e-5)


def test_response_contraction(tmp_path):
    pytest.importorskip("sunpy")
    import sunpy.coordinates.sun as sun

    obstime = "2020-06-01T00:00"
    loop = _semicircle(sun.L0(obstime).deg, 10, 0.1, 0.05)
    f = tmp_path / "w.flux"
    f.write_text(_world_text([loop]))
    resp = tmp_path / "r.txt"
    resp.write_text("# logT R\n5.0 2.0\n7.5 2.0\n")
    hdus, _ = fe.render(
        str(f), obstime=obstime, npix=300, scale_arcsec=8.0, responses={"flat": fe.load_response(str(resp))}
    )
    assert np.allclose(hdus["SYN_flat"].data, 2.0 * hdus["EM_TOTAL"].data, rtol=1e-5)
