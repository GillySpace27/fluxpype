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


def test_loop_model_reference_and_scaling(tmp_path):
    loops = [_semicircle(0, 0, 0.1, h) for h in (0.02, 0.05, 0.1)]
    f = tmp_path / "w.flux"
    f.write_text(_world_text(loops))
    fl = fe.read_world(str(f))
    geom = fe.tube_geometry(fl)
    pl = fe.loop_plasma(fl, geom, t_ref=1.5e6, alpha=0.0, beta=0.0)
    assert pl["closed"].all()
    # with no B or L dependence of the heating, the median-length loop sits at T_ref
    i = np.argsort(pl["half_length_cm"])[1]
    assert pl["T"][i] == pytest.approx(1.5e6, rel=1e-9)
    # RTV holds for every loop
    assert np.allclose(pl["T"], 1.4e3 * (pl["p"] * pl["half_length_cm"]) ** (1 / 3))
    assert np.allclose(pl["n_base"], pl["p"] / (2 * fe.K_B * pl["T"]))


def test_em_total_conserved_for_unocculted_loop(tmp_path):
    """A loop facing the observer, fully in the field of view: the image holds sum n^2 V / A_pix."""
    pytest.importorskip("sunpy")
    import sunpy.coordinates.sun as sun

    obstime = "2020-06-01T00:00"
    l0 = sun.L0(obstime).deg
    loop = _semicircle(l0, 0, 0.1, 0.05)
    f = tmp_path / "w.flux"
    f.write_text(_world_text([loop]))
    hdus, info = fe.render(
        str(f), obstime=obstime, npix=600, scale_arcsec=4.0, stratify=False, n_disc=8, r_tube_max=0.005
    )
    em = hdus["EM_TOTAL"].data.astype(float)
    fl = fe.read_world(str(f))
    geom = fe.tube_geometry(fl, r_tube_max=0.005)
    pl = fe.loop_plasma(fl, geom)
    _, dsun, _ = fe.observer_frame(obstime)
    pix_cm2 = (np.radians(4.0 / 3600) * dsun * fe.RSUN_CM) ** 2
    vol = (np.pi * geom["radius"] ** 2 * geom["len"]).sum() * fe.RSUN_CM**3
    expect = pl["n_base"][0] ** 2 * vol / pix_cm2
    # thin tube, so only the few samples that dip below r = 1 at the footpoints are lost
    assert em.sum() == pytest.approx(expect, rel=0.03)
    assert em.sum() <= expect * 1.0001
    assert hdus[0].data.shape[0] == len(hdus["LOGT"].data) - 1
    assert hdus[0].header["CTYPE1"] == "HPLN-TAN"


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
