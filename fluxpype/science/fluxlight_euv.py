"""FLUXlight EUV: render a FLUX world as a per-pixel emission-measure cube EM(T), matched to an observer and time.

The renderer has three separable parts, so each can be replaced without touching the others:

1. Geometry (this module): each fluxon is a flux tube. Its local cross-section comes from the distance to the
   nearest vertex on another fluxon (radius = half that distance, capped at ``r_tube_max``), and B is the fluxon's
   flux over that area, in relative units. Tube samples are projected into a helioprojective image for an observer
   at Earth at ``obstime``; samples hidden behind the photosphere are dropped.
2. Plasma (``loop_plasma``): a temperature and base density per closed fluxon. Default is the RTV scaling law
   T_max = 1400 (p L)^(1/3) (Rosner, Tucker & Vaiana 1978; T in K, p in dyn cm^-2, L the loop half-length in cm)
   with heating E_H ∝ <B>^alpha L^-beta, normalised so the median loop has T_ref, plus isothermal hydrostatic
   stratification in spherical gravity. Open fluxons emit nothing by default. All of alpha, beta, T_ref and
   mu are assumptions of this model, set by the caller.
3. Emissivity: the output is EM(T) per pixel (cm^-5 per log T bin), which is independent of any atomic model.
   A synthetic channel image is the contraction sum_T EM(T) R(T) with a temperature response R(T) the caller
   supplies (``load_response``), e.g. an AIA response exported from SolarSoft. A GHOSTS emissivity module would
   plug in at this same point.

Output: a FITS file with the EM cube (primary HDU, 3-D WCS: x, y, log T), the total EM image, one image per
supplied response, and a LOOPS table with the per-fluxon model values, so every number in the image can be
traced back to a fluxon.

    python -m fluxpype.science.fluxlight_euv WORLD.flux [--time ISO] [--npix 1024] [--scale 2.4]
                                             [--response NAME=FILE ...] [--out OUT.fits] [--png]
"""

import argparse
import os
import sys

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
from scipy.spatial import cKDTree

from fluxpype.plumber.fluxfile import parse_text

RSUN_CM = const.R_sun.to_value(u.cm)
K_B = const.k_B.cgs.value
M_P = const.m_p.cgs.value
G_SUN = (const.G * const.M_sun / const.R_sun**2).cgs.value
RTV_T_COEFF = 1.4e3  # RTV (1978) T_max = 1.4e3 (p L)^(1/3), cgs, L = half-length

DEFAULTS = dict(
    t_ref=1.5e6,  # K, temperature given to the median closed loop (assumption)
    alpha=1.0,  # heating exponent on mean |B| (assumption)
    beta=1.0,  # heating exponent on half-length (assumption)
    mu=0.6,  # mean molecular weight for the scale height (assumption)
    r_tube_max=0.05,  # R_sun, cap on tube radius (assumption)
    stratify=True,
    closed_rmax=1.05,  # a fluxon is closed when both ends lie below this radius (R_sun)
)


# ---------------------------------------------------------------------------------------------------- world


def read_world(path):
    """Fluxons of a .flux file as a list of dicts {id, flux, xyz (n,3) in R_sun}."""
    with open(path) as f:
        w = parse_text(f.read())
    fl = w["fluxons"] if "fluxons" in w else w["lines"] if "lines" in w else None
    if fl is None:
        raise ValueError(f"{path}: parser returned no fluxons")
    xyz = np.asarray(fl["xyz"], dtype=float).reshape(-1, 3)
    off = fl["offsets"]
    out = []
    for i, fid in enumerate(fl["id"]):
        pts = xyz[off[i] : off[i + 1]]
        if len(pts) >= 2:
            out.append({"id": int(fid), "flux": abs(float(fl["flux"][i])) or 1.0, "xyz": pts})
    return out


def tube_geometry(fluxons, r_tube_max=DEFAULTS["r_tube_max"]):
    """Per-segment midpoints, direction, length (R_sun), tube radius (R_sun) and owner index."""
    mids, dirs, lens, own = [], [], [], []
    for k, f in enumerate(fluxons):
        p = f["xyz"]
        d = np.diff(p, axis=0)
        ln = np.linalg.norm(d, axis=1)
        ok = ln > 0
        mids.append(0.5 * (p[1:] + p[:-1])[ok])
        dirs.append((d[ok] / ln[ok, None]))
        lens.append(ln[ok])
        own.append(np.full(ok.sum(), k))
    mids, dirs, lens, own = map(np.concatenate, (mids, dirs, lens, own))
    # nearest midpoint that belongs to another fluxon
    kq = min(len(mids), 24)
    dist, idx = cKDTree(mids).query(mids, k=kq)
    other = own[idx] != own[:, None]
    first = np.where(other.any(axis=1), other.argmax(axis=1), kq - 1)
    dnn = dist[np.arange(len(mids)), first]
    radius = np.clip(0.5 * dnn, 1e-4, r_tube_max)
    return {"mid": mids, "dir": dirs, "len": lens, "radius": radius, "owner": own}


# ---------------------------------------------------------------------------------------------------- plasma


def loop_plasma(fluxons, geom, t_ref=1.5e6, alpha=1.0, beta=1.0, closed_rmax=1.05, **_):
    """Per-fluxon model values. Returns a dict of arrays (one entry per fluxon): closed, half_length_cm, b_rel,
    heat_rel, p (dyn cm^-2), T (K), n_base (cm^-3). Open fluxons get T = n = 0."""
    nf = len(fluxons)
    own, ln, rad = geom["owner"], geom["len"], geom["radius"]
    flux = np.array([f["flux"] for f in fluxons])
    b_seg = flux[own] / (np.pi * rad**2)  # relative units: flux per fluxon / area in R_sun^2
    length = np.bincount(own, weights=ln, minlength=nf)
    b_mean = np.bincount(own, weights=b_seg * ln, minlength=nf) / np.maximum(length, 1e-30)
    r0 = np.array([np.linalg.norm(f["xyz"][0]) for f in fluxons])
    r1 = np.array([np.linalg.norm(f["xyz"][-1]) for f in fluxons])
    closed = (r0 < closed_rmax) & (r1 < closed_rmax)
    half = 0.5 * length * RSUN_CM
    out = dict(closed=closed, half_length_cm=half, b_rel=b_mean, heat_rel=np.zeros(nf), p=np.zeros(nf))
    out["T"] = np.zeros(nf)
    out["n_base"] = np.zeros(nf)
    if not closed.any():
        return out
    l_ref = np.median(half[closed])
    b_ref = np.median(b_mean[closed])
    p_ref = (t_ref / RTV_T_COEFF) ** 3 / l_ref
    h = (b_mean / b_ref) ** alpha * (half / l_ref) ** (-beta)
    # RTV: E_H ∝ p^(7/6) L^(-5/6)  =>  p = p_ref h^(6/7) (L/L_ref)^(5/7)
    p = p_ref * h ** (6.0 / 7.0) * (half / l_ref) ** (5.0 / 7.0)
    T = RTV_T_COEFF * (p * half) ** (1.0 / 3.0)
    n = p / (2.0 * K_B * T)  # fully ionised hydrogen, n_e = n_p
    out["heat_rel"] = np.where(closed, h, 0.0)
    out["p"] = np.where(closed, p, 0.0)
    out["T"] = np.where(closed, T, 0.0)
    out["n_base"] = np.where(closed, n, 0.0)
    return out


def density_along(r, T, n_base, mu=0.6, stratify=True):
    """Isothermal hydrostatic density at radius r (R_sun) above a base at 1 R_sun."""
    if not stratify:
        return n_base * np.ones_like(r)
    with np.errstate(divide="ignore", invalid="ignore"):
        h_rsun = K_B * T / (mu * M_P * G_SUN) / RSUN_CM
        return n_base * np.exp(-(1.0 / h_rsun) * (1.0 - 1.0 / np.maximum(r, 1.0)))


# ---------------------------------------------------------------------------------------------------- observer


def observer_frame(obstime):
    """Rotation from Carrington Cartesian (FLUX world axes: z = rotation axis, x = Carrington longitude 0) to
    heliocentric-Cartesian (z toward the observer, y toward solar north in the sky), and the observer distance in
    R_sun. Built from sunpy's own frames applied to the three unit vectors, so it carries sunpy's conventions."""
    from astropy.coordinates import CartesianRepresentation, SkyCoord
    from sunpy.coordinates import HeliographicCarrington, Heliocentric, get_earth

    earth = get_earth(obstime)
    hgc = HeliographicCarrington(obstime=obstime, observer=earth)
    hcc = Heliocentric(obstime=obstime, observer=earth)
    eye = np.eye(3)
    rep = CartesianRepresentation(eye[0] * u.R_sun, eye[1] * u.R_sun, eye[2] * u.R_sun)
    c = SkyCoord(hgc.realize_frame(rep)).transform_to(hcc).cartesian
    rot = np.column_stack([c.x.to_value(u.R_sun), c.y.to_value(u.R_sun), c.z.to_value(u.R_sun)])  # row i = image of e_i
    dsun = earth.radius.to_value(u.R_sun)
    return rot, dsun, earth


def to_helioprojective(hcc, dsun):
    """(Tx, Ty) in arcsec from heliocentric-Cartesian points in R_sun (Thompson 2006, eq. 15)."""
    x, y, z = hcc.T
    dz = dsun - z
    d = np.sqrt(x * x + y * y + dz * dz)
    tx = np.degrees(np.arctan2(x, dz)) * 3600.0
    ty = np.degrees(np.arcsin(y / d)) * 3600.0
    return tx, ty


def occulted(hcc, dsun):
    """True where the ray from the observer to the point passes through the photosphere (r = 1) first."""
    o = np.array([0.0, 0.0, dsun])
    v = hcc - o
    a = np.einsum("ij,ij->i", v, v)
    b = 2.0 * (v @ o)
    c = o @ o - 1.0
    disc = b * b - 4 * a * c
    hit = disc > 0
    t0 = np.where(hit, (-b - np.sqrt(np.where(hit, disc, 0.0))) / (2 * a), 2.0)
    return hit & (t0 > 0) & (t0 < 1.0 - 1e-9)


# ---------------------------------------------------------------------------------------------------- render


def render(
    world_path,
    obstime=None,
    npix=1024,
    scale_arcsec=2.4,
    logt_edges=None,
    n_disc=16,
    seed=0,
    responses=None,
    **model,
):
    """Render a world. Returns (hdulist, info). ``responses`` maps a name to a callable R(logT) in
    DN cm^5 s^-1 pix^-1 (or any per-EM unit), applied to the EM cube."""
    import sunpy.coordinates.sun as sun
    from astropy.time import Time

    opts = dict(DEFAULTS, **model)
    if logt_edges is None:
        logt_edges = np.arange(5.0, 7.5001, 0.05)
    logt_edges = np.asarray(logt_edges, dtype=float)
    fluxons = read_world(world_path)
    cr = _cr_from_name(world_path)
    if obstime is None:
        if cr is None:
            raise ValueError("no --time given and no Carrington rotation in the file name")
        obstime = sun.carrington_rotation_time(cr + 0.5)
    obstime = Time(obstime)

    geom = tube_geometry(fluxons, opts["r_tube_max"])
    plasma = loop_plasma(fluxons, geom, **opts)
    rot, dsun, earth = observer_frame(obstime)

    rng = np.random.default_rng(seed)
    pix_rsun = np.radians(scale_arcsec / 3600.0) * (dsun - 1.0)  # pixel size near the limb, for sampling
    pix_area_cm2 = (np.radians(scale_arcsec / 3600.0) * dsun * RSUN_CM) ** 2
    ny = nx = int(npix)
    nt = len(logt_edges) - 1
    cube = np.zeros((nt, ny, nx))

    seg_on = plasma["closed"][geom["owner"]]
    sel = np.nonzero(seg_on)[0]
    T_seg = plasma["T"][geom["owner"]]
    tbin = np.digitize(np.log10(np.maximum(T_seg, 1.0)), logt_edges) - 1
    n_kept = n_samples = 0
    for chunk in np.array_split(sel, max(1, len(sel) // 20000)):
        if not len(chunk):
            continue
        mid, d, ln, rad = geom["mid"][chunk], geom["dir"][chunk], geom["len"][chunk], geom["radius"][chunk]
        m = np.maximum(1, np.ceil(ln / (0.5 * pix_rsun)).astype(int))
        rep = np.repeat(np.arange(len(chunk)), m * n_disc)
        frac = (np.concatenate([(np.arange(k) + 0.5) / k for k in m]) - 0.5).repeat(n_disc)
        # point along the segment, then a uniform offset in the disc perpendicular to it
        e1 = np.cross(d, np.where(np.abs(d[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(d, e1)
        rr = rad[rep] * np.sqrt(rng.random(len(rep)))
        th = 2 * np.pi * rng.random(len(rep))
        pts = (
            mid[rep]
            + (frac * ln[rep])[:, None] * d[rep]
            + (rr * np.cos(th))[:, None] * e1[rep]
            + (rr * np.sin(th))[:, None] * e2[rep]
        )
        r = np.linalg.norm(pts, axis=1)
        own = geom["owner"][chunk][rep]
        n = density_along(r, plasma["T"][own], plasma["n_base"][own], opts["mu"], opts["stratify"])
        vol_cm3 = (np.pi * rad[rep] ** 2 * ln[rep] / (m[rep] * n_disc)) * RSUN_CM**3
        w = n * n * vol_cm3 / pix_area_cm2  # cm^-5 contributed to the pixel
        hcc = pts @ rot  # rot rows are the heliocentric images of the Carrington x, y, z unit vectors
        keep = (r >= 1.0) & ~occulted(hcc, dsun)
        tx, ty = to_helioprojective(hcc[keep], dsun)
        ix = np.floor(tx / scale_arcsec + nx / 2).astype(int)
        iy = np.floor(ty / scale_arcsec + ny / 2).astype(int)
        tb = tbin[chunk][rep][keep]
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (tb >= 0) & (tb < nt)
        np.add.at(cube, (tb[ok], iy[ok], ix[ok]), w[keep][ok])
        n_samples += len(rep)
        n_kept += int(ok.sum())

    hdus = _to_fits(cube, logt_edges, scale_arcsec, obstime, earth, dsun, plasma, fluxons, world_path, opts)
    if responses:
        tc = 0.5 * (logt_edges[1:] + logt_edges[:-1])
        for name, fn in responses.items():
            img = np.tensordot(fn(tc), cube, axes=(0, 0))
            h = fits.ImageHDU(img.astype(np.float32), header=_wcs2(hdus[0].header), name=f"SYN_{name}"[:68])
            h.header["BUNIT"] = "response unit x cm-5 (e.g. DN s-1 pix-1)"
            h.header["COMMENT"] = f"sum over log T of EM(T) * R_{name}(T)"
            hdus.append(h)
    info = dict(
        obstime=obstime.isot,
        n_fluxons=len(fluxons),
        n_closed=int(plasma["closed"].sum()),
        samples=n_samples,
        samples_in_image=n_kept,
        T_range=(
            (float(plasma["T"][plasma["closed"]].min()), float(plasma["T"][plasma["closed"]].max()))
            if plasma["closed"].any()
            else None
        ),
    )
    return hdus, info


def _cr_from_name(path):
    import re

    m = re.search(r"cr(\d{4})", os.path.basename(path))
    return int(m.group(1)) if m else None


def _wcs2(hdr3):
    h = fits.Header()
    for k in ("CTYPE", "CUNIT", "CRPIX", "CRVAL", "CDELT"):
        for a in (1, 2):
            h[f"{k}{a}"] = hdr3[f"{k}{a}"]
    for k in ("DATE-OBS", "HGLN_OBS", "HGLT_OBS", "DSUN_OBS", "RSUN_REF", "RSUN_OBS"):
        if k in hdr3:
            h[k] = hdr3[k]
    return h


def _to_fits(cube, edges, scale, obstime, earth, dsun, plasma, fluxons, world_path, opts):
    from sunpy.coordinates import HeliographicStonyhurst

    nt, ny, nx = cube.shape
    hgs = earth.transform_to(HeliographicStonyhurst(obstime=obstime))
    h = fits.Header()
    h["CTYPE1"], h["CUNIT1"], h["CRPIX1"], h["CRVAL1"], h["CDELT1"] = "HPLN-TAN", "arcsec", nx / 2 + 0.5, 0.0, scale
    h["CTYPE2"], h["CUNIT2"], h["CRPIX2"], h["CRVAL2"], h["CDELT2"] = "HPLT-TAN", "arcsec", ny / 2 + 0.5, 0.0, scale
    h["CTYPE3"], h["CUNIT3"], h["CRPIX3"] = "LOGT", "", 1.0
    h["CRVAL3"], h["CDELT3"] = 0.5 * (edges[0] + edges[1]), float(edges[1] - edges[0])
    h["BUNIT"] = "cm-5"
    h["DATE-OBS"] = obstime.isot
    h["HGLN_OBS"] = hgs.lon.to_value(u.deg)
    h["HGLT_OBS"] = hgs.lat.to_value(u.deg)
    h["DSUN_OBS"] = earth.radius.to_value(u.m)
    h["RSUN_REF"] = const.R_sun.to_value(u.m)
    h["RSUN_OBS"] = np.degrees(np.arcsin(1.0 / dsun)) * 3600.0
    h["CONTENT"] = "FLUXlight EUV emission measure per log T bin"
    h["WORLD"] = os.path.basename(world_path)[:68]
    for k in ("t_ref", "alpha", "beta", "mu", "r_tube_max", "closed_rmax"):
        h[f"M_{k.upper()}"[:8]] = (float(opts[k]), f"model {k} (assumption)")
    h["M_STRAT"] = (bool(opts["stratify"]), "hydrostatic stratification")
    h["COMMENT"] = "EM(T) per pixel: sum over the line of sight of n_e n_H dl, binned by loop temperature."
    h["COMMENT"] = "Loop plasma: RTV scaling + heating ~ <B>^alpha L^-beta normalised to T_ref at the median loop."
    prim = fits.PrimaryHDU(cube.astype(np.float32), header=h)
    tot = fits.ImageHDU(cube.sum(axis=0).astype(np.float32), header=_wcs2(h), name="EM_TOTAL")
    tot.header["BUNIT"] = "cm-5"
    edges_hdu = fits.BinTableHDU.from_columns([fits.Column(name="LOGT_EDGE", format="D", array=edges)], name="LOGT")
    ids = np.array([f["id"] for f in fluxons])
    cols = [
        fits.Column(name="FLUXON", format="K", array=ids),
        fits.Column(name="CLOSED", format="L", array=plasma["closed"]),
        fits.Column(name="HALF_LENGTH", format="D", unit="cm", array=plasma["half_length_cm"]),
        fits.Column(name="B_REL", format="D", array=plasma["b_rel"]),
        fits.Column(name="HEAT_REL", format="D", array=plasma["heat_rel"]),
        fits.Column(name="PRESSURE", format="D", unit="dyn cm-2", array=plasma["p"]),
        fits.Column(name="T", format="D", unit="K", array=plasma["T"]),
        fits.Column(name="N_BASE", format="D", unit="cm-3", array=plasma["n_base"]),
    ]
    loops = fits.BinTableHDU.from_columns(cols, name="LOOPS")
    return fits.HDUList([prim, tot, edges_hdu, loops])


# ---------------------------------------------------------------------------------------------------- response


def load_response(path):
    """A temperature response from a two-column text file: log10 T (or T in K, detected by value) and R(T).
    Returns R(logT), zero outside the tabulated range. No response is built in: export one from SolarSoft
    (aia_get_response) or aiapy and pass the file."""
    a = np.loadtxt(path, comments="#", delimiter=None if not path.endswith(".csv") else ",")
    t, r = a[:, 0], a[:, 1]
    if np.nanmax(t) > 100:  # kelvin
        t = np.log10(t)
    o = np.argsort(t)
    t, r = t[o], r[o]
    return lambda lt: np.interp(lt, t, r, left=0.0, right=0.0)


def quicklook_png(hdus, path, ext="EM_TOTAL"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = hdus[ext].data.astype(float)
    pos = img[img > 0]
    fig, ax = plt.subplots(figsize=(7, 7))
    if pos.size:
        lo, hi = np.percentile(pos, [5, 99.7])
        ax.imshow(np.log10(np.clip(img, lo, None)), origin="lower", cmap="magma", vmin=np.log10(lo), vmax=np.log10(hi))
    ax.set_title(f"FLUXlight EUV {ext}  {hdus[0].header['DATE-OBS'][:16]}")
    ax.set_axis_off()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("world")
    ap.add_argument("--time", default=None, help="observation time (ISO); default: middle of the CR in the file name")
    ap.add_argument("--npix", type=int, default=1024)
    ap.add_argument("--scale", type=float, default=2.4, help="arcsec per pixel (2.4 = AIA binned 4x)")
    ap.add_argument("--response", action="append", default=[], help="NAME=FILE, a temperature response table")
    ap.add_argument("--t-ref", type=float, default=DEFAULTS["t_ref"])
    ap.add_argument("--alpha", type=float, default=DEFAULTS["alpha"])
    ap.add_argument("--beta", type=float, default=DEFAULTS["beta"])
    ap.add_argument("--no-stratify", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args(argv)
    responses = {}
    for spec in a.response:
        name, _, f = spec.partition("=")
        responses[name] = load_response(f)
    hdus, info = render(
        a.world,
        obstime=a.time,
        npix=a.npix,
        scale_arcsec=a.scale,
        responses=responses,
        t_ref=a.t_ref,
        alpha=a.alpha,
        beta=a.beta,
        stratify=not a.no_stratify,
    )
    out = a.out or a.world.replace(".flux", "_euv.fits")
    hdus.writeto(out, overwrite=True)
    print(f"wrote {out}: {info}")
    if a.png:
        quicklook_png(hdus, out.replace(".fits", ".png"))
        for name in responses:
            quicklook_png(hdus, out.replace(".fits", f"_{name}.png"), ext=f"SYN_{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
