"""FLUXlight EUV: render a FLUX world as a per-pixel emission-measure cube EM(T), matched to an observer and time.

The renderer has three separable parts, so each can be replaced without touching the others:

1. Geometry (this module): each fluxon is a flux tube. Its local cross-section comes from the distance to the
   nearest vertex on another fluxon (radius = half that distance, capped at ``r_tube_max``), and B is the fluxon's
   flux over that area, in relative units. Tube samples are projected into a helioprojective image for an observer
   at Earth at ``obstime``; samples hidden behind the photosphere are dropped.
2. Plasma (``loop_plasma``): each closed fluxon gets T(s) and n_e(s) from the steady loop model of Cranmer & Gilly
   (2026) (``loop_model.LoopModel``: Martens 2010 scaling with their conductivity, cooling curve, heating exponents
   m, n and normalization T_norm, plus hydrostatic pressure). Its inputs per fluxon are the half-length L and the
   basal field B_sun, the geometric mean of the two footpoint fields Phi / (pi r^2). FLUX world files carry fluxon
   flux in relative units, so the flux per fluxon in Mx is a parameter (``flux_per_fluxon``; the default is an
   assumption). Open fluxons emit nothing yet.
3. Emissivity: the output is EM(T) per pixel (EM = n_e n_H dl in cm^-5, per log T bin), independent of any atomic
   model. With an AIA table from ``aia_response`` (K(T, n_e) from ChiantiPy multi-level spectra), each sample also
   adds K(T, n_e) EM to a synthetic channel image (SYN_A<channel>). A temperature-only response R(T)
   (``load_response``) is contracted with the EM cube instead.

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
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from fluxpype.plumber.fluxfile import parse_text

RSUN_CM = const.R_sun.to_value(u.cm)
K_B = const.k_B.cgs.value
M_P = const.m_p.cgs.value
G_SUN = (const.G * const.M_sun / const.R_sun**2).cgs.value

DEFAULTS = dict(
    m=1.607,  # heating exponents and normalization: Cranmer & Gilly (2026) Table 3, 2018 (all), t_n = 0
    n=1.144,
    t_norm=1.487e6,
    flux_per_fluxon=1.0e20,  # Mx per fluxon (assumption; set it from the magnetogram and fluxon count)
    nh_ne=0.83,  # n_H / n_e for a fully ionized coronal plasma (assumption)
    r_tube_max=0.05,  # R_sun, cap on tube radius (assumption)
    closed_rmax=1.05,  # a fluxon is closed when both ends lie below this radius (R_sun)
)
X_GRID = np.concatenate([[0.0], np.logspace(-7, 0, 1500)])  # s / L samples for the loop profiles


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
    mids, dirs, lens, own, s0 = [], [], [], [], []
    for k, f in enumerate(fluxons):
        p = f["xyz"]
        d = np.diff(p, axis=0)
        ln = np.linalg.norm(d, axis=1)
        ok = ln > 0
        mids.append(0.5 * (p[1:] + p[:-1])[ok])
        dirs.append((d[ok] / ln[ok, None]))
        lens.append(ln[ok])
        own.append(np.full(ok.sum(), k))
        s0.append((np.cumsum(ln) - ln)[ok])  # arc length at each segment's start, R_sun
    mids, dirs, lens, own, s0 = map(np.concatenate, (mids, dirs, lens, own, s0))
    # nearest midpoint that belongs to another fluxon
    kq = min(len(mids), 24)
    dist, idx = cKDTree(mids).query(mids, k=kq)
    other = own[idx] != own[:, None]
    first = np.where(other.any(axis=1), other.argmax(axis=1), kq - 1)
    dnn = dist[np.arange(len(mids)), first]
    radius = np.clip(0.5 * dnn, 1e-4, r_tube_max)
    return {"mid": mids, "dir": dirs, "len": lens, "radius": radius, "owner": own, "s0": s0}


# ---------------------------------------------------------------------------------------------------- plasma


def loop_plasma(fluxons, geom, m=1.607, n=1.144, t_norm=1.487e6, flux_per_fluxon=1.0e20, closed_rmax=1.05, **_):
    """Per-fluxon loop solutions. Returns a dict of arrays, one entry per fluxon: closed, half_length_cm, b_sun (G),
    delta, T_max (K), P0 (dyn cm^-2), and the profiles T_grid (K) and ne_grid (cm^-3) on ``X_GRID`` = s / L from
    the nearer footpoint. Open fluxons get zeros."""
    from fluxpype.science.loop_model import LoopModel, delta_of_length

    model = LoopModel(m=m, n=n, t_norm=t_norm)
    nf = len(fluxons)
    own, ln, rad = geom["owner"], geom["len"], geom["radius"]
    length = np.bincount(own, weights=ln, minlength=nf)
    r0 = np.array([np.linalg.norm(f["xyz"][0]) for f in fluxons])
    r1 = np.array([np.linalg.norm(f["xyz"][-1]) for f in fluxons])
    closed = (r0 < closed_rmax) & (r1 < closed_rmax)
    half = 0.5 * length * RSUN_CM
    # footpoint field: flux over the area of the first and last segment's tube
    first = np.full(nf, -1)
    last = np.full(nf, -1)
    idx = np.arange(len(own))
    np.maximum.at(last, own, idx)
    first[own[::-1]] = idx[::-1]
    flux = np.array([f["flux"] for f in fluxons]) * flux_per_fluxon
    area = lambda k: np.pi * (rad[k] * RSUN_CM) ** 2  # noqa: E731
    b_sun = np.where(closed, flux / np.sqrt(area(first) * area(last)), 0.0)
    out = dict(closed=closed, half_length_cm=half, b_sun=b_sun)
    for k in ("delta", "T_max", "P0"):
        out[k] = np.zeros(nf)
    out["T_grid"] = np.zeros((nf, len(X_GRID)))
    out["ne_grid"] = np.zeros((nf, len(X_GRID)))
    for k in np.nonzero(closed)[0]:
        sol = model.apex(half[k], b_sun[k], delta_of_length(half[k]))
        if not (np.isfinite(sol["T_max"]) and sol["T_max"] > 2 * 1e4):
            # no heating left (e.g. sunspot quenching of v_perp at very strong footpoint field): no corona
            closed[k] = False
            continue
        T, ne, sol = model.profile(X_GRID * half[k], half[k], b_sun[k], delta_of_length(half[k]))
        out["delta"][k], out["T_max"][k], out["P0"][k] = sol["delta"], sol["T_max"], sol["P0"]
        out["T_grid"][k], out["ne_grid"][k] = T, ne
    return out


def sample_plasma(plasma, owner, s_rsun):
    """T and n_e at arc length ``s_rsun`` (from the fluxon's first vertex) for samples of fluxons ``owner``."""
    L = plasma["half_length_cm"][owner] / RSUN_CM
    s = np.clip(s_rsun, 0.0, 2 * L)
    x = np.minimum(s, 2 * L - s) / np.maximum(L, 1e-30)
    j = np.clip(np.searchsorted(X_GRID, x) - 1, 0, len(X_GRID) - 2)
    w = (x - X_GRID[j]) / (X_GRID[j + 1] - X_GRID[j])
    lt = (
        np.log(np.maximum(plasma["T_grid"][owner, j], 1.0)) * (1 - w)
        + np.log(np.maximum(plasma["T_grid"][owner, j + 1], 1.0)) * w
    )
    ln_ = (
        np.log(np.maximum(plasma["ne_grid"][owner, j], 1e-30)) * (1 - w)
        + np.log(np.maximum(plasma["ne_grid"][owner, j + 1], 1e-30)) * w
    )
    return np.exp(lt), np.exp(ln_)


def segment_emission(plasma, geom, segs, logt_edges, nh_ne=0.83, table=None):
    """Exact (on the profile grid) emission of each segment in ``segs``: EM volume integral pi r^2 int n_e n_H ds
    per log T bin, shape (len(segs), nT) in cm^-3, and, with an AIA ``table``, int K(T, n_e) n_e n_H dV per channel,
    shape (len(segs), nchan) in DN s^-1 cm^2 pix^-1. Integrating the profile instead of point-sampling it keeps the
    thin, dense transition region from turning into shot noise."""
    nt = len(logt_edges) - 1
    chans = sorted(table.K) if table is not None else []
    em = np.zeros((len(segs), nt))
    syn = np.zeros((len(segs), len(chans)))
    own = geom["owner"][segs]
    for k in np.unique(own):
        rows = np.nonzero(own == k)[0]
        L = plasma["half_length_cm"][k]
        T, ne = plasma["T_grid"][k], plasma["ne_grid"][k]
        f = nh_ne * ne * ne
        dx = np.diff(X_GRID)
        fm = 0.5 * (f[1:] + f[:-1]) * dx * L  # cm^-6 * cm per interval
        tb = np.digitize(np.log10(np.sqrt(T[1:] * T[:-1])), logt_edges) - 1
        cum = np.zeros((nt, len(X_GRID)))
        good = (tb >= 0) & (tb < nt)
        contrib = np.zeros((nt, len(dx)))
        contrib[tb[good], np.nonzero(good)[0]] = fm[good]
        cum[:, 1:] = np.cumsum(contrib, axis=1)
        if chans:
            kk = np.array([table(c, T, ne) for c in chans])  # (nchan, nx)
            g = kk * f
            gm = 0.5 * (g[:, 1:] + g[:, :-1]) * dx * L * good  # same temperature range as the EM cube
            cumk = np.zeros((len(chans), len(X_GRID)))
            cumk[:, 1:] = np.cumsum(gm, axis=1)
        Lr = L / RSUN_CM
        for i in rows:
            j = segs[i]
            sa, sb = geom["s0"][j], geom["s0"][j] + geom["len"][j]
            pieces = []  # (x_lo, x_hi) in folded coordinate
            if sb <= Lr:
                pieces.append((sa / Lr, sb / Lr))
            elif sa >= Lr:
                pieces.append(((2 * Lr - sb) / Lr, (2 * Lr - sa) / Lr))
            else:
                pieces += [(sa / Lr, 1.0), ((2 * Lr - sb) / Lr, 1.0)]
            area = np.pi * (geom["radius"][j] * RSUN_CM) ** 2
            for lo, hi in pieces:
                lo, hi = np.clip([lo, hi], 0.0, 1.0)
                em[i] += area * np.array([np.interp(hi, X_GRID, c) - np.interp(lo, X_GRID, c) for c in cum])
                if chans:
                    syn[i] += area * np.array([np.interp(hi, X_GRID, c) - np.interp(lo, X_GRID, c) for c in cumk])
    return em, syn, chans


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
    aia_table=None,
    **model,
):
    """Render a world. Returns (hdulist, info). ``responses`` maps a name to a callable R(logT) in
    DN cm^5 s^-1 pix^-1 (or any per-EM unit), applied to the EM cube. ``aia_table`` is a K(T, n_e) file from
    ``aia_response build``; each of its channels becomes a SYN_A<channel> image in DN s^-1 per AIA pixel."""
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
    table = None
    if aia_table is not None:
        from fluxpype.science.aia_response import Response

        table = Response(aia_table)
        syn = {c: np.zeros((ny, nx)) for c in sorted(table.K)}

    seg_on = plasma["closed"][geom["owner"]]
    sel = np.nonzero(seg_on)[0]
    em_seg, syn_seg, chans = segment_emission(plasma, geom, sel, logt_edges, opts["nh_ne"], table)
    cube_flat = cube.reshape(nt, -1)
    syn_flat = np.zeros((len(chans), ny * nx))
    n_kept = n_samples = 0
    for chunk_idx in np.array_split(np.arange(len(sel)), max(1, len(sel) // 20000)):
        if not len(chunk_idx):
            continue
        chunk = sel[chunk_idx]
        mid, d, ln, rad = geom["mid"][chunk], geom["dir"][chunk], geom["len"][chunk], geom["radius"][chunk]
        m = np.maximum(1, np.ceil(ln / (0.5 * pix_rsun)).astype(int))
        rep = np.repeat(np.arange(len(chunk)), m * n_disc)
        k_along = np.concatenate([np.arange(k) for k in m]).repeat(n_disc)
        frac = (k_along + rng.random(len(rep))) / m[rep] - 0.5
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
        # tube offsets near a tilted footpoint can dip below the photosphere; lift those samples onto it so the
        # (dense, bright) base of the loop is not lost
        low = r < 1.0
        pts[low] *= (1.0 + 1e-9) / r[low, None]
        hcc = pts @ rot  # rot rows are the heliocentric images of the Carrington x, y, z unit vectors
        keep = ~occulted(hcc, dsun)
        tx, ty = to_helioprojective(hcc[keep], dsun)
        ix = np.floor(tx / scale_arcsec + nx / 2).astype(int)
        iy = np.floor(ty / scale_arcsec + ny / 2).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        seg_local = rep[keep][ok]
        pix = iy[ok] * nx + ix[ok]
        share = 1.0 / (m[seg_local] * n_disc) / pix_area_cm2  # each sample carries this fraction of its segment
        W = coo_matrix((share, (seg_local, pix)), shape=(len(chunk), ny * nx)).tocsr()
        cube_flat += (W.T @ em_seg[chunk_idx]).T
        if chans:
            syn_flat += (W.T @ syn_seg[chunk_idx]).T
        n_samples += len(rep)
        n_kept += int(ok.sum())
    if table is not None:
        syn = {c: syn_flat[i].reshape(ny, nx) for i, c in enumerate(chans)}

    hdus = _to_fits(cube, logt_edges, scale_arcsec, obstime, earth, dsun, plasma, fluxons, world_path, opts)
    if responses:
        tc = 0.5 * (logt_edges[1:] + logt_edges[:-1])
        for name, fn in responses.items():
            img = np.tensordot(fn(tc), cube, axes=(0, 0))
            h = fits.ImageHDU(img.astype(np.float32), header=_wcs2(hdus[0].header), name=f"SYN_{name}"[:68])
            h.header["BUNIT"] = "response unit x cm-5 (e.g. DN s-1 pix-1)"
            h.header["COMMENT"] = f"sum over log T of EM(T) * R_{name}(T)"
            hdus.append(h)
    if table is not None:
        for c, img in syn.items():
            h = fits.ImageHDU(img.astype(np.float32), header=_wcs2(hdus[0].header), name=f"SYN_A{c}")
            h.header["BUNIT"] = "DN s-1 pix-1"
            h.header["COMMENT"] = "sum of K(T, n_e) n_e n_H dl over the line of sight; per AIA 0.6 arcsec pixel"
            h.header["AIATABLE"] = os.path.basename(aia_table)[:68]
            hdus.append(h)
    info = dict(
        obstime=obstime.isot,
        n_fluxons=len(fluxons),
        n_closed=int(plasma["closed"].sum()),
        samples=n_samples,
        samples_in_image=n_kept,
        T_max_range=(
            (float(plasma["T_max"][plasma["closed"]].min()), float(plasma["T_max"][plasma["closed"]].max()))
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
    for key, k in (
        ("M_HEATM", "m"),
        ("M_HEATN", "n"),
        ("M_TNORM", "t_norm"),
        ("M_FLUX", "flux_per_fluxon"),
        ("M_NHNE", "nh_ne"),
        ("M_RTUBE", "r_tube_max"),
        ("M_CLOSED", "closed_rmax"),
    ):
        h[key] = (float(opts[k]), f"model {k}")
    h["COMMENT"] = "EM(T) per pixel: sum over the line of sight of n_e n_H dl, binned by local temperature."
    h["COMMENT"] = "Loop plasma: Cranmer & Gilly (2026) steady loops (fluxpype.science.loop_model)."
    prim = fits.PrimaryHDU(cube.astype(np.float32), header=h)
    tot = fits.ImageHDU(cube.sum(axis=0).astype(np.float32), header=_wcs2(h), name="EM_TOTAL")
    tot.header["BUNIT"] = "cm-5"
    edges_hdu = fits.BinTableHDU.from_columns([fits.Column(name="LOGT_EDGE", format="D", array=edges)], name="LOGT")
    ids = np.array([f["id"] for f in fluxons])
    cols = [
        fits.Column(name="FLUXON", format="K", array=ids),
        fits.Column(name="CLOSED", format="L", array=plasma["closed"]),
        fits.Column(name="HALF_LENGTH", format="D", unit="cm", array=plasma["half_length_cm"]),
        fits.Column(name="B_SUN", format="D", unit="G", array=plasma["b_sun"]),
        fits.Column(name="DELTA", format="D", array=plasma["delta"]),
        fits.Column(name="T_MAX", format="D", unit="K", array=plasma["T_max"]),
        fits.Column(name="P0", format="D", unit="dyn cm-2", array=plasma["P0"]),
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
    ap.add_argument("--aia-table", default=None, help="K(T, n_e) FITS from 'python -m fluxpype.science.aia_response'")
    ap.add_argument("--m", type=float, default=DEFAULTS["m"], help="heating exponent m (Cranmer & Gilly 2026)")
    ap.add_argument("--n", type=float, default=DEFAULTS["n"], help="heating exponent n")
    ap.add_argument("--t-norm", type=float, default=DEFAULTS["t_norm"], help="K, apex T of the reference loop")
    ap.add_argument("--flux-per-fluxon", type=float, default=DEFAULTS["flux_per_fluxon"], help="Mx")
    ap.add_argument("--out", default=None)
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--channel-fits", default=None, help="DIR: also write cr{CR}_f{N}_{inst}_aia{ch}.fits per channel")
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
        aia_table=a.aia_table,
        m=a.m,
        n=a.n,
        t_norm=a.t_norm,
        flux_per_fluxon=a.flux_per_fluxon,
    )
    out = a.out or a.world.replace(".flux", "_euv.fits")
    hdus.writeto(out, overwrite=True)
    print(f"wrote {out}: {info}")
    if a.channel_fits:
        from fluxpype import names
        from fluxpype.science.light_fits import write_euv_fits

        base = os.path.basename(a.world)
        cr, nf = _cr_from_name(a.world), (names._TASK_IN_NAME.search(base) or [None, None])[1]
        for p in write_euv_fits(hdus, a.channel_fits, cr, nf, names.inst_in_name(base), a.world):
            print(f"wrote {p}")
    if a.png:
        quicklook_png(hdus, out.replace(".fits", ".png"))
        names = [h.name for h in hdus if h.name.startswith("SYN_")]
        for name in names:
            quicklook_png(hdus, out.replace(".fits", f"_{name[4:]}.png"), ext=name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
