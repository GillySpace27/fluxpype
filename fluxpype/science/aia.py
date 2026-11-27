"""AIA side of the FLUXlight EUV comparison (stack audit A2-05): task time, fetch, prep and comparison.

    python -m fluxpype.science.aia fetch --cr 2150 --channel 193 [--time ISO] [--out DIR]
    python -m fluxpype.science.aia compare MODEL.fits AIA.fits [--channel 193] [--out OUT]

``fetch`` gets the level 1 image nearest to the time through sunpy Fido (VSO), falling back to the JSOC 1024^2
synoptic product over HTTPS, then preps it: aiapy ``update_pointing`` and ``register`` where possible, exposure
normalization to DN s^-1, and aiapy ``correct_degradation``. Steps that need JSOC tables are skipped with a warning
and recorded in the header (PREPSTEP), so a comparison can be read knowing what was done.

``compare`` reprojects the AIA image onto the model image's helioprojective grid (adaptive resampling, which
averages when the model pixels are larger), so both are DN s^-1 per AIA pixel, and writes AIA_REPROJ, MODEL and
LOG_RATIO images plus a PNG and a few summary numbers.
"""

import argparse
import os
import warnings

import numpy as np
from astropy.io import fits

JSOC_SYNOPTIC = (
    "https://jsoc1.stanford.edu/data/aia/synoptic/{t:%Y/%m/%d}/H{t:%H}00/AIA{t:%Y%m%d_%H}{m:02d}_{ch:04d}.fits"
)


def task_time(cr=None, time=None):
    """The observation time of a task: ``time`` if given, else the middle of Carrington rotation ``cr``."""
    from astropy.time import Time

    if time is not None:
        return Time(time)
    if cr is None:
        raise ValueError("need a Carrington rotation or an explicit time")
    import sunpy.coordinates.sun as sun

    return sun.carrington_rotation_time(int(cr) + 0.5)


def _fido_level1(t, channel, outdir, window_s=30):
    import astropy.units as u
    from sunpy.net import Fido
    from sunpy.net import attrs as a

    res = Fido.search(a.Time(t - window_s * u.s, t + window_s * u.s), a.Instrument.aia, a.Wavelength(channel * u.AA))
    if not len(res) or not len(res[0]):
        return None
    times = res[0]["Start Time"]
    i = int(np.argmin(np.abs((times - t).to_value(u.s))))
    files = Fido.fetch(res[0, i], path=os.path.join(outdir, "{file}"), progress=False)
    return files[0] if len(files) else None


def _jsoc_synoptic(t, channel, outdir):
    import urllib.request

    m = int(round(t.datetime.minute / 2.0)) * 2 % 60  # synoptic images are every 2 minutes
    url = JSOC_SYNOPTIC.format(t=t.datetime, m=m, ch=channel)
    path = os.path.join(outdir, os.path.basename(url))
    urllib.request.urlretrieve(url, path)
    return path


def fetch_aia(time, channel=193, outdir="aia"):
    """Download the AIA image nearest ``time`` and return the path of a prepped level 1.5 FITS file."""
    os.makedirs(outdir, exist_ok=True)
    raw = None
    errors = []
    for getter in (_fido_level1, _jsoc_synoptic):
        try:
            raw = getter(time, channel, outdir)
            if raw:
                break
        except Exception as e:  # network or provider failure: try the next source
            errors.append(f"{getter.__name__}: {e}")
    if not raw:
        raise RuntimeError("no AIA file could be downloaded: " + "; ".join(errors))
    return prep_aia(raw, os.path.join(outdir, f"aia_lev15_{channel}_{time.datetime:%Y%m%dT%H%M%S}.fits"))


def prep_aia(path, out):
    """Pointing, registration, exposure normalization and degradation correction, as far as possible offline."""
    import astropy.units as u
    import sunpy.map

    m = sunpy.map.Map(path)
    steps = []
    try:
        from aiapy.calibrate import correct_degradation, register, update_pointing
        from aiapy.calibrate.utils import get_pointing_table
    except ImportError:
        warnings.warn("aiapy not installed: no pointing, registration or degradation correction")
        update_pointing = None
    if update_pointing is not None:
        try:
            ptab = get_pointing_table("jsoc", time_range=(m.date - 6 * u.h, m.date + 6 * u.h))
            m = update_pointing(m, pointing_table=ptab)
            steps.append("pointing")
        except Exception as e:
            warnings.warn(f"pointing update skipped: {e}")
        if m.data.shape[0] >= 4096:  # synoptic images are already registered and binned
            m = register(m)
            steps.append("register")
        try:
            m = correct_degradation(m)
            steps.append("degradation")
        except Exception as e:
            warnings.warn(f"degradation correction skipped: {e}")
    exptime = m.exposure_time.to_value(u.s) if m.exposure_time is not None else 1.0
    data = m.data / exptime
    # synoptic or binned images have larger pixels; express per 0.6 arcsec AIA pixel to match the model unit
    pix = abs(m.scale.axis1.to_value(u.arcsec / u.pix))
    data = data * (0.6 / pix) ** 2
    hdr = m.fits_header
    hdr["BUNIT"] = "DN s-1 pix-1"
    hdr["COMMENT"] = "per 0.6 arcsec AIA pixel"
    hdr["PREPSTEP"] = ",".join(steps + ["exposure"]) or "exposure"
    fits.writeto(out, data.astype(np.float32), hdr, overwrite=True)
    return out


def compare(model_path, aia_path, channel=193, out=None, png=True):
    """Reproject AIA onto the model's grid and write AIA_REPROJ, MODEL and LOG_RATIO. Returns (out, summary)."""
    import sunpy.map
    from astropy.wcs import WCS

    with fits.open(model_path) as h:
        ext = f"SYN_A{channel}"
        model = h[ext].data.astype(float)
        mhdr = h[ext].header
    aia = sunpy.map.Map(aia_path)
    target = WCS(mhdr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = aia.reproject_to(target, algorithm="adaptive", return_footprint=False)
    a = np.asarray(rep.data, float)
    ok = np.isfinite(a) & (a > 0) & (model > 0)
    # Constant images give a NaN correlation by design; cranmer_wind/tempest set np.seterr(...="raise") at import and
    # leak it process-wide (pytest -n auto workers), so state the error handling here rather than inherit it.
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.where(ok, np.log10(model / a), np.nan)
        summary = dict(
            pixels_compared=int(ok.sum()),
            model_over_aia_total=float(model[ok].sum() / a[ok].sum()) if ok.any() else float("nan"),
            median_log_ratio=float(np.nanmedian(lr)) if ok.any() else float("nan"),
            log_correlation=(
                float(np.corrcoef(np.log10(model[ok]), np.log10(a[ok]))[0, 1]) if ok.sum() > 2 else float("nan")
            ),
        )
    out = out or model_path.replace(".fits", f"_vs_aia{channel}.fits")
    hdus = [fits.PrimaryHDU()]
    for name, img in (("AIA_REPROJ", a), ("MODEL", model), ("LOG_RATIO", lr)):
        h = fits.ImageHDU(np.asarray(img, np.float32), header=mhdr.copy(), name=name)
        h.header["BUNIT"] = "" if name == "LOG_RATIO" else "DN s-1 pix-1"
        hdus.append(h)
    for k, v in summary.items():
        hdus[0].header[k[:8].upper()] = (v if np.isfinite(v) else "nan", k)
    hdus[0].header["AIAFILE"] = os.path.basename(aia_path)[:68]
    hdus[0].header["MODFILE"] = os.path.basename(model_path)[:68]
    fits.HDUList(hdus).writeto(out, overwrite=True)
    if png:
        _png(a, model, lr, out.replace(".fits", ".png"), channel)
    return out, summary


def _png(a, model, lr, path, channel):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    both = np.concatenate([a[np.isfinite(a) & (a > 0)], model[model > 0]])
    lo, hi = np.percentile(both, [5, 99.7]) if both.size else (1, 10)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for k, (img, title) in enumerate(((a, f"AIA {channel}"), (model, f"FLUXlight {channel}"))):
        ax[k].imshow(
            np.log10(np.clip(np.nan_to_num(img), lo, None)),
            origin="lower",
            cmap="sdoaia%d" % channel if f"sdoaia{channel}" in plt.colormaps() else "magma",
            vmin=np.log10(lo),
            vmax=np.log10(hi),
        )
        ax[k].set_title(f"{title} (log DN/s, same scale)")
    ax[2].imshow(lr, origin="lower", cmap="RdBu_r", vmin=-2, vmax=2)
    ax[2].set_title("log10 model / AIA")
    for x in ax:
        x.set_axis_off()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--cr", type=int)
    f.add_argument("--time")
    f.add_argument("--channel", type=int, default=193)
    f.add_argument("--out", default="aia")
    c = sub.add_parser("compare")
    c.add_argument("model")
    c.add_argument("aia")
    c.add_argument("--channel", type=int, default=193)
    c.add_argument("--out")
    a = p.parse_args(argv)
    if a.cmd == "fetch":
        print(fetch_aia(task_time(a.cr, a.time), a.channel, a.out))
    else:
        out, s = compare(a.model, a.aia, a.channel, a.out)
        print(out, s)


if __name__ == "__main__":
    main()
