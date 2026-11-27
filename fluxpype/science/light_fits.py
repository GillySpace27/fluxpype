"""FITS output and cache key for FLUXlight images (stack audit A7-03).

Each image is written as its own file, ``cr{CR}_f{N}_{inst}_{channel}.fits`` (names.light_fits), with a helioprojective
WCS an AIA map can be reprojected onto, so nobody has to unpickle a world to get the array back. Two producers feed it:

* the Thomson renderer in fluxlight.py (channels tb, pb, pfrac, pangle, coldens);
* the EUV renderer in fluxlight_euv.py (one channel per SYN_A<ch> image, named aia<ch>).

Pixel values are written exactly as the renderers computed them (no rescaling, no masking change). BUNIT is the unit
the renderer attached to the array, read from the array itself (Thomson) or from the SYN_A<ch> header (EUV).

``cache_key`` is the key of the ``.light`` cache: the sha256 of the world file plus every render parameter, so a changed
world or a changed observer geometry, resolution or channel list re-renders instead of reusing a stale result.
"""

import hashlib
import json
import os

import numpy as np
from astropy.io import fits

from fluxpype import names

# Thomson channels: key in the result dict -> (channel name, BUNIT when the array carries no unit).
# B_total, B_polarized, Column_Density are astropy Quantities (thomson_scattering_flux.py, simulate_thomson_scattering:
# erg cm-2 s-1 sr-1 and cm-2); Polarization_angle is 0.5 arctan2(...) in radian (the row is built as ``* u.radian`` and
# stored in a plain array, so its unit is not on the array); Polarization_fraction is a ratio.
THOMSON_CHANNELS = {
    "B_total": ("tb", None),
    "B_polarized": ("pb", None),
    "Polarization_fraction": ("pfrac", ""),
    "Polarization_angle": ("pangle", "rad"),
    "Column_Density": ("coldens", None),
}


def world_sha(path, chunk=1 << 20):
    """sha256 of the world file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def cache_key(world_path, params):
    """Key of a ``.light`` result: sha256 over the world sha and the render ``params`` (a dict of JSON-able values:
    observer geometry, resolution, channels, method...). Same world and parameters give the same key."""
    blob = json.dumps({"world": world_sha(world_path), "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _unit_str(arr, fallback):
    unit = getattr(arr, "unit", None)
    return unit.to_string("fits") if unit is not None else fallback


def _write(path, data, header, bunit, channel, world_path):
    hdr = fits.Header()
    for k, v in header.items():
        if k.lower() not in ("naxis1", "naxis2"):  # the array defines the axes
            hdr[k] = v
    hdr["BUNIT"] = (bunit, "renderer's unit")
    hdr["CHANNEL"] = channel
    if world_path:
        hdr["WORLD"] = os.path.basename(str(world_path))[:68]
    fits.PrimaryHDU(np.asarray(data, dtype=np.float32), header=hdr).writeto(path, overwrite=True)
    return path


def thomson_header(results, date=None, observer_lon_deg=None):
    """Helioprojective header for a Thomson image (observer_wcs.observer_header, so one WCS definition).

    The renderer looks along the equator from Carrington longitude ``observer_lon_deg`` (HGLT_OBS 0, B0 ignored), or
    pole-on along the rotation axis when it is None (HGLT_OBS 90). ``date`` defaults to the middle of the world's
    rotation (aia.task_time), so the header never silently carries the date the render happened to run."""
    from fluxpype.science.aia import task_time
    from fluxpype.science.observer_wcs import observer_header

    fov = results["fov"]
    fov = float(getattr(fov, "value", fov))
    cr = getattr(results["world"], "cr", None)
    t = task_time(int(cr) if str(cr).isdigit() else None, date)
    npix = np.asarray(results["B_total"]).shape[0]
    meta = observer_header(npix, fov, t.isot, 0.0 if observer_lon_deg is not None else 90.0)
    if observer_lon_deg is not None:
        meta["crln_obs"] = float(observer_lon_deg)  # Carrington longitude of the render's observer
    return meta


def write_thomson_fits(results, out_dir=None, date=None, observer_lon_deg=None):
    """Write one FITS per Thomson channel next to the PNGs. Returns the list of paths."""
    w = results["world"]
    out_dir = out_dir or results["light_path"]
    os.makedirs(out_dir, exist_ok=True)
    meta = thomson_header(results, date, observer_lon_deg)
    paths = []
    for key, (channel, fallback) in THOMSON_CHANNELS.items():
        if key not in results:
            continue
        arr = results[key]
        name = names.light_fits(w.cr, w.nflx, getattr(w, "inst", None), channel)
        data = arr.value if hasattr(arr, "value") else arr
        paths.append(_write(os.path.join(out_dir, name), data, meta, _unit_str(arr, fallback), channel, None))
    return paths


def write_euv_fits(hdus, out_dir, cr, nflux, inst, world_path=None):
    """Write one FITS per SYN_A<ch> image of a fluxlight_euv render. The data and the BUNIT are the ones in ``hdus``."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for h in hdus:
        if not h.name.startswith("SYN_A"):
            continue
        channel = "aia" + h.name[len("SYN_A") :]
        hdr = {
            k: h.header[k]
            for k in h.header
            if k
            not in (
                "XTENSION",
                "BITPIX",
                "NAXIS",
                "NAXIS1",
                "NAXIS2",
                "PCOUNT",
                "GCOUNT",
                "EXTNAME",
                "BUNIT",
                "COMMENT",
            )
        }
        path = os.path.join(out_dir, names.light_fits(cr, nflux, inst, channel))
        paths.append(_write(path, h.data, hdr, h.header["BUNIT"], channel, world_path))
    return paths
