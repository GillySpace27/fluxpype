"""A helioprojective FITS header for a FLUXlight image seen from Earth on a given date.

FLUXlight's image grid is ``np.linspace(-fov, fov, npix)`` in solar radii (thomson_scattering_flux.py), so the pixel
scale is 2 fov / (npix - 1) and the Sun's centre is at FITS pixel (npix + 1) / 2. The old header in
fluxlight.densplot used 2 fov / npix, crpix = npix / 2, no date (so the angular radius of the day the render ran),
B0 = 0 and the solar radius in arcsec where FITS wants metres (audit finding A2-04). Anything that reprojects a
model image onto AIA should take its header from here.
"""

from typing import Optional

import astropy.units as u
from astropy.constants import R_sun
from astropy.time import Time


def observer_header(npix: int, fov_rsun: float, date: Optional[str] = None, observer_lat_deg: float = 0.0) -> dict:
    """FITS header dict (sunpy Map meta) for an npix x npix FLUXlight image spanning +-fov_rsun solar radii.

    ``date`` is the observation time (any astropy Time string). Without it the current time is used, and DATE-OBS
    records that, so the choice is visible in the file rather than silent.

    ``observer_lat_deg`` is HGLT_OBS. It defaults to 0 because the Thomson renderer places the observer in the
    solar equatorial plane (it ignores B0, which is up to 7.25 deg). Pass ``sun.B0(date)`` only for an image that
    was rendered with that tilt, so the header always describes the image as it was made.
    """
    from sunpy.coordinates import sun

    t = Time(date) if date is not None else Time.now()
    rsun_arcsec = sun.angular_radius(t).to_value(u.arcsec)
    scale = 2.0 * float(fov_rsun) / (int(npix) - 1) * rsun_arcsec
    centre = (int(npix) + 1) / 2.0
    return {
        "date-obs": t.isot,
        "naxis1": int(npix),
        "naxis2": int(npix),
        "ctype1": "HPLN-TAN",
        "ctype2": "HPLT-TAN",
        "cunit1": "arcsec",
        "cunit2": "arcsec",
        "cdelt1": scale,
        "cdelt2": scale,
        "crpix1": centre,
        "crpix2": centre,
        "crval1": 0.0,
        "crval2": 0.0,
        "rsun_ref": R_sun.to_value(u.m),
        "rsun_obs": rsun_arcsec,
        "dsun_obs": sun.earth_distance(t).to_value(u.m),
        "hgln_obs": 0.0,
        "hglt_obs": float(observer_lat_deg),
        "obsrvtry": "Earth",
    }


def carrington_observer_lon(date: str) -> float:
    """Carrington longitude of the Earth-facing central meridian (L0) on ``date``, in degrees.

    This is the value FLUXlight's ``observer_lon`` should take for an image to be compared with an Earth-based
    observation at that time, instead of a hand-typed number.
    """
    from sunpy.coordinates import sun

    return float(sun.L0(Time(date)).to_value(u.deg))


def derived_observer_lon(configs: dict, default=None):
    """FLUXlight ``observer_lon`` for a run: from ``observation_date`` when the run sets it, else ``default`` unchanged.

    ``observation_date`` unset (absent, None or blank) leaves the hand-set ``observer_lon`` (or pole-on) exactly as it was,
    so runs that predate the "Compare with AIA" form keep their images. When it is set it wins over a typed
    ``observer_lon``: the time is the single input and the longitude follows from it.
    """
    t = configs.get("observation_date")
    if t is None or str(t).strip() in ("", "None"):
        return default
    return carrington_observer_lon(str(t).strip())
