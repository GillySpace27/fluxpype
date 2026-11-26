"""observer_wcs.observer_header: the header must put the Sun where FLUXlight's grid puts it (audit A2-04)."""

import astropy.units as u
import numpy as np
import pytest

sunpy = pytest.importorskip("sunpy")
from sunpy.coordinates import sun  # noqa: E402
from sunpy.map import Map  # noqa: E402

from fluxpype.science.observer_wcs import carrington_observer_lon, observer_header  # noqa: E402

DATE = "2014-05-18T20:00:00"


@pytest.mark.parametrize("npix,fov", [(61, 3.0), (401, 3.0), (300, 215.0)])
def test_header_matches_fluxlight_grid(npix, fov):
    m = Map(np.zeros((npix, npix)), observer_header(npix, fov, DATE))
    # Solar radius in the right unit and for the right day.
    assert m.rsun_obs.to_value(u.arcsec) == pytest.approx(sun.angular_radius(DATE).to_value(u.arcsec), rel=1e-6)
    assert m.rsun_meters.to_value(u.m) == pytest.approx(695_700_000.0, rel=1e-9)
    # Grid: pixel (npix-1)/2 (0-based) is disk centre and pixel 0 is -fov solar radii.
    c = m.pixel_to_world(((npix - 1) / 2) * u.pix, ((npix - 1) / 2) * u.pix)
    assert abs(c.Tx.to_value(u.arcsec)) < 1e-6 and abs(c.Ty.to_value(u.arcsec)) < 1e-6
    edge = m.pixel_to_world(0 * u.pix, ((npix - 1) / 2) * u.pix)
    rsun = sun.angular_radius(DATE).to_value(u.arcsec)
    if fov <= 5:  # TAN projection is close to linear only near the Sun; at 215 R_sun (57 deg) it is not
        assert edge.Tx.to_value(u.arcsec) == pytest.approx(-fov * rsun, rel=1e-3)
    assert str(m.date.isot).startswith("2014-05-18T20:00")


def test_carrington_observer_lon_matches_profile_comment():
    # config.ini's [light] profile hand-types observer_lon = 172.0 for this date.
    assert carrington_observer_lon(DATE) == pytest.approx(171.95, abs=0.01)
