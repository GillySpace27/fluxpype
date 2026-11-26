"""AIA comparison helpers, offline (no downloads)."""

import numpy as np
import pytest

pytest.importorskip("sunpy")
pytest.importorskip("reproject")

from astropy.io import fits  # noqa: E402

from fluxpype.science import aia  # noqa: E402


def test_task_time_mid_rotation():
    t = aia.task_time(cr=2150)
    assert t.isot.startswith("2014-05-18")
    assert aia.task_time(time="2020-06-01T00:00").isot.startswith("2020-06-01T00:00")


def _hpc_header(n, scale, date="2020-06-01T00:00:00"):
    h = fits.Header()
    h["CTYPE1"], h["CUNIT1"], h["CRPIX1"], h["CRVAL1"], h["CDELT1"] = "HPLN-TAN", "arcsec", n / 2 + 0.5, 0.0, scale
    h["CTYPE2"], h["CUNIT2"], h["CRPIX2"], h["CRVAL2"], h["CDELT2"] = "HPLT-TAN", "arcsec", n / 2 + 0.5, 0.0, scale
    h["DATE-OBS"] = date
    h["HGLN_OBS"], h["HGLT_OBS"], h["DSUN_OBS"] = 0.0, -0.3, 1.5e11
    h["RSUN_REF"] = 6.957e8
    return h


def test_compare_constant_images(tmp_path):
    m = np.zeros((100, 100))
    m[30:70, 30:70] = 5.0
    model = tmp_path / "model.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(m, header=_hpc_header(100, 8.0), name="SYN_A193")]).writeto(model)
    ah = _hpc_header(800, 1.0)
    ah["INSTRUME"], ah["TELESCOP"], ah["WAVELNTH"], ah["WAVEUNIT"] = "AIA_3", "SDO/AIA", 193, "angstrom"
    aiaf = tmp_path / "aia.fits"
    fits.writeto(aiaf, np.full((800, 800), 10.0, np.float32), ah)
    out, s = aia.compare(str(model), str(aiaf), 193, png=False)
    assert s["pixels_compared"] == 1600
    assert s["model_over_aia_total"] == pytest.approx(0.5, rel=1e-3)
    assert s["median_log_ratio"] == pytest.approx(np.log10(0.5), abs=1e-3)
    with fits.open(out) as h:
        assert np.nanmax(h["AIA_REPROJ"].data) == pytest.approx(10.0, rel=1e-3)
