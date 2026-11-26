"""AIA K(T, n_e) folding and lookup, with a fake channel response (no aiapy or ChiantiPy needed)."""

import numpy as np
import pytest

from fluxpype.science import aia_response as ar


def test_fold_single_line(monkeypatch):
    monkeypatch.setattr(ar, "channel_response", lambda c: (np.array([190.0, 200.0]), np.array([2.0, 2.0]), 1e-11))
    wvl = np.arange(190.0, 200.0, 0.01)
    spec = np.zeros((1, wvl.size))
    i = np.argmin(abs(wvl - 195.0))
    spec[0, i] = 1e-24 / 0.01  # 1e-24 erg cm^-2 s^-1 sr^-1 in one bin
    k = ar.fold(wvl, spec, channels=(193,))[193]
    photons = 1e-24 * (wvl[i] * 1e-8) / (ar.H_ERG_S * ar.C_CMS)
    assert k[0] == pytest.approx(1e-11 * 2.0 * photons, rel=1e-6)


def test_response_lookup(tmp_path):
    from astropy.io import fits

    logt, logne = np.array([6.0, 6.1, 6.2]), np.array([8.0, 9.0])
    K = np.outer([1.0, 2.0], [1.0, 3.0, 5.0])
    hdus = [fits.PrimaryHDU(), fits.ImageHDU(K, name="A193")]
    for name, arr in (("LOGT", logt), ("LOGNE", logne)):
        hdus.append(fits.BinTableHDU.from_columns([fits.Column(name=name, format="D", array=arr)], name=name))
    p = tmp_path / "k.fits"
    fits.HDUList(hdus).writeto(p)
    r = ar.Response(str(p))
    assert r(193, 10**6.05, 1e8) == pytest.approx(2.0)
    assert r(193, 10**6.1, 10**8.5) == pytest.approx(4.5)
    assert r(193, 1e7, 1e12) == pytest.approx(10.0)  # clamped
