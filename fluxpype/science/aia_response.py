"""AIA temperature and density response K(T, n_e) from ChiantiPy multi-level spectra.

K is in DN cm^5 s^-1 pix^-1, per unit emission measure EM = integral n_e n_H dl (ChiantiPy's ``em`` convention),
so a pixel's count rate is sum over the line of sight of K(T, n_e) n_e n_H dl.

Build (slow, minutes per density; needs ChiantiPy, $XUVTOP and aiapy):
    python -m fluxpype.science.aia_response build out.fits --logt 5.5 7.5 0.05 --logne 8 9 10 11

The spectrum includes lines and continuum (ChiantiPy ``mspectrum``), CHIANTI's default ionization equilibrium and
the abundance file set in the ChiantiPy rc. Channel responses come from aiapy ``Channel.wavelength_response`` at
launch (no time-dependent degradation).
"""

import argparse

import numpy as np

H_ERG_S = 6.62607015e-27
C_CMS = 2.99792458e10
CHANNELS = (94, 131, 171, 193, 211, 335)


def channel_response(channel):
    """(wavelength A, response cm^2 DN ph^-1, plate scale sr pix^-1) for one AIA EUV channel, from aiapy."""
    import astropy.units as u
    from aiapy.response import Channel

    c = Channel(channel * u.angstrom)
    return (
        c.wavelength.to_value(u.angstrom),
        c.wavelength_response().to_value(u.cm**2 * u.DN / u.ph),
        c.plate_scale.to_value(u.sr / u.pix),
    )


def fold(wvl, spectrum, channels=CHANNELS):
    """Fold ChiantiPy spectra (erg cm^-2 s^-1 sr^-1 A^-1 per unit EM, shape (nT, nwvl)) through AIA responses.
    Returns dict channel -> K(T), DN cm^5 s^-1 pix^-1."""
    dl = np.gradient(wvl)
    photons = spectrum * (wvl * 1e-8) / (H_ERG_S * C_CMS)  # ph cm^-2 s^-1 sr^-1 A^-1
    out = {}
    for ch in channels:
        w, r, ps = channel_response(ch)
        out[ch] = ps * np.sum(photons * np.interp(wvl, w, r, left=0, right=0) * dl, axis=-1)
    return out


def build(path, logt, logne, wmin=80.0, wmax=380.0, dw=0.05, min_abund=1e-6, proc=4, channels=CHANNELS):
    """Compute K(T, n_e) on the grid and write a FITS file: one image HDU per channel, shape (n_logne, n_logt),
    with LOGT and LOGNE in a table HDU."""
    import ChiantiPy.core as ch
    import ChiantiPy.tools.filters as chf
    from astropy.io import fits

    logt, logne = np.asarray(logt, float), np.asarray(logne, float)
    wvl = np.arange(wmin, wmax, dw)
    T = 10**logt
    K = {c: np.zeros((len(logne), len(logt))) for c in channels}
    for j, ne in enumerate(10**logne):
        s = ch.mspectrum(
            T,
            np.full_like(T, ne),
            wvl,
            filter=(chf.gaussian, 2 * dw),
            minAbund=min_abund,
            em=np.ones_like(T),
            proc=proc,
        )
        for c, k in fold(wvl, np.atleast_2d(s.Spectrum["intensity"]), channels).items():
            K[c][j] = k
        print(f"log n_e {logne[j]:.2f} done", flush=True)
    hdus = [fits.PrimaryHDU()]
    for c in channels:
        h = fits.ImageHDU(K[c], name=f"A{c}")
        h.header["BUNIT"] = "DN cm5 s-1 pix-1"
        h.header["COMMENT"] = "K(T, n_e) per unit EM = int n_e n_H dl; axis 1 = LOGT, axis 2 = LOGNE"
        hdus.append(h)
    for name, arr in (("LOGT", logt), ("LOGNE", logne)):
        hdus.append(fits.BinTableHDU.from_columns([fits.Column(name=name, format="D", array=arr)], name=name))
    hdus[0].header["MINABUND"] = min_abund
    hdus[0].header["DWVL"] = dw
    fits.HDUList(hdus).writeto(path, overwrite=True)


class Response:
    """K(T, n_e) lookup from a ``build`` file, bilinear in (log T, log n_e), clamped at the grid edges."""

    def __init__(self, path):
        from astropy.io import fits

        with fits.open(path) as f:
            self.logt = np.array(f["LOGT"].data["LOGT"])
            self.logne = np.array(f["LOGNE"].data["LOGNE"])
            self.K = {int(h.name[1:]): np.array(h.data) for h in f[1:] if h.name.startswith("A")}

    def __call__(self, channel, T, n_e):
        from scipy.interpolate import RegularGridInterpolator

        f = RegularGridInterpolator((self.logne, self.logt), self.K[channel])
        x = np.clip(np.log10(n_e), self.logne[0], self.logne[-1])
        y = np.clip(np.log10(T), self.logt[0], self.logt[-1])
        x, y = np.broadcast_arrays(x, y)
        return f(np.stack([x.ravel(), y.ravel()], -1)).reshape(x.shape)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("out")
    b.add_argument("--logt", nargs=3, type=float, default=[5.5, 7.5, 0.05])
    b.add_argument("--logne", nargs="+", type=float, default=[8, 9, 10, 11])
    b.add_argument("--min-abund", type=float, default=1e-6)
    b.add_argument("--proc", type=int, default=4)
    a = p.parse_args(argv)
    lo, hi, st = a.logt
    build(a.out, np.arange(lo, hi + st / 2, st), a.logne, min_abund=a.min_abund, proc=a.proc)


if __name__ == "__main__":
    main()
