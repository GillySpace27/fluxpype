"""Collisionally excited line emissivity, GHOSTS style (paper eq. 11, thesis eq. 2.11), frequency-integrated:
j = (h nu0 / 4 pi) n_e n_ion q(T)   [erg cm^-3 s^-1 sr^-1].

``Line`` bundles one transition; ``contribution(T)`` gives the per-n_e n_H form used to build band images:
G(T) = (h nu0 / 4 pi) A_Z f_ion(T) q(T), so that j = n_e n_H G(T).
"""

import numpy as np

from . import chianti, ionbalance, rates


class Line:
    """One emission line of ion (z, ion) near ``wavelength`` A."""

    def __init__(self, z, ion, wavelength, root=None, spline="cubic", tol=0.01):
        self.root = chianti.chianti_dir(root)
        self.z, self.ion = z, ion
        lower, upper, lam, gf, a_ul = chianti.find_line(self.root, z, ion, wavelength, tol)
        self.lower, self.upper, self.wavelength, self.a_ul = lower, upper, abs(lam), a_ul
        if lower != 1:
            # GHOSTS assumed excitation from the lower level of the line and that every excitation gives one photon
            # of this line (coronal approximation, two-level). That only holds for resonance lines from ground.
            self.note = f"lower level {lower} is not the ground level: two-level coronal approximation is doubtful"
        else:
            self.note = ""
        self.w_lower = 2 * chianti.read_elvlc_j(self.root, z, ion)[lower] + 1
        self.ups = rates.Upsilon(chianti.read_scups(self.root, z, ion, lower, upper), spline=spline)
        self.nu0 = rates.C_CMS / (self.wavelength * 1e-8)

    def q(self, T):
        return rates.q_excitation(self.ups, T, self.w_lower)

    def emissivity(self, n_e, n_ion, T):
        """erg cm^-3 s^-1 sr^-1."""
        return rates.H_ERG_S * self.nu0 / (4 * np.pi) * n_e * n_ion * self.q(T)

    def contribution(self, T, rate_set, abundance):
        """G(T) in erg cm^3 s^-1 sr^-1 with equilibrium ion fractions from ``rate_set``."""
        f = ionbalance.equilibrium(rate_set, T)[self.ion - 1]
        return rates.H_ERG_S * self.nu0 / (4 * np.pi) * abundance * f * self.q(T)

    def __repr__(self):
        return f"Line({chianti.ion_name(self.z, self.ion)} {self.wavelength:.3f} A, {self.lower}-{self.upper})"
