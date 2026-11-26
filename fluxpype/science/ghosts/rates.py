"""Atomic rates, ported from GHOSTS ``environment`` (coronasim.py: findUpsilon, findQt, recomb1-3, collisFit,
cLoadOneCollisRate) with constants as GHOSTS defined them."""

import os
import re
from functools import lru_cache

import numpy as np
from scipy import interpolate

from . import chianti

KB = 1.380e-16  # erg/K (GHOSTS value)
K_EV = 8.6173303e-5  # eV/K
H_ERG_S = 6.626e-27
C_CMS = 2.998e10
I_INF = 2.18056334613e-11  # erg, 1 Rydberg
RYD_K = 1.57888e5  # K per Rydberg (GHOSTS findUpsilon)


# ------------------------------------------------------------------------------------------- collisional excitation


class Upsilon:
    """Effective collision strength Upsilon(T) for one transition from Burgess-Tully scaled values.

    ``spline="cubic"`` interpolates the scaled points with a cubic spline through them (CHIANTI's practice).
    ``spline="ghosts"`` reproduces GHOSTS: scipy ``UnivariateSpline`` with its default smoothing, sampled on 200
    points in [0, 1] and then interpolated linearly.
    """

    def __init__(self, scups, spline="cubic"):
        self.s = scups
        x, y = scups["x"], scups["y"]
        if spline == "ghosts":
            xs = np.linspace(0, 1, 200)
            ys = interpolate.UnivariateSpline(x, y)(xs)
            self._f = lambda q: np.interp(q, xs, ys, left=np.nan, right=np.nan)
        elif spline == "cubic":
            f = interpolate.CubicSpline(x, y, bc_type="natural")
            self._f = lambda q: np.where((q >= x[0]) & (q <= x[-1]), f(q), np.nan)
        else:
            raise ValueError(spline)

    def __call__(self, T):
        T = np.asarray(T, dtype=float)
        dE, K, C = self.s["dE"], self.s["type"], self.s["C"]
        E = np.abs(T / (RYD_K * dE))
        if K in (1, 4):
            X = np.log10((E + C) / C) / np.log10(E + C)
        elif K in (2, 3):
            X = E / (E + C)
        else:
            raise ValueError(f"Burgess-Tully type {K} not supported (GHOSTS handled 1-4)")
        Y = self._f(X)
        if K == 1:
            Y = Y * np.log10(E + np.e)
        elif K == 3:
            Y = Y / (E + 1)
        elif K == 4:
            Y = Y * np.log10(E + C)
        return Y


def q_excitation(upsilon, T, w_lower):
    """Collisional excitation rate coefficient q(T), cm^3 s^-1 (GHOSTS findQt):
    2.172e-8 sqrt(I_inf / kT) exp(-dE / kT) Upsilon / w_lower."""
    T = np.asarray(T, dtype=float)
    kt = KB * T
    dE = upsilon.s["dE"] * I_INF  # Rydberg -> erg (GHOSTS ryd2erg)
    return 2.172e-8 * np.sqrt(I_INF / kt) * np.exp(-dE / kt) * upsilon(T) / w_lower


# --------------------------------------------------------------------------------------------- recombination


def recomb1(A, B, T0, T1, T):
    TT0 = np.sqrt(T / T0)
    TT1 = np.sqrt(T / T1)
    return A / (TT0 * (1 + TT0) ** (1 - B) * (1 + TT1) ** (1 + B))


def recomb2(A, B, T0, T1, C, T2, T):
    b = B + C * np.exp(-T2 / T)
    TT0 = np.sqrt(T / T0)
    TT1 = np.sqrt(T / T1)
    return A / (TT0 * (1 + TT0) ** (1 - b) * (1 + TT1) ** (1 + b))


def recomb3(A, eta, T):
    return A * (T / 10000) ** (-eta)


def rr_rate(fit, T):
    """Radiative recombination, cm^3 s^-1, from read_rrparams output."""
    if fit is None:
        return np.zeros_like(np.asarray(T, dtype=float))
    typ, p = fit
    T = np.asarray(T, dtype=float)
    if typ == 1:
        return recomb1(*p[:4], T)
    if typ == 2:
        return recomb2(*p[:6], T)
    if typ == 3:
        return recomb3(*p[:2], T)
    raise ValueError(typ)


def dr_rate(fit, T):
    """Dielectronic recombination, cm^3 s^-1, from read_drparams output (Badnell type 1, Shull type 2).
    Not in GHOSTS."""
    T = np.asarray(T, dtype=float)
    if fit is None:
        return np.zeros_like(T)
    if fit[0] == 1:
        _, e, c = fit
        k = e > 0
        return np.sum(c[k, None] * np.exp(-e[k, None] / np.atleast_1d(T)[None, :]), axis=0).reshape(T.shape) / T**1.5
    A, B, T0, T1 = fit[1]
    return A * np.exp(-T0 / T) * (1 + B * np.exp(-T1 / T)) / T**1.5


# --------------------------------------------------------------------------------------------- ionization


def collis_fit(dE, P, A, X, K, T):
    """GHOSTS collisFit: A (1 + P sqrt(U)) / (X + U) U^K exp(-U), U = dE / kT[eV]; cm^3 s^-1."""
    U = dE / (K_EV * np.asarray(T, dtype=float))
    return A * (1 + P * np.sqrt(U)) / (X + U) * U**K * np.exp(-U)


@lru_cache(maxsize=4)
def read_collisions(path):
    """GHOSTS ``collisions.dat``: dict (Z, ion) -> (dE, P, A, X, K), ion = Z - n_electrons + 1, parsed as
    cLoadOneCollisRate did (Fortran DATA statements, D exponents)."""
    out = {}
    with open(path) as f:
        for line in f:
            if "/" not in line:
                continue
            line = line.replace("D", "E")
            data = line.split("/")
            info = data[0].split(",")[1:3]
            try:
                info[1] = re.sub(r"\D", "", info[1])
                z, nel = int(re.sub(r"\D", "", info[0])), int(info[1])
                vals = [float(v) for v in data[1].split(",")]
            except (ValueError, IndexError):
                continue
            out[(z, z - nel + 1)] = tuple(vals[:5])
    return out


class RateSet:
    """Ionization C_i(T) and recombination R_i(T) for every ion of element ``z`` (i = 1 neutral .. z + 1 bare).

    ``dielectronic=False`` reproduces GHOSTS (radiative recombination only). Ionization uses the GHOSTS
    ``collisions.dat`` fits; its path comes from ``collisions`` or ``$GHOSTS_COLLISIONS``.
    """

    def __init__(self, z, root=None, collisions=None, dielectronic=True):
        self.z = z
        self.root = chianti.chianti_dir(root)
        path = collisions or os.environ.get("GHOSTS_COLLISIONS")
        if not path or not os.path.isfile(path):
            raise FileNotFoundError("collisional ionization table (GHOSTS collisions.dat) not found")
        tab = read_collisions(path)
        self.ion_fit = {i: tab.get((z, i)) for i in range(1, z + 1)}
        self.rr = {i: chianti.read_rrparams(self.root, z, i) for i in range(2, z + 2)}
        self.dr = {i: (chianti.read_drparams(self.root, z, i) if dielectronic else None) for i in range(2, z + 2)}

    def C(self, i, T):
        """Ionization rate coefficient out of ion i (to i + 1), cm^3 s^-1."""
        fit = self.ion_fit.get(i)
        if fit is None or not 1 <= i <= self.z:
            return np.zeros_like(np.asarray(T, dtype=float))
        return collis_fit(*fit, T)

    def R(self, i, T):
        """Recombination rate coefficient out of ion i (to i - 1), cm^3 s^-1."""
        if not 2 <= i <= self.z + 1:
            return np.zeros_like(np.asarray(T, dtype=float))
        return rr_rate(self.rr.get(i), T) + dr_rate(self.dr.get(i), T)
