"""Charge states: equilibrium (GHOSTS element.chargeDistributionEq) and time-steady non-equilibrium along a flux
tube (Gilly & Cranmer 2020, eqs. 4-6)."""

import numpy as np
from scipy.integrate import solve_ivp


def equilibrium(rates, T):
    """Ion fractions n_i / n_Z for i = 1 .. z + 1, shape (z + 1, len(T)), from n_i / n_{i-1} = C_{i-1} / R_i
    (paper eq. 2). Computed in log space so very small fractions do not underflow; GHOSTS floored them at 1e-50."""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    n = rates.z + 1
    logr = np.zeros((n, len(T)))
    with np.errstate(divide="ignore"):
        for i in range(2, n + 1):
            logr[i - 1] = logr[i - 2] + np.log(rates.C(i - 1, T)) - np.log(rates.R(i, T))
    logr -= logr.max(axis=0)
    f = np.exp(logr)
    return f / f.sum(axis=0)


def nei_along(rates, s, n_e, T, u, area=None, f0=None, method="Radau", rtol=1e-6, atol=1e-30):
    """Ion fractions along a flow, solving d(n_i u A)/ds = n_e A I_i with I_i the ionization/recombination
    balance (paper eqs. 4-5), written for the fractions y_i = n_i / n_Z of an element that is conserved along the
    tube (n_Z u A = const). That form removes the 2/r and expansion terms of eq. 6, which only carry the total
    density. Inputs are arrays on ``s`` (cm): electron density (cm^-3), temperature (K), speed (cm/s). Starts in
    equilibrium at s[0] unless ``f0`` is given. Returns fractions, shape (z + 1, len(s))."""
    s = np.asarray(s, dtype=float)
    n = rates.z + 1
    if f0 is None:
        f0 = equilibrium(rates, [T[0]])[:, 0]

    def interp(arr):
        return lambda x: np.interp(x, s, arr)

    ne_f, T_f, u_f = interp(np.asarray(n_e)), interp(np.log(np.asarray(T))), interp(np.asarray(u))

    def rhs(x, y):
        t = np.exp(T_f(x))
        C = np.array([rates.C(i, t) for i in range(1, n + 1)]).ravel()
        R = np.array([rates.R(i, t) for i in range(1, n + 1)]).ravel()
        dy = -y * (C + R)
        dy[1:] += y[:-1] * C[:-1]
        dy[:-1] += y[1:] * R[1:]
        return ne_f(x) * dy / u_f(x)

    sol = solve_ivp(rhs, (s[0], s[-1]), f0, t_eval=s, method=method, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y
