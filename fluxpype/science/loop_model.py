"""Time-steady closed-loop plasma, following Cranmer & Gilly (2026, arXiv 2607.19225) sections 2-3, which build on
Martens (2010).

For a loop of half-length L with basal field B_sun, heating Q = H T^alpha P^beta (paper eqs. 6, 8, 10) balances
conduction (Spitzer with an ambipolar correction, eqs. 19-22) and optically thin radiation fitted as
Q_rad = -P^2 chi0 T^(-2-gamma) (eq. 24). The nondimensional equation eps eta'' = eta^mu - xi eta^nu (eq. 25) has
the Martens solution x(eta) = I_{eta^(nu-mu)}(lambda + 1, 1/2) (regularized incomplete beta), with xi and eps fixed by
eq. 29; eqs. 27-28 then give the apex temperature T_max and base pressure P0. Pressure falls off hydrostatically
along a semicircle (eq. 32) and n_e = P / (2 k T).

The heating normalization E0 is set so that a reference loop (L = 15 Mm, B_sun = 100 G) has apex temperature
T_norm (paper section 3.4). Defaults for m, n and T_norm are the paper's Table 3 "2018 (all), t_n = 0" values.

Choices not pinned down by the paper and made here (flagged so they can be revisited):
- eta is mapped to T with eta = (C(T) - C(T_base)) / (C(T_max) - C(T_base)) so that eta(0) = 0 at T_base.
- The mean particle mass in P = rho k T / <m> is 0.5 m_H (the paper calls this the usual choice).
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import beta as beta_fn
from scipy.special import betaincinv

KB = 1.380649e-16
M_H = 1.6735575e-24
G_SUN = 2.7398e4  # cm s^-2, G M_sun / R_sun^2
R_SUN = 6.957e10
MEAN_MASS = 0.5 * M_H
KAPPA0 = 1.1e-6  # erg s^-1 cm^-1 K^-7/2 (paper section 3.2)
T_A = 4.0e4  # ambipolar transition temperature (paper section 3.2)
T_BASE = 1.0e4
ALPHA_FLOOR = -2.25  # paper section 3.4
CORONAL = dict(gamma=0.4955, chi0=1.729e12)  # paper section 3.3, coronal abundances
PHOTOSPHERIC = dict(gamma=0.6413, chi0=9.198e12)


def delta_of_length(L_cm):
    """Paper eq. 15: delta = (L / 975 Mm)^0.638, the B(s) ~ T(s)^-delta exponent fitted to PFSS loops."""
    return (L_cm / 9.75e10) ** 0.638


def delta_from_field(b_sun, b_min, t_apex=3.0e6):
    """delta from a traced loop's base and minimum field, taking the minimum at a 3 MK apex (paper section 3.1)."""
    return max(np.log(b_sun / b_min) / np.log(t_apex / T_BASE), 0.0)


class LoopModel:
    """Steady loop solutions for heating exponents m, n (paper eqs. 3-4)."""

    def __init__(
        self,
        m=1.607,
        n=1.144,
        t_norm=1.487e6,
        v_perp=1.0e5,
        lam_perp=1.2e7,
        b_spot=500.0,
        gamma=CORONAL["gamma"],
        chi0=CORONAL["chi0"],
        t_a=T_A,
    ):
        self.m, self.n, self.t_norm = m, n, t_norm
        self.v_perp, self.lam_perp, self.b_spot = v_perp, lam_perp, b_spot
        self.gamma, self.chi0, self.t_a = gamma, chi0, t_a
        self.E0 = 1.0
        ref = self.apex(1.5e9, 100.0, delta_of_length(1.5e9), H=None, T_max=t_norm)
        self.E0 = ref["H"] / self._h_bracket(1.5e9, 100.0, ref["delta"])

    # ------------------------------------------------------------------ exponents and constants
    def exponents(self, delta):
        m = self.m
        alpha = max((m - 2) * delta - m / 2, ALPHA_FLOOR)  # eq. 10
        beta = m / 2
        mu = (delta - 2 - self.gamma) / (3.5 + delta)  # eq. 26
        nu = (alpha + delta) / (3.5 + delta)
        lam = (1 - 2 * nu + mu) / (2 * nu - 2 * mu)
        xi = (nu + 1) / (mu + 1)  # eq. 29
        eps = 2 * (nu - mu) ** 2 / ((mu + 1) * beta_fn(lam + 1, 0.5) ** 2)
        return dict(alpha=alpha, beta=beta, mu=mu, nu=nu, lam=lam, xi=xi, eps=eps, delta=delta)

    def _h_bracket(self, L, b_sun, delta):
        """H / E0 from eqs. 6 and 8 with rho = P <m> / (k T); includes the sunspot quenching of v_perp."""
        m, n = self.m, self.n
        v = self.v_perp * np.exp(-((b_sun / self.b_spot) ** 2))
        return (
            v ** (m + 1)
            * self.lam_perp ** (n - m)
            * L ** (m - n - 1)
            * T_BASE ** (delta * (2 - m))
            * (b_sun / np.sqrt(4 * np.pi)) ** (2 - m)
            * (MEAN_MASS / KB) ** (m / 2)
        )

    # ------------------------------------------------------------------ apex solution
    def apex(self, L, b_sun, delta=None, H=None, T_max=None):
        """T_max (K), P0 (dyn cm^-2) and H for a loop of half-length L (cm) and basal field b_sun (G).
        With T_max given instead of H, returns the H that sustains it (used for the normalization)."""
        if delta is None:
            delta = delta_of_length(L)
        e = self.exponents(delta)
        g, chi0 = self.gamma, self.chi0
        if T_max is not None:
            # eq. 27 for P0, then eq. 28 for H
            P0 = np.sqrt(KAPPA0 * T_max ** (5.5 + g) / ((3.5 + delta) * L**2 * chi0 * e["eps"]))
            H = e["xi"] * P0 ** (2 - e["beta"]) * chi0 / T_max ** (e["alpha"] + g + 2)
        else:
            if H is None:
                H = self.E0 * self._h_bracket(L, b_sun, delta)
            if not H > 0:  # no heating (v_perp fully quenched): no coronal solution
                return dict(T_max=0.0, P0=0.0, H=0.0, **e)
            # eqs. 27-28 in logs: 2 lnP - (5.5+g) lnT = A ; (beta-2) lnP + (alpha+g+2) lnT = Bc
            A = np.log(KAPPA0 / ((3.5 + delta) * L**2 * chi0 * e["eps"]))
            Bc = np.log(e["xi"] * chi0 / H)
            M = np.array([[2.0, -(5.5 + g)], [e["beta"] - 2, e["alpha"] + g + 2]])
            lnP, lnT = np.linalg.solve(M, [A, Bc])
            P0, T_max = np.exp(lnP), np.exp(lnT)
        return dict(T_max=T_max, P0=P0, H=H, **e)

    # ------------------------------------------------------------------ spatial profiles
    def _C(self, T, delta):
        """Paper eq. 22."""
        return T**delta * (T**3.5 + (10 * delta + 35) / (2 * delta + 1) * self.t_a**3 * T**0.5)

    def eta(self, x, sol):
        """Martens solution eta(x), x = s / L in [0, 1]."""
        x = np.clip(np.asarray(x, float), 0.0, 1.0)
        w = betaincinv(sol["lam"] + 1, 0.5, x)
        return w ** (1.0 / (sol["nu"] - sol["mu"]))

    def temperature(self, x, sol):
        d, tmax = sol["delta"], sol["T_max"]
        c0, c1 = self._C(T_BASE, d), self._C(tmax, d)
        target = c0 + self.eta(x, sol) * (c1 - c0)
        out = np.empty_like(target)
        for i, c in np.ndenumerate(target):
            out[i] = T_BASE if c <= c0 else tmax if c >= c1 else brentq(lambda t: self._C(t, d) - c, T_BASE, tmax)
        return out

    def pressure(self, s, L, sol):
        """Eq. 32: isothermal hydrostatic fall-off along a semicircle of half-length L (s from the nearer foot)."""
        h = KB * sol["T_max"] / (MEAN_MASS * G_SUN)
        th = np.sin(np.pi * np.asarray(s, float) / (2 * L))
        return sol["P0"] * np.exp(-2 * L / (np.pi * h) * th / (1 + 2 * L / (np.pi * R_SUN) * th))

    def profile(self, s, L, b_sun, delta=None):
        """T (K), n_e (cm^-3) at distances ``s`` (cm) along a loop of total length 2L; s is folded about the apex."""
        sol = self.apex(L, b_sun, delta)
        s = np.asarray(s, float)
        s_near = np.minimum(s, 2 * L - s)
        T = self.temperature(s_near / L, sol)
        P = self.pressure(s_near, L, sol)
        return T, P / (2 * KB * T), sol
