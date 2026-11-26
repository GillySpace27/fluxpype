"""Closed-loop model after Cranmer & Gilly (2026) / Martens (2010)."""

import numpy as np
import pytest

from fluxpype.science.loop_model import T_BASE, LoopModel


def test_normalization_reproduces_t_norm():
    lm = LoopModel()
    assert lm.apex(1.5e9, 100.0)["T_max"] == pytest.approx(lm.t_norm, rel=1e-8)


def test_rtv_limit():
    # alpha = beta = delta = 0 with Martens' (2010) cooling-curve fit: T_max = c (P0 L)^(1/3), RTV's c = 1400
    lm = LoopModel(m=0.0, n=1.0, gamma=0.5, chi0=2.570e12)
    c = [lm.apex(L, 100.0, delta=0.0, T_max=2e6) for L in (1e9, 1e10)]
    k = [a["T_max"] / (a["P0"] * L) ** (1 / 3) for a, L in zip(c, (1e9, 1e10))]
    assert k[0] == pytest.approx(k[1], rel=1e-10)
    assert k[0] == pytest.approx(1400, rel=0.05)


def test_eta_solves_ode():
    lm = LoopModel()
    sol = lm.apex(5e9, 100.0)
    x = np.linspace(0.05, 0.95, 400)
    eta = lm.eta(x, sol)
    d2 = np.gradient(np.gradient(eta, x), x)
    res = sol["eps"] * d2 - (eta ** sol["mu"] - sol["xi"] * eta ** sol["nu"])
    assert np.max(np.abs(res[5:-5]) / eta[5:-5] ** sol["mu"]) < 5e-3
    assert lm.eta([0.0, 1.0], sol) == pytest.approx([0.0, 1.0])


def test_profile_symmetric_and_bounded():
    lm = LoopModel()
    L = 3e9
    s = np.linspace(0, 2 * L, 21)
    T, ne, sol = lm.profile(s, L, 50.0)
    np.testing.assert_allclose(T, T[::-1])
    assert T[0] == pytest.approx(T_BASE)
    assert T.max() == pytest.approx(sol["T_max"], rel=1e-6)
    assert np.all(np.diff(ne[: len(s) // 2 + 1]) <= 0)
