"""GHOSTS atomic-physics port. Unit tests run anywhere; the Table 1 regression needs CHIANTI 8 ($XUVTOP or
$GHOSTS_CHIANTI8) and the GHOSTS collisions.dat ($GHOSTS_COLLISIONS) and is skipped without them."""

import os

import numpy as np
import pytest

from fluxpype.science.ghosts import chianti, ionbalance, rates


class ToyRates:
    """Three-state element with constant ionization and recombination coefficients."""

    z = 2

    def __init__(self, c=(2.0, 1.0), r=(1.0, 4.0)):
        self.c, self.r = c, r

    def C(self, i, T):
        T = np.atleast_1d(np.asarray(T, float))
        return np.full_like(T, self.c[i - 1] if i <= 2 else 0.0)

    def R(self, i, T):
        T = np.atleast_1d(np.asarray(T, float))
        return np.full_like(T, self.r[i - 2] if i >= 2 else 0.0)


def test_equilibrium_ratios():
    f = ionbalance.equilibrium(ToyRates(), [1e6, 2e6])
    assert f.shape == (3, 2)
    np.testing.assert_allclose(f.sum(axis=0), 1)
    np.testing.assert_allclose(f[1] / f[0], 2.0)  # C_1 / R_2
    np.testing.assert_allclose(f[2] / f[1], 0.25)  # C_2 / R_3


def test_nei_relaxes_to_equilibrium_and_freezes():
    toy = ToyRates()
    s = np.linspace(0, 10, 50)
    T = np.full_like(s, 1e6)
    f0 = np.array([1.0, 0.0, 0.0])
    slow = ionbalance.nei_along(toy, s, np.full_like(s, 10.0), T, np.ones_like(s), f0=f0)
    np.testing.assert_allclose(slow[:, -1], ionbalance.equilibrium(toy, [1e6])[:, 0], rtol=1e-4)
    fast = ionbalance.nei_along(toy, s, np.full_like(s, 1e-6), T, np.ones_like(s), f0=f0)
    np.testing.assert_allclose(fast[:, -1], f0, atol=1e-4)


def test_collis_fit_and_recomb3():
    # U = 1 at T = dE / k: A (1 + P) / (X + 1) e^-1
    dE = 10.0
    T = dE / rates.K_EV
    assert rates.collis_fit(dE, 0.5, 1e-8, 0.2, 0.3, T) == pytest.approx(1e-8 * 1.5 / 1.2 * np.exp(-1))
    assert rates.recomb3(2e-12, 0.7, 1e4) == pytest.approx(2e-12)


def test_dr_shull_form():
    T = np.array([1e5, 1e6])
    A, B, T0, T1 = 1e-3, 0.1, 2e5, 1e5
    expect = A * np.exp(-T0 / T) * (1 + B * np.exp(-T1 / T)) / T**1.5
    np.testing.assert_allclose(rates.dr_rate((2, [A, B, T0, T1]), T), expect)


def test_upsilon_type2_constant():
    scups = {"dE": 1.0, "gf": 0.1, "lim": 0.0, "type": 2, "C": 1.0, "x": np.linspace(0, 1, 5), "y": np.full(5, 3.0)}
    for spline in ("cubic", "ghosts"):
        np.testing.assert_allclose(rates.Upsilon(scups, spline=spline)([1e5, 1e6, 1e7]), 3.0, rtol=1e-6)


def test_read_ioneq_fixed_width(tmp_path):
    root = tmp_path
    (root / "ioneq").mkdir()
    (root / "ioneq" / "chianti.ioneq").write_text(
        "  3 26\n  5.00  5.05  5.10\n 26 12 0.000e+001.203e-104 5.000e-01\n -1\n"
    )
    logt, frac = chianti.read_ioneq(str(root))
    np.testing.assert_allclose(logt, [5.0, 5.05, 5.1])
    np.testing.assert_allclose(frac[(26, 12)], [0.0, 1.203e-104, 0.5])


# ----------------------------------------------------------------------- regression against GHOSTS Table 1

CH8 = os.environ.get("GHOSTS_CHIANTI8") or os.environ.get("XUVTOP")
COLL = os.environ.get("GHOSTS_COLLISIONS")
have_data = bool(CH8 and os.path.isdir(CH8) and COLL and os.path.isfile(COLL))

# Gilly & Cranmer (2020) Table 1: ion, wavelength, log T_eq, log q(T_eq)
TABLE1 = [
    (8, 6, 1031.91, 5.36, -8.10),
    (26, 10, 184.54, 5.68, -9.36),
    (26, 11, 188.22, 5.76, -8.99),
    (26, 12, 195.12, 5.84, -8.70),
    (26, 13, 202.04, 5.91, -8.55),
]


@pytest.mark.skipif(not have_data, reason="needs CHIANTI 8 and GHOSTS collisions.dat")
@pytest.mark.parametrize("z, ion, wl, logt_eq, logq", TABLE1)
def test_ghosts_table1(z, ion, wl, logt_eq, logq):
    from fluxpype.science.ghosts.emissivity import Line

    logT = np.arange(5.0, 6.6, 0.005)
    f = ionbalance.equilibrium(rates.RateSet(z, root=CH8, collisions=COLL, dielectronic=False), 10**logT)
    assert logT[f[ion - 1].argmax()] == pytest.approx(logt_eq, abs=0.011)
    line = Line(z, ion, wl, root=CH8, spline="ghosts")
    assert np.log10(line.q(10**logt_eq)) == pytest.approx(logq, abs=0.011)


def test_upsilon_type1_natural_log_vs_ghosts_log10():
    scups = {"dE": 1.0, "gf": 0.1, "lim": 0.0, "type": 1, "C": 2.0, "x": np.linspace(0, 1, 5), "y": np.full(5, 1.0)}
    T = np.array([1e5, 1e6])
    E = T / (rates.RYD_K * 1.0)
    np.testing.assert_allclose(rates.Upsilon(scups)(T), np.log(E + np.e), rtol=1e-6)
    ghosts = rates.Upsilon(scups, spline="ghosts")(T)
    np.testing.assert_allclose(rates.Upsilon(scups)(T) / ghosts, np.log(10), rtol=1e-3)


def test_read_elvlc_chianti9_format(tmp_path):
    d = tmp_path / "o" / "o_6"
    d.mkdir(parents=True)
    (d / "o_6.elvlc").write_text(
        "      1 1s2.2s      1     2    S  0.5          0.000          0.000 ,      0.00000000 ,     -1.00000000\n"
        "      3 1s2.2p      2     2    P  1.5      96907.500      96909.115 ,      0.88308608 ,      0.88310080\n"
        " -1\n"
    )
    assert chianti.read_elvlc_j(str(tmp_path), 8, 6) == {1: 0.5, 3: 1.5}
    assert chianti.read_elvlc_energy(str(tmp_path), 8, 6)[3] == pytest.approx(96907.5 / 109737.31568)
