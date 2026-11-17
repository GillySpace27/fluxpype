import math

import pytest

from fluxpype.scenes.potential import interaction_energy, lid_energy_shift, potential_reference, sources_from_spec
from fluxpype.scenes.spec import PlanarArcadeSpec


def test_pair_energy_of_a_dipole():
    # +1 and -1 a distance d apart: two ordered pairs, each -1/d
    d = 2.0
    assert interaction_energy([(0, 0, 1), (d, 0, -1)]) == pytest.approx(-2 / d / (16 * math.pi**2))


def test_lid_shift_matches_far_field_dipole_formula():
    # far lid: shift -> 2 * zeta(3) * (-sum F_i F_j rho^2) / (2 (2H)^3) / (16 pi^2), with -sum = 2 d^2
    d, H = 1.0, 50.0
    zeta3 = 1.2020569031595942
    want = 2 * zeta3 * (2 * d * d) / (2 * (2 * H) ** 3) / (16 * math.pi**2)
    assert lid_energy_shift([(0, 0, 1), (d, 0, -1)], H) == pytest.approx(want, rel=1e-3)


def test_lid_raises_energy_and_falls_with_height():
    src = sources_from_spec(PlanarArcadeSpec(n_lines=9))
    shifts = [lid_energy_shift(src, H) for H in (3, 6, 12, 24)]
    assert all(s > 0 for s in shifts)
    assert shifts == sorted(shifts, reverse=True)


def test_image_sum_converged():
    src = sources_from_spec(PlanarArcadeSpec(n_lines=9))
    assert lid_energy_shift(src, 3, n_images=200) == pytest.approx(lid_energy_shift(src, 3, n_images=4000), rel=1e-6)


def test_unbalanced_flux_is_refused():
    with pytest.raises(ValueError):
        lid_energy_shift([(0, 0, 1), (1, 0, -0.5)], 5)


def test_reference_does_not_depend_on_pairing():
    a = potential_reference(PlanarArcadeSpec(n_lines=9, lid_height=6))
    b = potential_reference(PlanarArcadeSpec(n_lines=9, lid_height=6, pairing=[2, 3, 4, 5, 6, 7, 8, 1, 0]))
    assert a == b
