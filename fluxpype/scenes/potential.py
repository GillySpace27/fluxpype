"""Analytic potential-field energy for a planar scene: point flux sources on z = 0 under a mirror lid.

This is the reference a free energy needs. FLUX's B2 plane is a mirror wall (image fluxels), so the matching
continuum problem is a slab 0 < z < H with B_z = 0 on both walls except at the sources. Its potential field is the
field of each source plus its images at z = 2 n H (all integers n). With B = -grad(phi) and a source of flux F
feeding the half space, phi = F / (2 pi r), and the energy (Gaussian units, the same 1/(8 pi) that `e_simple2`
uses) is W = (1/(8 pi)) * sum_i F_i phi(r_i).

The self energy of a point source is infinite and does not depend on the connectivity or (for a core much smaller
than H) on the lid, so it is left out. What is returned is finite and exact for point sources:

* `interaction_energy`: the pair energy with no lid, (1/(16 pi^2)) sum_{i != j} F_i F_j / r_ij.
* `lid_energy_shift`: W(H) - W(no lid), which includes each source's interaction with its own images.

The lid shift is the quantity to compare with FLUX: differences of relaxed K = 0 energies between two lid heights
should equal differences of `lid_energy_shift` if the relaxed K = 0 world is the potential field.
Nothing here runs FLUX or touches the FLUX core.
"""

import math
from typing import List, Sequence, Tuple

import numpy as np

from .spec import PlanarArcadeSpec

Source = Tuple[float, float, float]  # x, y, flux (signed)


def sources_from_spec(spec: PlanarArcadeSpec) -> List[Source]:
    pos, neg = spec.footpoints()
    f = spec.flux_per_line
    return [(x, y, f) for x, y in pos] + [(x, y, -f) for x, y in neg]


def _arrays(sources: Sequence[Source]):
    s = np.asarray(sources, dtype=float)
    d = s[:, None, :2] - s[None, :, :2]
    rho2 = (d**2).sum(-1)
    ff = s[:, 2][:, None] * s[:, 2][None, :]
    return s, rho2, ff


def interaction_energy(sources: Sequence[Source]) -> float:
    """Pair energy of the no-lid (half space) potential field, self energies left out."""
    _, rho2, ff = _arrays(sources)
    n = len(rho2)
    off = ~np.eye(n, dtype=bool)
    if np.any(rho2[off] == 0):
        raise ValueError("two sources share a position")
    return float((ff[off] / np.sqrt(rho2[off])).sum() / (16 * math.pi**2))


def lid_energy_shift(sources: Sequence[Source], lid_height: float, n_images: int = 2000) -> float:
    """W(lid at H) - W(no lid) for the mirror lid, exact for point sources up to the image-sum tail.

    Needs zero net flux (otherwise the image sum diverges). Each image layer contributes
    sum_ij F_i F_j (1/sqrt(rho_ij^2 + a^2) - 1/a) with a = 2 n H; the 1/a term is zero by flux balance and is
    subtracted only to keep the sum well conditioned. Layers fall off as 1/n^3; the tail past `n_images` is added
    from the leading term, -sum_ij F_i F_j rho_ij^2 / (2 a^3).
    """
    if not lid_height > 0:
        raise ValueError("lid_height must be positive")
    s, rho2, ff = _arrays(sources)
    net = s[:, 2].sum()
    if abs(net) > 1e-9 * np.abs(s[:, 2]).sum():
        raise ValueError("the sources must carry zero net flux")
    total = 0.0
    for k in range(1, n_images + 1):
        a = 2 * k * lid_height
        total += float((ff * (1 / np.sqrt(rho2 + a * a) - 1 / a)).sum())
    # tail: sum_{k>N} 1/k^3 ~ 1/(2 N^2) (integral estimate from N + 1/2)
    lead = -float((ff * rho2).sum()) / (2 * (2 * lid_height) ** 3)
    total += lead / (2 * (n_images + 0.5) ** 2)
    return 2 * total / (16 * math.pi**2)  # images above and below


def potential_reference(spec: PlanarArcadeSpec) -> dict:
    """The numbers for one scene: pair energy with no lid, and the lid shift when the scene has a lid."""
    src = sources_from_spec(spec)
    out = {"n_sources": len(src), "interaction_energy_no_lid": interaction_energy(src), "lid_height": spec.lid_height}
    out["lid_energy_shift"] = lid_energy_shift(src, spec.lid_height) if spec.lid_height is not None else 0.0
    return out
