"""GHOSTS atomic physics, ported from CoronaSim (github.com/GillySpace27/CoronaSim, master 2ec9a68, ``coronasim.py``).

GHOSTS (Gilly & Cranmer 2020, ApJ; arXiv 2008.09580) forward-models optically thin coronal lines. This package keeps
its atomic physics and drops its environment (ZEPHYR background, B maps, line-of-sight grid), which FLUX replaces:

- ``chianti``: readers for the raw CHIANTI files GHOSTS used (wgfa, elvlc, scups, rrparams, ip, ioneq, abund) plus
  drparams, and the database locator (``chianti_dir`` argument, else ``$XUVTOP``).
- ``rates``: Burgess-Tully upsilon descaling and the collisional excitation rate q(T) (``environment.findUpsilon``,
  ``findQt``); radiative recombination fits (``recomb1-3``); dielectronic recombination (new, see below);
  collisional ionization from the Voronov-type fit table GHOSTS read from ``collisions.dat`` (``collisFit``).
- ``ionbalance``: equilibrium charge states (``element.chargeDistributionEq``) and the time-steady non-equilibrium
  integration along a flux tube (paper eqs. 4-6, scipy Radau).
- ``emissivity``: collisionally excited line emissivity j = (h nu0 / 4 pi) n_e n_i q(T) (paper eq. 11, thesis
  eq. 2.11), integrated over the line profile.

One deliberate difference from GHOSTS: GHOSTS recombination used CHIANTI ``.rrparams`` only, with no dielectronic
recombination. ``RateSet(dielectronic=False)`` reproduces GHOSTS; the default adds CHIANTI ``.drparams``.
"""
