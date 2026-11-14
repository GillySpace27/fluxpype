"""Scenes: a small, versioned description of a FLUX world that is not a global PFSS run.

Slice A of the Scene Designer plan: a planar bipolar arcade over a polarity inversion line (PIL), described by a
`PlanarArcadeSpec` and compiled to the plain-text `.flux` world format that FLUX reads. Pure Python, no PDL needed,
so it runs and is tested anywhere; relaxing the world needs FLUX itself.

Everything is in FLUX model units (the menagerie worlds use a box of about +-1). Physical units are a later layer.
"""

from .checks import line_separation, summarize
from .planar import compile_flux
from .spec import PlanarArcadeSpec, shift_pairing

__all__ = ["PlanarArcadeSpec", "compile_flux", "shift_pairing", "summarize", "line_separation"]
