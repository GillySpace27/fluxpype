"""The scene document for a planar arcade: validated, JSON round-trippable, with a stable content hash."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = 1
MAX_LINES = 200


def shift_pairing(n: int, k: int) -> List[int]:
    """Which negative footpoint each positive one connects to (same row order along the PIL), sheared by `k` steps.

    k = 0 gives the identity (every arcade line crosses the PIL at a right angle). For k > 0, positive j connects
    to negative j + k; the k positives left over at the far end connect to the k leftover negatives at the near end,
    in reverse order, so the result is still a permutation (every footpoint is used once, the flux stays balanced).
    Those wrap lines are long and are an end effect of a finite arcade; `checks.line_separation` shows how close
    they come to their neighbours.
    """
    if n < 1 or not 0 <= k < n:
        raise ValueError("shift_pairing needs n >= 1 and 0 <= k < n")
    pair = [j + k for j in range(n - k)]
    pair += [k - 1 - i for i in range(k)]
    return pair


@dataclass
class PlanarArcadeSpec:
    """A row of `n_lines` arcade lines across a PIL at x = 0.

    Positive footpoints sit at x = -separation/2, negative at x = +separation/2, spaced `spacing` apart along y and
    centred on y = 0. `shear_offset` moves the whole negative row along y (the footpoints themselves are sheared);
    `pairing` is an explicit list saying which negative footpoint each positive one connects to (connectivity is a
    hand-set input here, not something FLUX derives). Each line is an arc whose height is `height_ratio` times half
    its chord length, capped at `max_height`.
    """

    name: str = "planar-arcade"
    n_lines: int = 5
    separation: float = 2.0
    spacing: float = 0.25
    shear_offset: float = 0.0
    pairing: Optional[List[int]] = None  # None means the identity
    flux_per_line: float = 1.0
    height_ratio: float = 1.0
    max_height: float = 1.5
    vertices_per_line: int = 19
    photosphere_z: float = -0.001  # the same plane for every variant that is to be compared
    lid_height: Optional[float] = None  # a B2 plane at this z, normal pointing down; None means no lid
    notes: str = ""
    schema: int = field(default=SCHEMA)

    def validate(self) -> "PlanarArcadeSpec":
        if self.schema != SCHEMA:
            raise ValueError(f"unknown scene schema {self.schema}; this code reads schema {SCHEMA}")
        if not 1 <= self.n_lines <= MAX_LINES:
            raise ValueError(f"n_lines must be between 1 and {MAX_LINES}")
        for key in ("separation", "spacing", "flux_per_line", "height_ratio", "max_height"):
            if not getattr(self, key) > 0:
                raise ValueError(f"{key} must be positive")
        if self.vertices_per_line < 3:
            raise ValueError("vertices_per_line must be at least 3")
        if not self.photosphere_z < 0:
            raise ValueError("photosphere_z must be below the footpoints (negative)")
        pair = self.pairs()
        if sorted(pair) != list(range(self.n_lines)):
            raise ValueError("pairing must be a permutation of 0..n_lines-1")
        if self.lid_height is not None and self.lid_height <= self.max_height * 1.2:
            raise ValueError("lid_height must clear max_height by at least 20 percent")
        return self

    def pairs(self) -> List[int]:
        return list(range(self.n_lines)) if self.pairing is None else list(self.pairing)

    def footpoints(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """(positive, negative) footpoint (x, y) rows, in row order."""
        ys = [(j - (self.n_lines - 1) / 2) * self.spacing for j in range(self.n_lines)]
        pos = [(-self.separation / 2, y) for y in ys]
        neg = [(self.separation / 2, y + self.shear_offset) for y in ys]
        return pos, neg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanarArcadeSpec":
        known = {f for f in cls.__dataclass_fields__}
        extra = set(data) - known
        if extra:
            raise ValueError("unknown scene fields: " + ", ".join(sorted(extra)))
        return cls(**data).validate()

    def content_hash(self) -> str:
        """Hash of everything that changes the compiled world (the name and notes do not)."""
        d = self.to_dict()
        d.pop("name"), d.pop("notes")
        d["pairing"] = self.pairs()
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]
