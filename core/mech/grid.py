"""Hex grid primitives for mech combat positioning."""

from __future__ import annotations

from typing import Iterable

from pydantic import Field, computed_field

from core.shared.models import FrozenModel
from core.shared.enums import SizeClass


__all__ = [
    "HexCoord",
    "HexPosition",
    "size_to_radius",
    "footprint_coords",
    "hex_line",
    "hexes_between",
    "hexes_in_radius",
    "hex_cone",
    "hex_cone_centered",
    "hex_line_from_direction",
    "hex_add",
    "hex_scale",
    "iter_neighbors",
    "normalize_hex_direction",
    "adjacency_distance",
    "is_adjacent_by_size",
]


class HexCoord(FrozenModel):
    """Axial hex coordinate (q, r).

    Supports tuple-like comparison and indexing for migration compatibility:
    - h == (x, y) returns True if coordinates match
    - h[0] == h.q, h[1] == h.r
    - hash(h) == hash((x, y))
    """

    q: int = Field(..., description="Axial q coordinate")
    r: int = Field(..., description="Axial r coordinate")

    @computed_field
    @property
    def s(self) -> int:
        """Derived s coordinate (q + r + s = 0)."""
        return -self.q - self.r

    def __getitem__(self, index: int) -> int:
        """Enable tuple-like indexing (hex[0] == hex.q, hex[1] == hex.r)."""
        if index == 0:
            return self.q
        elif index == 1:
            return self.r
        raise IndexError("HexCoord index out of range (only 0 and 1 are valid)")

    def __len__(self) -> int:
        """Return 2 for tuple compatibility."""
        return 2

    def __eq__(self, other: object) -> bool:
        """Support equality with HexCoord and tuple[int, int]."""
        if isinstance(other, HexCoord):
            return self.q == other.q and self.r == other.r
        if isinstance(other, tuple) and len(other) == 2:
            return self.q == other[0] and self.r == other[1]
        return NotImplemented

    def __hash__(self) -> int:
        """Support use as dict key and set member."""
        return hash((self.q, self.r))

    def __repr__(self) -> str:
        """Tuple-like repr for debugging."""
        return f"HexCoord({self.q}, {self.r})"

    def distance_to(self, other: "HexCoord") -> int:
        """Hex distance between two axial coordinates."""
        return max(
            abs(self.q - other.q),
            abs(self.r - other.r),
            abs(self.s - other.s),
        )

    def neighbors(self) -> list["HexCoord"]:
        """Return axial neighbors (6 directions)."""
        return [HexCoord(q=self.q + dq, r=self.r + dr) for dq, dr in _AXIAL_DIRECTIONS]

    def is_adjacent(self, other: "HexCoord") -> bool:
        """Check whether another coordinate is adjacent."""
        return self.distance_to(other) == 1

    def line_to(self, other: "HexCoord") -> list["HexCoord"]:
        """Return axial coordinates along a line to another coordinate."""
        return hex_line(self, other)


class HexPosition(FrozenModel):
    """Hex position with optional elevation."""

    coord: HexCoord
    elevation: int = Field(default=0, ge=0, description="Vertical elevation in spaces")

    def distance_2d(self, other: "HexPosition") -> int:
        """2D hex distance ignoring elevation."""
        return self.coord.distance_to(other.coord)

    def distance_3d(self, other: "HexPosition") -> int:
        """3D distance using max of axial distance and elevation delta."""
        return max(self.distance_2d(other), abs(self.elevation - other.elevation))


_AXIAL_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
)

HEX_DIRECTIONS: tuple[HexCoord, ...] = tuple(
    HexCoord(q=q, r=r) for q, r in _AXIAL_DIRECTIONS
)

_SIZE_TO_RADIUS: dict[SizeClass, int] = {
    "size_half": 0,
    "size_1": 0,
    "size_2": 1,
    "size_3": 2,
    "size_4": 3,
    "size_5": 4,
}


def size_to_radius(size: SizeClass) -> int:
    """Map Lancer size class to hex radius for footprint occupancy."""
    return _SIZE_TO_RADIUS.get(size, 0)


_SIZE_TO_ADJACENCY: dict[SizeClass, int] = {
    "size_half": 1,
    "size_1": 1,
    "size_2": 2,
    "size_3": 3,
    "size_4": 4,
    "size_5": 5,
}


def adjacency_distance(size_a: SizeClass, size_b: SizeClass) -> int:
    """Compute max adjacency distance based on unit sizes.

    Per PR2 rules, size represents "area of influence" - larger units
    can engage/interact at extended range. Returns the maximum distance
    at which two units are considered adjacent for engagement, mine
    triggers, and similar mechanics.

    Examples:
        size_1 vs size_1: 1 (standard adjacency)
        size_2 vs size_1: 2 (size 2 extends reach)
        size_3 vs size_2: 3 (use larger size)
    """
    dist_a = _SIZE_TO_ADJACENCY.get(size_a, 1)
    dist_b = _SIZE_TO_ADJACENCY.get(size_b, 1)
    return max(dist_a, dist_b)


def is_adjacent_by_size(
    coord_a: "HexCoord",
    coord_b: "HexCoord",
    size_a: SizeClass,
    size_b: SizeClass,
) -> bool:
    """Check adjacency considering unit sizes.

    Two units are adjacent if their hex distance is within the adjacency
    distance determined by their sizes. Same position is not adjacent.

    Args:
        coord_a: First unit's hex coordinate
        coord_b: Second unit's hex coordinate
        size_a: First unit's size class
        size_b: Second unit's size class

    Returns:
        True if units are adjacent (within size-based adjacency distance)
    """
    if coord_a == coord_b:
        return False
    distance = coord_a.distance_to(coord_b)
    max_adj_dist = adjacency_distance(size_a, size_b)
    return distance <= max_adj_dist


def footprint_coords(center: HexCoord, size: SizeClass) -> set[HexCoord]:
    """Return occupied hexes for a size class centered on a coord."""
    radius = size_to_radius(size)
    return set(hexes_in_radius(center, radius))


def hex_line(start: HexCoord, end: HexCoord) -> list[HexCoord]:
    """Draw a straight line between two axial coordinates."""
    distance = start.distance_to(end)
    if distance == 0:
        return [start]

    start_cube = _axial_to_cube(start)
    end_cube = _axial_to_cube(end)

    results: list[HexCoord] = []
    for step in range(distance + 1):
        t = step / distance
        cube = _cube_lerp(start_cube, end_cube, t)
        rounded = _cube_round(cube)
        results.append(_cube_to_axial(rounded))
    return results


def hexes_between(
    start: HexCoord, end: HexCoord, include_endpoints: bool = False
) -> list[HexCoord]:
    """Return coordinates between two points on the hex line."""
    line = hex_line(start, end)
    if include_endpoints:
        return line
    return line[1:-1]


def iter_neighbors(coord: HexCoord) -> Iterable[HexCoord]:
    """Iterate over axial neighbors."""
    for dq, dr in _AXIAL_DIRECTIONS:
        yield HexCoord(q=coord.q + dq, r=coord.r + dr)


def normalize_hex_direction(direction: HexCoord) -> HexCoord | None:
    """Normalize a direction to a unit axial direction if possible."""
    if direction.q == 0 and direction.r == 0:
        return None
    for unit in HEX_DIRECTIONS:
        if unit.q == 0:
            if direction.q != 0:
                continue
            if direction.r % unit.r == 0 and direction.r // unit.r > 0:
                return unit
        elif unit.r == 0:
            if direction.r != 0:
                continue
            if direction.q % unit.q == 0 and direction.q // unit.q > 0:
                return unit
        else:
            if direction.q % unit.q != 0 or direction.r % unit.r != 0:
                continue
            scale_q = direction.q // unit.q
            scale_r = direction.r // unit.r
            if scale_q == scale_r and scale_q > 0:
                return unit
    return None


def hex_add(a: HexCoord, b: HexCoord) -> HexCoord:
    """Add two axial coordinates."""
    return HexCoord(q=a.q + b.q, r=a.r + b.r)


def hex_scale(coord: HexCoord, scale: int) -> HexCoord:
    """Scale an axial coordinate by an integer."""
    return HexCoord(q=coord.q * scale, r=coord.r * scale)


def hexes_in_radius(center: HexCoord, radius: int) -> list[HexCoord]:
    """Return axial coordinates within a hex radius (including center)."""
    results: list[HexCoord] = []
    for dq in range(-radius, radius + 1):
        for dr in range(-radius, radius + 1):
            coord = HexCoord(q=center.q + dq, r=center.r + dr)
            if center.distance_to(coord) <= radius:
                results.append(coord)
    return results


def hex_line_from_direction(
    origin: HexCoord,
    direction: HexCoord,
    length: int,
) -> list[HexCoord]:
    """Return axial coordinates along a line from origin in a direction."""
    step = normalize_hex_direction(direction)
    if not step or length <= 0:
        return []
    return [
        HexCoord(q=origin.q + step.q * distance, r=origin.r + step.r * distance)
        for distance in range(1, length + 1)
    ]


def hex_cone(
    origin: HexCoord,
    direction: HexCoord,
    length: int,
) -> list[HexCoord]:
    """Return axial coordinates for a cone of given length."""
    step = normalize_hex_direction(direction)
    if not step or length <= 0:
        return []
    direction_index = HEX_DIRECTIONS.index(step)
    left = HEX_DIRECTIONS[(direction_index - 1) % len(HEX_DIRECTIONS)]
    results: list[HexCoord] = []
    for distance in range(1, length + 1):
        for offset in range(distance):
            results.append(
                hex_add(
                    origin,
                    hex_add(
                        hex_scale(step, distance),
                        hex_scale(left, offset),
                    ),
                )
            )
    return results


def hex_cone_centered(
    origin: HexCoord,
    direction: HexCoord,
    length: int,
) -> list[HexCoord]:
    """Return axial coordinates for a cone centered on the axis direction."""
    step = normalize_hex_direction(direction)
    if not step or length <= 0:
        return []
    step_cube = _axial_to_cube(step)
    axes = [int(step_cube[0]), int(step_cube[1]), int(step_cube[2])]
    forward_idx = axes.index(1)
    lateral_idx = axes.index(0)
    backward_idx = axes.index(-1)
    origin_cube = _axial_to_cube(origin)
    results: list[HexCoord] = []

    for distance in range(1, length + 1):
        for offset in range(-(distance - 1), distance):
            cube = [0, 0, 0]
            cube[forward_idx] = distance
            cube[lateral_idx] = offset
            cube[backward_idx] = -distance - offset
            results.append(
                _cube_to_axial(
                    (
                        origin_cube[0] + cube[0],
                        origin_cube[1] + cube[1],
                        origin_cube[2] + cube[2],
                    )
                )
            )

    return results


def _axial_to_cube(coord: HexCoord) -> tuple[float, float, float]:
    return (float(coord.q), float(coord.r), float(coord.s))


def _cube_to_axial(cube: tuple[float, float, float]) -> HexCoord:
    return HexCoord(q=int(cube[0]), r=int(cube[1]))


def _cube_lerp(
    a: tuple[float, float, float], b: tuple[float, float, float], t: float
) -> tuple[float, float, float]:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def _cube_round(cube: tuple[float, float, float]) -> tuple[float, float, float]:
    rx = round(cube[0])
    ry = round(cube[1])
    rz = round(cube[2])

    x_diff = abs(rx - cube[0])
    y_diff = abs(ry - cube[1])
    z_diff = abs(rz - cube[2])

    if x_diff > y_diff and x_diff > z_diff:
        rx = -ry - rz
    elif y_diff > z_diff:
        ry = -rx - rz
    else:
        rz = -rx - ry

    return (rx, ry, rz)


if __name__ == "__main__":
    pass
