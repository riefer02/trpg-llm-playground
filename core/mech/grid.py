"""Hex grid primitives for mech combat positioning."""

from __future__ import annotations

from typing import Iterable
from pydantic import BaseModel, Field, computed_field


class HexCoord(BaseModel):
    """Axial hex coordinate (q, r)."""

    q: int = Field(..., description="Axial q coordinate")
    r: int = Field(..., description="Axial r coordinate")

    model_config = {"frozen": True}

    @computed_field
    @property
    def s(self) -> int:
        """Derived s coordinate (q + r + s = 0)."""
        return -self.q - self.r

    def distance_to(self, other: "HexCoord") -> int:
        """Hex distance between two axial coordinates."""
        return max(
            abs(self.q - other.q),
            abs(self.r - other.r),
            abs(self.s - other.s),
        )

    def neighbors(self) -> list["HexCoord"]:
        """Return axial neighbors (6 directions)."""
        return [
            HexCoord(q=self.q + dq, r=self.r + dr)
            for dq, dr in _AXIAL_DIRECTIONS
        ]

    def is_adjacent(self, other: "HexCoord") -> bool:
        """Check whether another coordinate is adjacent."""
        return self.distance_to(other) == 1

    def line_to(self, other: "HexCoord") -> list["HexCoord"]:
        """Return axial coordinates along a line to another coordinate."""
        return hex_line(self, other)


class HexPosition(BaseModel):
    """Hex position with optional elevation."""

    coord: HexCoord
    elevation: int = Field(default=0, ge=0, description="Vertical elevation in spaces")

    model_config = {"frozen": True}

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


def hexes_between(start: HexCoord, end: HexCoord, include_endpoints: bool = False) -> list[HexCoord]:
    """Return coordinates between two points on the hex line."""
    line = hex_line(start, end)
    if include_endpoints:
        return line
    return line[1:-1]


def iter_neighbors(coord: HexCoord) -> Iterable[HexCoord]:
    """Iterate over axial neighbors."""
    for dq, dr in _AXIAL_DIRECTIONS:
        yield HexCoord(q=coord.q + dq, r=coord.r + dr)


def _axial_to_cube(coord: HexCoord) -> tuple[float, float, float]:
    return (float(coord.q), float(coord.r), float(coord.s))


def _cube_to_axial(cube: tuple[float, float, float]) -> HexCoord:
    return HexCoord(q=int(cube[0]), r=int(cube[1]))


def _cube_lerp(a: tuple[float, float, float], b: tuple[float, float, float], t: float) -> tuple[float, float, float]:
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
