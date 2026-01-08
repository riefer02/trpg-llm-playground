"""Hex grid primitives and terrain-aware geometry helpers for mech combat.

Per PR2 4000-4100 (Combat Positioning):
- Hex grid with axial coordinates (q, r)
- Line of sight and cover mechanics
- Movement and engagement rules

Consolidates geometry utilities from:
- core/mech/grid.py: Core hex types and math
- core/mech/terrain.py: Terrain indexing

This module serves as the canonical import path for all geometry types.
"""

from __future__ import annotations

from core.mech.grid import (
    HexCoord,
    HexPosition,
    hex_line,
    hexes_between,
    hexes_in_radius,
    hex_cone,
    hex_cone_centered,
    hex_line_from_direction,
    hex_add,
    hex_scale,
    iter_neighbors,
    normalize_hex_direction,
)

from core.shared.terrain import (
    TerrainHex,
    TerrainMap,
    terrain_index,
    terrain_at,
    get_terrain_at,
    calculate_movement_cost,
)

__all__ = [
    "HexCoord",
    "HexPosition",
    "TerrainHex",
    "TerrainMap",
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
    "terrain_index",
    "terrain_at",
    "get_terrain_at",
    "calculate_movement_cost",
]
