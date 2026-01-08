"""Terrain types for mech combat.

This module re-exports terrain types from core/shared/terrain.py.
All terrain types are now defined in core/shared/terrain.py.
"""

from core.shared.terrain import TerrainHex, TerrainMap, terrain_index

__all__ = [
    "TerrainHex",
    "TerrainMap",
    "terrain_index",
]
