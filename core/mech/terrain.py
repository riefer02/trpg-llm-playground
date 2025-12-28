"""Terrain tiles and map data for mech combat."""

from pydantic import BaseModel, Field

from core.mech.grid import HexCoord
from core.shared.enums import SizeClass


class TerrainHex(BaseModel):
    """Terrain entry for a single hex."""

    coord: HexCoord
    elevation: int = Field(default=0, ge=0)
    blocks_line_of_sight: bool = False
    provides_soft_cover: bool = False
    provides_hard_cover: bool = False
    hard_cover_size: SizeClass | None = None
    difficult: bool = False
    dangerous: bool = False

    model_config = {"frozen": True}


class TerrainMap(BaseModel):
    """Sparse terrain map for combat scenarios."""

    tiles: list[TerrainHex] = Field(default_factory=list)

    model_config = {"frozen": True}


def terrain_index(terrain: TerrainMap | None) -> dict[tuple[int, int], TerrainHex]:
    """Build a lookup table for terrain by axial coordinates."""
    if terrain is None:
        return {}
    return {(tile.coord.q, tile.coord.r): tile for tile in terrain.tiles}
