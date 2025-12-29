"""Terrain tiles and map data for mech combat."""

from pydantic import Field
from core.shared.models import FrozenModel

from core.mech.grid import HexCoord
from core.shared.enums import SizeClass


class TerrainHex(FrozenModel):
    """Terrain entry for a single hex."""

    coord: HexCoord
    elevation: int = Field(default=0, ge=0)
    blocks_line_of_sight: bool = False
    provides_soft_cover: bool = False
    provides_hard_cover: bool = False
    hard_cover_size: SizeClass | None = None
    difficult: bool = False
    dangerous: bool = False



class TerrainMap(FrozenModel):
    """Sparse terrain map for combat scenarios."""

    tiles: list[TerrainHex] = Field(default_factory=list)



def terrain_index(terrain: TerrainMap | None) -> dict[tuple[int, int], TerrainHex]:
    """Build a lookup table for terrain by axial coordinates."""
    if terrain is None:
        return {}
    return {(tile.coord.q, tile.coord.r): tile for tile in terrain.tiles}
