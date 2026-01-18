"""Terrain primitives and material system for Lancer combat.

This module provides composable terrain primitives that compile into TerrainMap
for existing core rules (cover/LOS/movement), with SITREP linkage and material
primitives for destructible terrain.

Material Rules (per PR2 ~4123):
- Objects have armor 0-4 depending on material type
- Objects have 10 HP per size
- Objects have evasion 5

Terrain Primitives:
- FloorTile: Normal/difficult/dangerous floor
- Obstacle: Walls, rocks, debris (destructible)
- SoftCoverZone: Smoke, foliage areas (for Hide rules)
- Hazard: Lava, radiation, acid
- Objective: Control points, extraction markers
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import DamageType, SizeClass
from core.mech.grid import HexCoord, HexPosition
from core.shared.terrain import TerrainHex, TerrainMap


__all__ = [
    # Material system
    "MaterialType",
    "MaterialProperties",
    "MATERIAL_ORGANIC",
    "MATERIAL_TOUGH",
    "MATERIAL_HARDY",
    "MATERIAL_FORTIFIED",
    "MATERIAL_ARMORED",
    # Primitives
    "TerrainPrimitive",
    "FloorTile",
    "Obstacle",
    "SoftCoverZone",
    "Hazard",
    "Objective",
    # Destructible state
    "DestructibleTerrainState",
    # Composition
    "GeneratedTerrain",
    "compose_terrain_map",
    "damage_destructible_terrain",
]


# =============================================================================
# Phase 1: Material System
# =============================================================================

MaterialType = Literal["organic", "tough", "hardy", "fortified", "armored"]


class MaterialProperties(FrozenModel):
    """Material properties for destructible terrain.

    Per PR2 ~4123:
    - Objects have armor 0-4 depending on material type
    - Objects have 10 HP per size
    - Objects have evasion 5
    """

    material_type: MaterialType
    armor: int = Field(default=0, ge=0, le=4)
    hp_per_size: int = Field(default=10, ge=1)
    evasion: int = Field(default=5, ge=0)
    is_flammable: bool = False


# Default material constants
MATERIAL_ORGANIC = MaterialProperties(
    material_type="organic",
    armor=0,
    is_flammable=True,
)
MATERIAL_TOUGH = MaterialProperties(
    material_type="tough",
    armor=1,
)
MATERIAL_HARDY = MaterialProperties(
    material_type="hardy",
    armor=2,
)
MATERIAL_FORTIFIED = MaterialProperties(
    material_type="fortified",
    armor=3,
)
MATERIAL_ARMORED = MaterialProperties(
    material_type="armored",
    armor=4,
)


def get_default_material(material_type: MaterialType) -> MaterialProperties:
    """Get default material properties for a material type."""
    material_map = {
        "organic": MATERIAL_ORGANIC,
        "tough": MATERIAL_TOUGH,
        "hardy": MATERIAL_HARDY,
        "fortified": MATERIAL_FORTIFIED,
        "armored": MATERIAL_ARMORED,
    }
    return material_map[material_type]


# =============================================================================
# Phase 2: Terrain Primitives
# =============================================================================

TerrainPrimitiveKind = Literal["floor", "obstacle", "zone", "hazard", "objective"]
SitrepZoneType = Literal["deployment", "extraction", "objective", "ingress"]
FloorType = Literal["normal", "difficult", "dangerous", "climbing"]
SoftCoverSubtype = Literal["smoke", "foliage", "mist", "darkness"]
HazardSubtype = Literal["lava", "acid", "radiation", "electricity"]
ObjectiveType = Literal["control_point", "escort_target", "extraction", "ingress"]


class TerrainPrimitive(FrozenModel):
    """Base terrain primitive representing a single or multi-hex terrain feature.

    Primitives can be composed into a TerrainMap using compose_terrain_map().
    Later primitives in the composition list override earlier ones for overlapping coords.
    """

    id: str
    kind: TerrainPrimitiveKind
    name: str
    coords: list[HexCoord] = Field(default_factory=list)

    # Standard terrain flags (same as TerrainHex)
    elevation: int = Field(default=0, ge=0)
    blocks_line_of_sight: bool = False
    provides_soft_cover: bool = False
    provides_hard_cover: bool = False
    hard_cover_size: SizeClass | None = None
    difficult: bool = False
    dangerous: bool = False

    # Extended properties
    material: MaterialProperties | None = None
    zone_type: SitrepZoneType | None = None


class FloorTile(FrozenModel):
    """Floor tile primitive for normal/difficult/dangerous/climbing terrain."""

    id: str
    kind: Literal["floor"] = "floor"
    name: str
    coords: list[HexCoord] = Field(default_factory=list)
    floor_type: FloorType = "normal"

    # Standard terrain flags
    elevation: int = Field(default=0, ge=0)
    blocks_line_of_sight: bool = False
    provides_soft_cover: bool = False
    provides_hard_cover: bool = False
    hard_cover_size: SizeClass | None = None

    # Derived from floor_type
    @property
    def difficult(self) -> bool:
        """Floor is difficult terrain."""
        return self.floor_type in ("difficult", "climbing")

    @property
    def dangerous(self) -> bool:
        """Floor is dangerous terrain."""
        return self.floor_type == "dangerous"

    # Extended
    material: MaterialProperties | None = None
    zone_type: SitrepZoneType | None = None


class Obstacle(FrozenModel):
    """Obstacle primitive for walls, rocks, debris (destructible).

    Per PR2:
    - Objects have 10 HP per size
    - Objects have armor based on material (0-4)
    - Objects have evasion 5
    """

    id: str
    kind: Literal["obstacle"] = "obstacle"
    name: str
    coords: list[HexCoord] = Field(default_factory=list)

    # Standard terrain flags
    elevation: int = Field(default=0, ge=0)
    blocks_line_of_sight: bool = True
    provides_soft_cover: bool = False
    provides_hard_cover: bool = True
    hard_cover_size: SizeClass | None = "size_1"
    difficult: bool = False
    dangerous: bool = False

    # Obstacle-specific
    size: int = Field(default=1, ge=1)
    hp: int | None = None  # Default: 10 * size
    is_destructible: bool = True
    material: MaterialProperties | None = None
    zone_type: SitrepZoneType | None = None

    @property
    def max_hp(self) -> int:
        """Calculate max HP for this obstacle."""
        if self.hp is not None:
            return self.hp
        hp_per_size = 10
        if self.material is not None:
            hp_per_size = self.material.hp_per_size
        return hp_per_size * self.size


class SoftCoverZone(FrozenModel):
    """Soft cover zone primitive for smoke, foliage areas (for Hide rules).

    These zones provide soft cover for hiding and may have limited duration.
    """

    id: str
    kind: Literal["zone"] = "zone"
    name: str
    coords: list[HexCoord] = Field(default_factory=list)
    zone_subtype: SoftCoverSubtype = "smoke"

    # Standard terrain flags
    elevation: int = Field(default=0, ge=0)
    blocks_line_of_sight: bool = False
    provides_soft_cover: bool = True
    provides_hard_cover: bool = False
    hard_cover_size: SizeClass | None = None
    difficult: bool = False
    dangerous: bool = False

    # Zone-specific
    duration_rounds: int | None = None  # None = permanent
    created_round: int | None = None
    material: MaterialProperties | None = None
    zone_type: SitrepZoneType | None = None


class Hazard(FrozenModel):
    """Hazard primitive for lava, radiation, acid, electricity.

    Per PR2: Default 5 damage, engineering check DC 10.
    """

    id: str
    kind: Literal["hazard"] = "hazard"
    name: str
    coords: list[HexCoord] = Field(default_factory=list)
    hazard_subtype: HazardSubtype = "lava"

    # Standard terrain flags
    elevation: int = Field(default=0, ge=0)
    blocks_line_of_sight: bool = False
    provides_soft_cover: bool = False
    provides_hard_cover: bool = False
    hard_cover_size: SizeClass | None = None
    difficult: bool = False
    dangerous: bool = True  # Hazards are always dangerous terrain

    # Hazard-specific
    damage: int = Field(default=5, ge=0)
    damage_type: DamageType = "kinetic"
    check_dc: int = Field(default=10, ge=0)
    material: MaterialProperties | None = None
    zone_type: SitrepZoneType | None = None


class Objective(FrozenModel):
    """Objective primitive for control points, extraction markers, etc."""

    id: str
    kind: Literal["objective"] = "objective"
    name: str
    coords: list[HexCoord] = Field(default_factory=list)
    objective_type: ObjectiveType = "control_point"

    # Standard terrain flags
    elevation: int = Field(default=0, ge=0)
    blocks_line_of_sight: bool = False
    provides_soft_cover: bool = False
    provides_hard_cover: bool = False
    hard_cover_size: SizeClass | None = None
    difficult: bool = False
    dangerous: bool = False

    # Objective-specific
    zone_id: str | None = None  # Links to SitrepZone
    material: MaterialProperties | None = None
    zone_type: SitrepZoneType | None = None


# Union type for all primitives
AnyTerrainPrimitive = (
    TerrainPrimitive | FloorTile | Obstacle | SoftCoverZone | Hazard | Objective
)


# =============================================================================
# Phase 6: Destructible Terrain State
# =============================================================================


class DestructibleTerrainState(FrozenModel):
    """Runtime state for a destructible terrain feature.

    Per PR2:
    - Objects have armor 0-4 depending on material
    - Objects have 10 HP per size
    - Objects have evasion 5
    """

    primitive_id: str
    position: HexPosition
    size: int = Field(default=1, ge=1)
    hp: int = Field(ge=0)
    max_hp: int = Field(ge=1)
    armor: int = Field(default=0, ge=0, le=4)
    evasion: int = Field(default=5, ge=0)
    material: MaterialType = "hardy"
    is_destroyed: bool = False
    provides_soft_cover: bool = False
    provides_hard_cover: bool = True
    hard_cover_size: SizeClass | None = None


def damage_destructible_terrain(
    state: DestructibleTerrainState,
    damage: int,
    armor_piercing: int = 0,
) -> tuple[DestructibleTerrainState, bool]:
    """Apply damage to destructible terrain.

    Args:
        state: Current state of the destructible terrain
        damage: Amount of damage to apply
        armor_piercing: Amount of armor to ignore

    Returns:
        Tuple of (new_state, was_destroyed)
    """
    if state.is_destroyed:
        return state, False

    # Calculate effective armor (reduced by armor piercing)
    effective_armor = max(0, state.armor - armor_piercing)

    # Apply damage reduction from armor
    actual_damage = max(0, damage - effective_armor)

    # Calculate new HP
    new_hp = max(0, state.hp - actual_damage)
    was_destroyed = new_hp == 0

    # Create new state
    new_state = DestructibleTerrainState(
        primitive_id=state.primitive_id,
        position=state.position,
        size=state.size,
        hp=new_hp,
        max_hp=state.max_hp,
        armor=state.armor,
        evasion=state.evasion,
        material=state.material,
        is_destroyed=was_destroyed,
        provides_soft_cover=state.provides_soft_cover if not was_destroyed else False,
        provides_hard_cover=state.provides_hard_cover if not was_destroyed else False,
        hard_cover_size=state.hard_cover_size if not was_destroyed else None,
    )

    return new_state, was_destroyed


def create_destructible_state(
    obstacle: Obstacle,
    position: HexPosition,
) -> DestructibleTerrainState:
    """Create destructible state from an Obstacle primitive.

    Args:
        obstacle: The obstacle primitive
        position: Position including elevation

    Returns:
        DestructibleTerrainState for runtime tracking
    """
    material = obstacle.material or MATERIAL_HARDY
    max_hp = obstacle.max_hp

    return DestructibleTerrainState(
        primitive_id=obstacle.id,
        position=position,
        size=obstacle.size,
        hp=max_hp,
        max_hp=max_hp,
        armor=material.armor,
        evasion=material.evasion,
        material=material.material_type,
        is_destroyed=False,
        provides_soft_cover=obstacle.provides_soft_cover,
        provides_hard_cover=obstacle.provides_hard_cover,
        hard_cover_size=obstacle.hard_cover_size,
    )


# =============================================================================
# Phase 3: Composition Function
# =============================================================================


class GeneratedTerrain(FrozenModel):
    """Result of composing terrain primitives into a TerrainMap.

    Contains:
    - terrain_map: The compiled TerrainMap for use with existing rules
    - primitives: Original primitives for reference
    - zones: Zone ID to coordinate mapping
    - soft_cover_zones: Soft cover zones for Hide tracking
    - destructibles: Runtime state for destructible terrain
    """

    terrain_map: TerrainMap
    primitives: list[AnyTerrainPrimitive] = Field(default_factory=list)
    zones: dict[str, list[HexCoord]] = Field(default_factory=dict)
    soft_cover_zones: list[SoftCoverZone] = Field(default_factory=list)
    destructibles: list[DestructibleTerrainState] = Field(default_factory=list)


def _primitive_to_terrain_hex(
    primitive: AnyTerrainPrimitive,
    coord: HexCoord,
) -> TerrainHex:
    """Convert a primitive at a coord to a TerrainHex."""
    # Handle FloorTile specially for computed properties
    if isinstance(primitive, FloorTile):
        return TerrainHex(
            coord=coord,
            elevation=primitive.elevation,
            blocks_line_of_sight=primitive.blocks_line_of_sight,
            provides_soft_cover=primitive.provides_soft_cover,
            provides_hard_cover=primitive.provides_hard_cover,
            hard_cover_size=primitive.hard_cover_size,
            difficult=primitive.difficult,
            dangerous=primitive.dangerous,
        )

    return TerrainHex(
        coord=coord,
        elevation=primitive.elevation,
        blocks_line_of_sight=primitive.blocks_line_of_sight,
        provides_soft_cover=primitive.provides_soft_cover,
        provides_hard_cover=primitive.provides_hard_cover,
        hard_cover_size=primitive.hard_cover_size,
        difficult=primitive.difficult,
        dangerous=primitive.dangerous,
    )


def compose_terrain_map(
    primitives: list[AnyTerrainPrimitive],
) -> GeneratedTerrain:
    """Merge primitives into a TerrainMap.

    Later primitives override earlier ones for overlapping coordinates.

    Layering order (by priority, lowest to highest):
    1. Floor tiles (base)
    2. Elevation changes
    3. Obstacles (blocking/cover)
    4. Hazards (dangerous)
    5. Zones (soft cover areas)

    Args:
        primitives: List of terrain primitives to compose

    Returns:
        GeneratedTerrain with compiled map and metadata
    """
    # Track terrain by coordinate (later primitives override)
    coord_to_terrain: dict[HexCoord, TerrainHex] = {}

    # Track zones
    zones: dict[str, list[HexCoord]] = {}

    # Collect soft cover zones
    soft_cover_zones: list[SoftCoverZone] = []

    # Collect destructibles
    destructibles: list[DestructibleTerrainState] = []

    # Sort primitives by layer priority
    def layer_priority(p: AnyTerrainPrimitive) -> int:
        kind = p.kind
        if kind == "floor":
            return 0
        elif kind == "obstacle":
            return 2
        elif kind == "hazard":
            return 3
        elif kind == "zone":
            return 4
        elif kind == "objective":
            return 1
        return 0

    sorted_primitives = sorted(primitives, key=layer_priority)

    for primitive in sorted_primitives:
        # Process each coordinate in the primitive
        for coord in primitive.coords:
            terrain_hex = _primitive_to_terrain_hex(primitive, coord)
            coord_to_terrain[coord] = terrain_hex

        # Track zones by ID
        if primitive.zone_type is not None:
            if primitive.id not in zones:
                zones[primitive.id] = []
            zones[primitive.id].extend(primitive.coords)

        # Collect soft cover zones
        if isinstance(primitive, SoftCoverZone):
            soft_cover_zones.append(primitive)

        # Create destructible state for obstacles
        if isinstance(primitive, Obstacle) and primitive.is_destructible:
            for coord in primitive.coords:
                position = HexPosition(coord=coord, elevation=primitive.elevation)
                state = create_destructible_state(primitive, position)
                destructibles.append(state)

    # Build terrain map
    terrain_map = TerrainMap(tiles=list(coord_to_terrain.values()))

    return GeneratedTerrain(
        terrain_map=terrain_map,
        primitives=list(primitives),  # type: ignore[arg-type]
        zones=zones,
        soft_cover_zones=soft_cover_zones,
        destructibles=destructibles,
    )
