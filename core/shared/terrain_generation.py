"""Terrain generation with SITREP linkage for Lancer combat.

This module provides terrain generation based on SITREP templates and tile sets,
producing composable terrain primitives that compile into TerrainMap.

SITREP Linkage:
- SitrepTemplate.deployment_zones → Zone primitives (type=deployment)
- SitrepTemplate.extraction_zone → Zone primitive (type=extraction)
- SitrepTemplate.objective_zones → Objective primitives + cover based on terrain_notes
- SitrepTemplate.ingress_zones → Zone primitives (type=ingress)
- SitrepZone.terrain_notes="fortified" → Material.FORTIFIED obstacles nearby

Tile Sets:
- urban: Buildings, walls, rubble, barricades, vehicles
- industrial: Machinery, catwalks, pipes, containers, vats
- wilderness: Trees, rocks, streams, brush, fallen logs
- zero_g: Debris, asteroids, hull sections, cargo
"""

from __future__ import annotations

import random
from typing import Literal

from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.scenario import SitrepTemplate, SitrepZone
from core.mech.grid import HexCoord
from core.shared.terrain_primitives import (
    MaterialType,
    MaterialProperties,
    MATERIAL_ORGANIC,
    MATERIAL_TOUGH,
    MATERIAL_HARDY,
    MATERIAL_FORTIFIED,
    MATERIAL_ARMORED,
    get_default_material,
    FloorTile,
    Obstacle,
    SoftCoverZone,
    Hazard,
    Objective,
    GeneratedTerrain,
    AnyTerrainPrimitive,
    compose_terrain_map,
)


__all__ = [
    "TileSetType",
    "TileSetConfig",
    "TILE_SETS",
    "TerrainGeneratorParams",
    "generate_terrain_from_sitrep",
    "generate_zone_coords",
    "place_obstacles_around_zone",
    "generate_random_terrain",
]


# =============================================================================
# Phase 7: Tile Set Configurations
# =============================================================================

TileSetType = Literal["urban", "industrial", "wilderness", "zero_g"]
FeatureType = Literal[
    # Urban
    "building",
    "wall",
    "rubble",
    "barricade",
    "vehicle",
    # Industrial
    "machinery",
    "catwalk",
    "pipe",
    "container",
    "vat",
    # Wilderness
    "tree",
    "rock",
    "stream",
    "brush",
    "fallen_log",
    # Zero-G
    "debris",
    "asteroid",
    "hull",
    "cargo",
]

HazardType = Literal["electricity", "acid", "steam", "lava", "radiation"]


class TileSetConfig(FrozenModel):
    """Configuration for a terrain tile set."""

    name: str
    features: list[FeatureType] = Field(default_factory=list)
    materials: list[MaterialType] = Field(default_factory=list)
    hazards: list[HazardType] = Field(default_factory=list)
    cover_density: float = Field(default=0.3, ge=0.0, le=1.0)
    elevation_range: tuple[int, int] = Field(default=(0, 0))
    soft_cover_zones: bool = False
    special_rules: list[str] = Field(default_factory=list)


TILE_SETS: dict[TileSetType, TileSetConfig] = {
    "urban": TileSetConfig(
        name="Urban",
        features=["building", "wall", "rubble", "barricade", "vehicle"],
        materials=["hardy", "fortified"],
        cover_density=0.35,
        elevation_range=(0, 3),
    ),
    "industrial": TileSetConfig(
        name="Industrial",
        features=["machinery", "catwalk", "pipe", "container", "vat"],
        materials=["tough", "hardy"],
        hazards=["electricity", "acid", "steam"],
        cover_density=0.4,
        elevation_range=(0, 2),
    ),
    "wilderness": TileSetConfig(
        name="Wilderness",
        features=["tree", "rock", "stream", "brush", "fallen_log"],
        materials=["organic", "hardy"],
        soft_cover_zones=True,
        cover_density=0.3,
        elevation_range=(0, 1),
    ),
    "zero_g": TileSetConfig(
        name="Zero-G",
        features=["debris", "asteroid", "hull", "cargo"],
        materials=["tough", "hardy"],
        cover_density=0.25,
        elevation_range=(0, 0),
        special_rules=["no_falling", "3d_movement"],
    ),
}


# Feature templates for generating obstacles
FEATURE_TEMPLATES: dict[FeatureType, dict] = {
    # Urban features
    "building": {
        "blocks_los": True,
        "hard_cover": True,
        "size": (2, 4),
        "default_material": "fortified",
    },
    "wall": {
        "blocks_los": True,
        "hard_cover": True,
        "size": (1, 2),
        "default_material": "hardy",
    },
    "rubble": {
        "blocks_los": False,
        "hard_cover": True,
        "difficult": True,
        "size": (1, 1),
        "default_material": "hardy",
    },
    "barricade": {
        "blocks_los": False,
        "hard_cover": True,
        "size": (1, 1),
        "default_material": "tough",
    },
    "vehicle": {
        "blocks_los": True,
        "hard_cover": True,
        "size": (1, 2),
        "default_material": "tough",
    },
    # Industrial features
    "machinery": {
        "blocks_los": True,
        "hard_cover": True,
        "size": (2, 3),
        "default_material": "hardy",
    },
    "catwalk": {
        "blocks_los": False,
        "hard_cover": False,
        "elevation": 1,
        "size": (1, 1),
        "default_material": "tough",
    },
    "pipe": {
        "blocks_los": False,
        "hard_cover": True,
        "size": (1, 2),
        "default_material": "tough",
    },
    "container": {
        "blocks_los": True,
        "hard_cover": True,
        "size": (1, 2),
        "default_material": "hardy",
    },
    "vat": {
        "blocks_los": True,
        "hard_cover": True,
        "hazard_adjacent": True,
        "size": (1, 1),
        "default_material": "hardy",
    },
    # Wilderness features
    "tree": {
        "blocks_los": True,
        "hard_cover": True,
        "size": (1, 1),
        "default_material": "organic",
    },
    "rock": {
        "blocks_los": True,
        "hard_cover": True,
        "size": (1, 2),
        "default_material": "hardy",
    },
    "stream": {
        "blocks_los": False,
        "hard_cover": False,
        "difficult": True,
        "size": (1, 1),
        "default_material": "organic",
    },
    "brush": {
        "blocks_los": False,
        "hard_cover": False,
        "soft_cover": True,
        "size": (2, 4),
        "default_material": "organic",
    },
    "fallen_log": {
        "blocks_los": False,
        "hard_cover": True,
        "size": (1, 2),
        "default_material": "organic",
    },
    # Zero-G features
    "debris": {
        "blocks_los": False,
        "hard_cover": True,
        "difficult": True,
        "size": (1, 2),
        "default_material": "tough",
    },
    "asteroid": {
        "blocks_los": True,
        "hard_cover": True,
        "size": (2, 4),
        "default_material": "hardy",
    },
    "hull": {
        "blocks_los": True,
        "hard_cover": True,
        "size": (2, 3),
        "default_material": "hardy",
    },
    "cargo": {
        "blocks_los": True,
        "hard_cover": True,
        "size": (1, 2),
        "default_material": "tough",
    },
}


# =============================================================================
# Phase 4: Generation Parameters
# =============================================================================


class TerrainGeneratorParams(FrozenModel):
    """Parameters for terrain generation."""

    map_width: int = Field(default=20, ge=5, le=40)
    map_height: int = Field(default=16, ge=4, le=40)
    sitrep_template: SitrepTemplate | None = None
    tile_set: TileSetType = "urban"
    seed: int | None = None
    density: float = Field(default=0.3, ge=0.0, le=1.0)


# =============================================================================
# Zone Coordinate Generation
# =============================================================================


def generate_zone_coords(
    zone: SitrepZone,
    map_width: int,
    map_height: int,
    zone_index: int = 0,
    total_zones: int = 1,
) -> list[HexCoord]:
    """Generate hex coordinates for a SITREP zone.

    Uses zone.location to determine placement on the map.

    Args:
        zone: The SITREP zone configuration
        map_width: Map width in hexes
        map_height: Map height in hexes
        zone_index: Index of this zone (for multiple zones)
        total_zones: Total number of zones of this type

    Returns:
        List of HexCoords for this zone
    """
    # Default zone size
    width = zone.width or 4
    height = zone.height or 4
    coords: list[HexCoord] = []

    location = zone.location or "center"

    # Calculate base position based on location
    if location == "center":
        base_q = (map_width - width) // 2
        base_r = (map_height - height) // 2
    elif location == "map_edge":
        base_q = 0
        base_r = (map_height - height) // 2
    elif location == "opposite_edge":
        base_q = map_width - width
        base_r = (map_height - height) // 2
    elif location == "left_flank":
        base_q = 0
        base_r = map_height // 4
    elif location == "right_flank":
        base_q = 0
        base_r = 3 * map_height // 4
    elif location == "quadrant":
        # For multiple deployment zones, distribute in quadrants
        quadrant = zone_index % 4
        if quadrant == 0:  # Top-left
            base_q = map_width // 4 - width // 2
            base_r = map_height // 4 - height // 2
        elif quadrant == 1:  # Top-right
            base_q = 3 * map_width // 4 - width // 2
            base_r = map_height // 4 - height // 2
        elif quadrant == 2:  # Bottom-left
            base_q = map_width // 4 - width // 2
            base_r = 3 * map_height // 4 - height // 2
        else:  # Bottom-right
            base_q = 3 * map_width // 4 - width // 2
            base_r = 3 * map_height // 4 - height // 2
    else:
        # Default to center for unknown locations
        base_q = (map_width - width) // 2
        base_r = (map_height - height) // 2

    # Generate rectangular zone
    for q_offset in range(width):
        for r_offset in range(height):
            coord = HexCoord(q=base_q + q_offset, r=base_r + r_offset)
            coords.append(coord)

    return coords


def place_obstacles_around_zone(
    zone_coords: list[HexCoord],
    zone_id: str,
    terrain_notes: str | None,
    tile_set_config: TileSetConfig,
    rng: random.Random,
    primitive_counter: list[int],
) -> list[AnyTerrainPrimitive]:
    """Place obstacles around a zone based on terrain notes.

    Args:
        zone_coords: Coordinates of the zone
        zone_id: Zone ID for naming
        terrain_notes: Terrain notes (e.g., "fortified", "exposed")
        tile_set_config: Tile set configuration
        rng: Random number generator
        primitive_counter: Counter for unique IDs [mutable]

    Returns:
        List of obstacle primitives placed around the zone
    """
    primitives: list[AnyTerrainPrimitive] = []

    if not zone_coords:
        return primitives

    # Find zone boundary
    zone_set = set(zone_coords)
    boundary_coords: list[HexCoord] = []

    for coord in zone_coords:
        for neighbor in coord.neighbors():
            if neighbor not in zone_set:
                boundary_coords.append(neighbor)

    # Remove duplicates
    boundary_coords = list(set(boundary_coords))

    # Determine material based on terrain_notes
    material: MaterialProperties = MATERIAL_HARDY
    if terrain_notes:
        notes_lower = terrain_notes.lower()
        if "fortified" in notes_lower:
            material = MATERIAL_FORTIFIED
        elif "armored" in notes_lower:
            material = MATERIAL_ARMORED
        elif "organic" in notes_lower or "natural" in notes_lower:
            material = MATERIAL_ORGANIC
        elif "exposed" in notes_lower:
            # Exposed zones have less cover
            boundary_coords = boundary_coords[: len(boundary_coords) // 3]

    # Place obstacles on some boundary hexes
    coverage = 0.4 if terrain_notes and "fortified" in terrain_notes.lower() else 0.25
    num_obstacles = int(len(boundary_coords) * coverage)

    if num_obstacles > 0:
        selected = rng.sample(boundary_coords, min(num_obstacles, len(boundary_coords)))
        for coord in selected:
            primitive_counter[0] += 1
            obstacle = Obstacle(
                id=f"obstacle_{zone_id}_{primitive_counter[0]}",
                name=f"Cover near {zone_id}",
                coords=[coord],
                material=material,
                hard_cover_size="size_1",
            )
            primitives.append(obstacle)

    return primitives


# =============================================================================
# SITREP Terrain Generation
# =============================================================================


def generate_terrain_from_sitrep(
    template: SitrepTemplate,
    params: TerrainGeneratorParams,
) -> GeneratedTerrain:
    """Generate terrain based on SITREP template and parameters.

    Flow:
    - SitrepTemplate.deployment_zones → Zone primitives (type=deployment)
    - SitrepTemplate.extraction_zone → Zone primitive (type=extraction)
    - SitrepTemplate.objective_zones → Objective primitives + cover
    - SitrepTemplate.ingress_zones → Zone primitives (type=ingress)
    - terrain_notes="fortified" → FORTIFIED obstacles nearby

    Args:
        template: The SITREP template
        params: Generation parameters

    Returns:
        GeneratedTerrain with compiled map and primitives
    """
    rng = random.Random(params.seed)
    tile_set_config = TILE_SETS[params.tile_set]
    primitives: list[AnyTerrainPrimitive] = []
    primitive_counter = [0]

    # Process deployment zones
    for i, zone in enumerate(template.deployment_zones):
        coords = generate_zone_coords(
            zone=zone,
            map_width=params.map_width,
            map_height=params.map_height,
            zone_index=i,
            total_zones=len(template.deployment_zones),
        )

        primitive_counter[0] += 1
        floor = FloorTile(
            id=f"deployment_{i}",
            name=f"Deployment Zone {i + 1}",
            coords=coords,
            zone_type="deployment",
        )
        primitives.append(floor)

    # Process extraction zone
    if template.extraction_zone is not None:
        coords = generate_zone_coords(
            zone=template.extraction_zone,
            map_width=params.map_width,
            map_height=params.map_height,
        )

        primitive_counter[0] += 1
        floor = FloorTile(
            id="extraction_0",
            name="Extraction Zone",
            coords=coords,
            zone_type="extraction",
        )
        primitives.append(floor)

    # Process objective zones
    for i, zone in enumerate(template.objective_zones):
        coords = generate_zone_coords(
            zone=zone,
            map_width=params.map_width,
            map_height=params.map_height,
            zone_index=i,
            total_zones=len(template.objective_zones),
        )

        primitive_counter[0] += 1
        objective = Objective(
            id=f"objective_{i}",
            name=f"Objective {i + 1}",
            coords=coords,
            objective_type="control_point",
            zone_type="objective",
        )
        primitives.append(objective)

        # Place obstacles around objective based on terrain_notes
        obstacles = place_obstacles_around_zone(
            zone_coords=coords,
            zone_id=f"objective_{i}",
            terrain_notes=zone.terrain_notes,
            tile_set_config=tile_set_config,
            rng=rng,
            primitive_counter=primitive_counter,
        )
        primitives.extend(obstacles)

    # Process ingress zones
    for i, zone in enumerate(template.ingress_zones):
        coords = generate_zone_coords(
            zone=zone,
            map_width=params.map_width,
            map_height=params.map_height,
            zone_index=i,
            total_zones=len(template.ingress_zones),
        )

        primitive_counter[0] += 1
        floor = FloorTile(
            id=f"ingress_{i}",
            name=f"Ingress Zone {i + 1}",
            coords=coords,
            zone_type="ingress",
        )
        primitives.append(floor)

    # Add random terrain features from tile set
    random_primitives = _generate_tile_set_features(
        params=params,
        tile_set_config=tile_set_config,
        existing_primitives=primitives,
        rng=rng,
        primitive_counter=primitive_counter,
    )
    primitives.extend(random_primitives)

    return compose_terrain_map(primitives)


def _generate_tile_set_features(
    params: TerrainGeneratorParams,
    tile_set_config: TileSetConfig,
    existing_primitives: list[AnyTerrainPrimitive],
    rng: random.Random,
    primitive_counter: list[int],
) -> list[AnyTerrainPrimitive]:
    """Generate random terrain features from tile set.

    Args:
        params: Generation parameters
        tile_set_config: Tile set configuration
        existing_primitives: Already placed primitives to avoid
        rng: Random number generator
        primitive_counter: Counter for unique IDs

    Returns:
        List of generated feature primitives
    """
    primitives: list[AnyTerrainPrimitive] = []

    # Collect occupied coords
    occupied: set[HexCoord] = set()
    for p in existing_primitives:
        occupied.update(p.coords)

    # Calculate how many features to place
    total_hexes = params.map_width * params.map_height
    available_hexes = total_hexes - len(occupied)
    target_features = int(available_hexes * params.density * tile_set_config.cover_density)

    features_placed = 0
    attempts = 0
    max_attempts = target_features * 10

    while features_placed < target_features and attempts < max_attempts:
        attempts += 1

        # Pick random feature type
        if not tile_set_config.features:
            break
        feature_type = rng.choice(tile_set_config.features)
        template = FEATURE_TEMPLATES.get(feature_type)
        if template is None:
            continue

        # Pick random position
        q = rng.randint(0, params.map_width - 1)
        r = rng.randint(0, params.map_height - 1)
        coord = HexCoord(q=q, r=r)

        if coord in occupied:
            continue

        # Get material
        if tile_set_config.materials:
            material_type: MaterialType = rng.choice(tile_set_config.materials)
        else:
            material_type = template.get("default_material", "hardy")
        material = get_default_material(material_type)

        # Get size
        size_range = template.get("size", (1, 1))
        size = rng.randint(size_range[0], size_range[1])

        # Get elevation
        elevation = template.get("elevation", 0)
        if tile_set_config.elevation_range[1] > 0:
            elevation = rng.randint(*tile_set_config.elevation_range)

        primitive_counter[0] += 1

        # Create primitive based on feature properties
        if template.get("soft_cover"):
            # Soft cover zone (e.g., brush)
            zone = SoftCoverZone(
                id=f"{feature_type}_{primitive_counter[0]}",
                name=feature_type.replace("_", " ").title(),
                coords=[coord],
                zone_subtype="foliage",
                elevation=elevation,
            )
            primitives.append(zone)
        elif template.get("difficult") and not template.get("hard_cover"):
            # Difficult floor tile (e.g., stream, rubble floor)
            floor = FloorTile(
                id=f"{feature_type}_{primitive_counter[0]}",
                name=feature_type.replace("_", " ").title(),
                coords=[coord],
                floor_type="difficult",
                elevation=elevation,
            )
            primitives.append(floor)
        else:
            # Standard obstacle
            obstacle = Obstacle(
                id=f"{feature_type}_{primitive_counter[0]}",
                name=feature_type.replace("_", " ").title(),
                coords=[coord],
                material=material,
                size=size,
                blocks_line_of_sight=template.get("blocks_los", False),
                provides_hard_cover=template.get("hard_cover", False),
                hard_cover_size="size_1" if template.get("hard_cover") else None,
                difficult=template.get("difficult", False),
                elevation=elevation,
            )
            primitives.append(obstacle)

        occupied.add(coord)
        features_placed += 1

        # Add hazard adjacent to certain features
        if template.get("hazard_adjacent") and tile_set_config.hazards:
            neighbors = coord.neighbors()
            valid_neighbors = [n for n in neighbors if n not in occupied]
            if valid_neighbors:
                hazard_coord = rng.choice(valid_neighbors)
                hazard_type = rng.choice(tile_set_config.hazards)
                primitive_counter[0] += 1
                hazard = Hazard(
                    id=f"hazard_{primitive_counter[0]}",
                    name=f"{hazard_type.title()} Hazard",
                    coords=[hazard_coord],
                    hazard_subtype=hazard_type if hazard_type in ("lava", "acid", "radiation", "electricity") else "acid",
                    damage_type="energy" if hazard_type == "electricity" else "kinetic",
                )
                primitives.append(hazard)
                occupied.add(hazard_coord)

    return primitives


# =============================================================================
# Simple Random Terrain Generation (without SITREP)
# =============================================================================


def generate_random_terrain(
    params: TerrainGeneratorParams,
) -> GeneratedTerrain:
    """Generate random terrain without SITREP linkage.

    Uses tile set configuration to generate appropriate features.

    Args:
        params: Generation parameters (sitrep_template ignored)

    Returns:
        GeneratedTerrain with compiled map and primitives
    """
    rng = random.Random(params.seed)
    tile_set_config = TILE_SETS[params.tile_set]
    primitives: list[AnyTerrainPrimitive] = []
    primitive_counter = [0]

    # Generate features from tile set
    random_primitives = _generate_tile_set_features(
        params=params,
        tile_set_config=tile_set_config,
        existing_primitives=[],
        rng=rng,
        primitive_counter=primitive_counter,
    )
    primitives.extend(random_primitives)

    return compose_terrain_map(primitives)
