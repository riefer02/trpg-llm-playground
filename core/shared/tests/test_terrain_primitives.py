"""Unit tests for terrain primitives and material system."""

import pytest
from core.mech.grid import HexCoord, HexPosition
from core.shared.terrain_primitives import (
    # Materials
    MaterialProperties,
    MATERIAL_ORGANIC,
    MATERIAL_TOUGH,
    MATERIAL_HARDY,
    MATERIAL_FORTIFIED,
    MATERIAL_ARMORED,
    get_default_material,
    # Primitives
    TerrainPrimitive,
    FloorTile,
    Obstacle,
    SoftCoverZone,
    Hazard,
    Objective,
    # Destructible
    DestructibleTerrainState,
    damage_destructible_terrain,
    create_destructible_state,
    # Composition
    compose_terrain_map,
)


# =============================================================================
# Material System Tests
# =============================================================================


class TestMaterialProperties:
    """Tests for MaterialProperties model."""

    def test_material_defaults(self) -> None:
        """MaterialProperties has sensible defaults."""
        mat = MaterialProperties(material_type="hardy")
        assert mat.armor == 0
        assert mat.hp_per_size == 10
        assert mat.evasion == 5
        assert mat.is_flammable is False

    def test_material_with_armor(self) -> None:
        """MaterialProperties accepts armor value."""
        mat = MaterialProperties(material_type="fortified", armor=3)
        assert mat.armor == 3

    def test_material_armor_bounds(self) -> None:
        """MaterialProperties enforces armor bounds 0-4."""
        with pytest.raises(ValueError):
            MaterialProperties(material_type="armored", armor=5)

        with pytest.raises(ValueError):
            MaterialProperties(material_type="armored", armor=-1)

    def test_flammable_material(self) -> None:
        """MaterialProperties can be flammable."""
        mat = MaterialProperties(material_type="organic", is_flammable=True)
        assert mat.is_flammable is True


class TestMaterialConstants:
    """Tests for default material constants matching PR2 values."""

    def test_organic_is_armor_0(self) -> None:
        """Organic material has armor 0 per PR2."""
        assert MATERIAL_ORGANIC.armor == 0
        assert MATERIAL_ORGANIC.is_flammable is True

    def test_tough_is_armor_1(self) -> None:
        """Tough material has armor 1 per PR2."""
        assert MATERIAL_TOUGH.armor == 1

    def test_hardy_is_armor_2(self) -> None:
        """Hardy material has armor 2 per PR2."""
        assert MATERIAL_HARDY.armor == 2

    def test_fortified_is_armor_3(self) -> None:
        """Fortified material has armor 3 per PR2."""
        assert MATERIAL_FORTIFIED.armor == 3

    def test_armored_is_armor_4(self) -> None:
        """Armored material has armor 4 per PR2."""
        assert MATERIAL_ARMORED.armor == 4

    def test_all_materials_have_10_hp_per_size(self) -> None:
        """All materials have 10 HP per size per PR2."""
        for mat in [
            MATERIAL_ORGANIC,
            MATERIAL_TOUGH,
            MATERIAL_HARDY,
            MATERIAL_FORTIFIED,
            MATERIAL_ARMORED,
        ]:
            assert mat.hp_per_size == 10

    def test_all_materials_have_evasion_5(self) -> None:
        """All materials have evasion 5 per PR2."""
        for mat in [
            MATERIAL_ORGANIC,
            MATERIAL_TOUGH,
            MATERIAL_HARDY,
            MATERIAL_FORTIFIED,
            MATERIAL_ARMORED,
        ]:
            assert mat.evasion == 5


class TestGetDefaultMaterial:
    """Tests for get_default_material helper."""

    def test_get_organic(self) -> None:
        """get_default_material returns correct organic material."""
        mat = get_default_material("organic")
        assert mat == MATERIAL_ORGANIC

    def test_get_all_materials(self) -> None:
        """get_default_material works for all material types."""
        assert get_default_material("organic") == MATERIAL_ORGANIC
        assert get_default_material("tough") == MATERIAL_TOUGH
        assert get_default_material("hardy") == MATERIAL_HARDY
        assert get_default_material("fortified") == MATERIAL_FORTIFIED
        assert get_default_material("armored") == MATERIAL_ARMORED


# =============================================================================
# Terrain Primitive Tests
# =============================================================================


class TestTerrainPrimitive:
    """Tests for base TerrainPrimitive model."""

    def test_basic_primitive(self) -> None:
        """TerrainPrimitive can be created with minimal fields."""
        prim = TerrainPrimitive(
            id="test_1",
            kind="floor",
            name="Test Floor",
        )
        assert prim.id == "test_1"
        assert prim.kind == "floor"
        assert prim.name == "Test Floor"
        assert prim.coords == []

    def test_primitive_with_coords(self) -> None:
        """TerrainPrimitive accepts coordinates."""
        coords = [HexCoord(q=0, r=0), HexCoord(q=1, r=0)]
        prim = TerrainPrimitive(
            id="test_2",
            kind="obstacle",
            name="Wall",
            coords=coords,
        )
        assert len(prim.coords) == 2

    def test_primitive_with_terrain_flags(self) -> None:
        """TerrainPrimitive accepts terrain flags."""
        prim = TerrainPrimitive(
            id="test_3",
            kind="obstacle",
            name="Rock",
            elevation=2,
            blocks_line_of_sight=True,
            provides_hard_cover=True,
            hard_cover_size="size_2",
        )
        assert prim.elevation == 2
        assert prim.blocks_line_of_sight is True
        assert prim.provides_hard_cover is True
        assert prim.hard_cover_size == "size_2"


class TestFloorTile:
    """Tests for FloorTile primitive."""

    def test_normal_floor(self) -> None:
        """Normal floor is not difficult or dangerous."""
        floor = FloorTile(
            id="floor_1",
            name="Normal Floor",
            coords=[HexCoord(q=0, r=0)],
            floor_type="normal",
        )
        assert floor.difficult is False
        assert floor.dangerous is False

    def test_difficult_floor(self) -> None:
        """Difficult floor type sets difficult flag."""
        floor = FloorTile(
            id="floor_2",
            name="Rubble",
            coords=[HexCoord(q=0, r=0)],
            floor_type="difficult",
        )
        assert floor.difficult is True
        assert floor.dangerous is False

    def test_dangerous_floor(self) -> None:
        """Dangerous floor type sets dangerous flag."""
        floor = FloorTile(
            id="floor_3",
            name="Hazardous",
            coords=[HexCoord(q=0, r=0)],
            floor_type="dangerous",
        )
        assert floor.difficult is False
        assert floor.dangerous is True

    def test_climbing_floor(self) -> None:
        """Climbing floor type sets difficult flag."""
        floor = FloorTile(
            id="floor_4",
            name="Climbing Surface",
            coords=[HexCoord(q=0, r=0)],
            floor_type="climbing",
        )
        assert floor.difficult is True
        assert floor.dangerous is False


class TestObstacle:
    """Tests for Obstacle primitive."""

    def test_default_obstacle(self) -> None:
        """Obstacle has sensible defaults."""
        obs = Obstacle(
            id="obs_1",
            name="Wall",
            coords=[HexCoord(q=0, r=0)],
        )
        assert obs.blocks_line_of_sight is True
        assert obs.provides_hard_cover is True
        assert obs.hard_cover_size == "size_1"
        assert obs.size == 1
        assert obs.is_destructible is True

    def test_obstacle_hp_calculation(self) -> None:
        """Obstacle max_hp is 10 * size per PR2."""
        obs = Obstacle(
            id="obs_2",
            name="Large Wall",
            coords=[HexCoord(q=0, r=0)],
            size=3,
        )
        assert obs.max_hp == 30  # 10 * 3

    def test_obstacle_hp_with_material(self) -> None:
        """Obstacle uses material hp_per_size for calculation."""
        custom_material = MaterialProperties(
            material_type="hardy",
            hp_per_size=15,
        )
        obs = Obstacle(
            id="obs_3",
            name="Custom Wall",
            coords=[HexCoord(q=0, r=0)],
            size=2,
            material=custom_material,
        )
        assert obs.max_hp == 30  # 15 * 2

    def test_obstacle_explicit_hp(self) -> None:
        """Obstacle explicit hp overrides calculation."""
        obs = Obstacle(
            id="obs_4",
            name="Special Wall",
            coords=[HexCoord(q=0, r=0)],
            size=2,
            hp=50,
        )
        assert obs.max_hp == 50


class TestSoftCoverZone:
    """Tests for SoftCoverZone primitive."""

    def test_default_soft_cover_zone(self) -> None:
        """SoftCoverZone has sensible defaults."""
        zone = SoftCoverZone(
            id="zone_1",
            name="Smoke Cloud",
            coords=[HexCoord(q=0, r=0)],
        )
        assert zone.provides_soft_cover is True
        assert zone.provides_hard_cover is False
        assert zone.zone_subtype == "smoke"
        assert zone.duration_rounds is None

    def test_soft_cover_zone_with_duration(self) -> None:
        """SoftCoverZone can have duration."""
        zone = SoftCoverZone(
            id="zone_2",
            name="Smoke Grenade",
            coords=[HexCoord(q=0, r=0)],
            duration_rounds=3,
            created_round=1,
        )
        assert zone.duration_rounds == 3
        assert zone.created_round == 1

    def test_soft_cover_zone_subtypes(self) -> None:
        """SoftCoverZone accepts different subtypes."""
        for subtype in ["smoke", "foliage", "mist", "darkness"]:
            zone = SoftCoverZone(
                id=f"zone_{subtype}",
                name=subtype.title(),
                coords=[HexCoord(q=0, r=0)],
                zone_subtype=subtype,
            )
            assert zone.zone_subtype == subtype


class TestHazard:
    """Tests for Hazard primitive."""

    def test_default_hazard(self) -> None:
        """Hazard has sensible defaults per PR2."""
        hazard = Hazard(
            id="hazard_1",
            name="Lava Pool",
            coords=[HexCoord(q=0, r=0)],
        )
        assert hazard.dangerous is True
        assert hazard.damage == 5
        assert hazard.check_dc == 10
        assert hazard.damage_type == "kinetic"
        assert hazard.hazard_subtype == "lava"

    def test_hazard_custom_damage(self) -> None:
        """Hazard can have custom damage."""
        hazard = Hazard(
            id="hazard_2",
            name="Radiation Zone",
            coords=[HexCoord(q=0, r=0)],
            hazard_subtype="radiation",
            damage=10,
            damage_type="energy",
        )
        assert hazard.damage == 10
        assert hazard.damage_type == "energy"


class TestObjective:
    """Tests for Objective primitive."""

    def test_default_objective(self) -> None:
        """Objective has sensible defaults."""
        obj = Objective(
            id="obj_1",
            name="Control Point A",
            coords=[HexCoord(q=0, r=0)],
        )
        assert obj.objective_type == "control_point"
        assert obj.zone_id is None
        assert obj.dangerous is False

    def test_objective_types(self) -> None:
        """Objective accepts different types."""
        for obj_type in ["control_point", "escort_target", "extraction", "ingress"]:
            obj = Objective(
                id=f"obj_{obj_type}",
                name=obj_type.replace("_", " ").title(),
                coords=[HexCoord(q=0, r=0)],
                objective_type=obj_type,
            )
            assert obj.objective_type == obj_type


# =============================================================================
# Destructible Terrain Tests
# =============================================================================


class TestDestructibleTerrainState:
    """Tests for DestructibleTerrainState model."""

    def test_basic_state(self) -> None:
        """DestructibleTerrainState can be created."""
        state = DestructibleTerrainState(
            primitive_id="wall_1",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
            hp=20,
            max_hp=20,
        )
        assert state.hp == 20
        assert state.is_destroyed is False

    def test_destroyed_state(self) -> None:
        """DestructibleTerrainState can be destroyed."""
        state = DestructibleTerrainState(
            primitive_id="wall_1",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
            hp=0,
            max_hp=20,
            is_destroyed=True,
        )
        assert state.is_destroyed is True


class TestDamageDestructibleTerrain:
    """Tests for damage_destructible_terrain function."""

    def test_basic_damage(self) -> None:
        """Basic damage reduces HP."""
        state = DestructibleTerrainState(
            primitive_id="wall_1",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
            hp=20,
            max_hp=20,
            armor=0,
        )
        new_state, destroyed = damage_destructible_terrain(state, 5)
        assert new_state.hp == 15
        assert destroyed is False

    def test_armor_reduces_damage(self) -> None:
        """Armor reduces incoming damage."""
        state = DestructibleTerrainState(
            primitive_id="wall_1",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
            hp=20,
            max_hp=20,
            armor=3,
        )
        new_state, destroyed = damage_destructible_terrain(state, 5)
        assert new_state.hp == 18  # 5 - 3 = 2 damage
        assert destroyed is False

    def test_armor_piercing(self) -> None:
        """Armor piercing ignores armor."""
        state = DestructibleTerrainState(
            primitive_id="wall_1",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
            hp=20,
            max_hp=20,
            armor=3,
        )
        new_state, destroyed = damage_destructible_terrain(state, 5, armor_piercing=2)
        assert new_state.hp == 16  # armor 3-2=1, damage 5-1=4

    def test_destruction(self) -> None:
        """Terrain is destroyed when HP reaches 0."""
        state = DestructibleTerrainState(
            primitive_id="wall_1",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
            hp=5,
            max_hp=20,
            armor=0,
        )
        new_state, destroyed = damage_destructible_terrain(state, 10)
        assert new_state.hp == 0
        assert new_state.is_destroyed is True
        assert destroyed is True

    def test_destroyed_terrain_not_damaged_again(self) -> None:
        """Already destroyed terrain cannot be damaged."""
        state = DestructibleTerrainState(
            primitive_id="wall_1",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
            hp=0,
            max_hp=20,
            is_destroyed=True,
        )
        new_state, destroyed = damage_destructible_terrain(state, 10)
        assert new_state.hp == 0
        assert destroyed is False  # Was already destroyed

    def test_destruction_removes_cover(self) -> None:
        """Destroyed terrain no longer provides cover."""
        state = DestructibleTerrainState(
            primitive_id="wall_1",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
            hp=5,
            max_hp=20,
            provides_hard_cover=True,
            hard_cover_size="size_1",
        )
        new_state, destroyed = damage_destructible_terrain(state, 10)
        assert new_state.provides_hard_cover is False
        assert new_state.hard_cover_size is None


class TestCreateDestructibleState:
    """Tests for create_destructible_state function."""

    def test_basic_creation(self) -> None:
        """create_destructible_state creates state from Obstacle."""
        obs = Obstacle(
            id="wall_1",
            name="Wall",
            coords=[HexCoord(q=0, r=0)],
            size=2,
        )
        position = HexPosition(coord=HexCoord(q=0, r=0))
        state = create_destructible_state(obs, position)

        assert state.primitive_id == "wall_1"
        assert state.hp == 20  # 10 * 2
        assert state.max_hp == 20
        assert state.armor == 2  # MATERIAL_HARDY default
        assert state.is_destroyed is False

    def test_creation_with_material(self) -> None:
        """create_destructible_state uses material properties."""
        obs = Obstacle(
            id="wall_1",
            name="Fortified Wall",
            coords=[HexCoord(q=0, r=0)],
            size=2,
            material=MATERIAL_FORTIFIED,
        )
        position = HexPosition(coord=HexCoord(q=0, r=0))
        state = create_destructible_state(obs, position)

        assert state.armor == 3  # MATERIAL_FORTIFIED armor


# =============================================================================
# Composition Tests
# =============================================================================


class TestComposeTerrainMap:
    """Tests for compose_terrain_map function."""

    def test_empty_composition(self) -> None:
        """Empty primitive list produces empty terrain map."""
        result = compose_terrain_map([])
        assert len(result.terrain_map.tiles) == 0
        assert len(result.primitives) == 0

    def test_single_floor_tile(self) -> None:
        """Single floor tile produces correct terrain."""
        floor = FloorTile(
            id="floor_1",
            name="Floor",
            coords=[HexCoord(q=0, r=0)],
        )
        result = compose_terrain_map([floor])
        assert len(result.terrain_map.tiles) == 1
        assert result.terrain_map.tiles[0].coord == HexCoord(q=0, r=0)

    def test_obstacle_composition(self) -> None:
        """Obstacle composition produces terrain with cover flags."""
        obs = Obstacle(
            id="wall_1",
            name="Wall",
            coords=[HexCoord(q=0, r=0)],
        )
        result = compose_terrain_map([obs])
        tile = result.terrain_map.tiles[0]
        assert tile.blocks_line_of_sight is True
        assert tile.provides_hard_cover is True

    def test_overlapping_primitives(self) -> None:
        """Later primitives override earlier for same coord."""
        floor = FloorTile(
            id="floor_1",
            name="Floor",
            coords=[HexCoord(q=0, r=0)],
            floor_type="normal",
        )
        hazard = Hazard(
            id="hazard_1",
            name="Hazard",
            coords=[HexCoord(q=0, r=0)],
        )
        # Hazard placed after floor should override
        result = compose_terrain_map([floor, hazard])
        assert len(result.terrain_map.tiles) == 1
        tile = result.terrain_map.tiles[0]
        assert tile.dangerous is True

    def test_multiple_coords(self) -> None:
        """Primitive with multiple coords creates multiple tiles."""
        obs = Obstacle(
            id="wall_1",
            name="Long Wall",
            coords=[
                HexCoord(q=0, r=0),
                HexCoord(q=1, r=0),
                HexCoord(q=2, r=0),
            ],
        )
        result = compose_terrain_map([obs])
        assert len(result.terrain_map.tiles) == 3

    def test_soft_cover_zones_collected(self) -> None:
        """Soft cover zones are collected separately."""
        zone = SoftCoverZone(
            id="smoke_1",
            name="Smoke",
            coords=[HexCoord(q=0, r=0)],
        )
        result = compose_terrain_map([zone])
        assert len(result.soft_cover_zones) == 1
        assert result.soft_cover_zones[0].id == "smoke_1"

    def test_destructibles_collected(self) -> None:
        """Destructible obstacles create state entries."""
        obs = Obstacle(
            id="wall_1",
            name="Wall",
            coords=[HexCoord(q=0, r=0)],
            is_destructible=True,
        )
        result = compose_terrain_map([obs])
        assert len(result.destructibles) == 1
        assert result.destructibles[0].primitive_id == "wall_1"

    def test_zones_tracked_by_id(self) -> None:
        """Zone primitives are tracked by ID."""
        floor = FloorTile(
            id="deploy_1",
            name="Deployment",
            coords=[HexCoord(q=0, r=0), HexCoord(q=1, r=0)],
            zone_type="deployment",
        )
        result = compose_terrain_map([floor])
        assert "deploy_1" in result.zones
        assert len(result.zones["deploy_1"]) == 2


class TestGeneratedTerrain:
    """Tests for GeneratedTerrain model."""

    def test_terrain_map_usable(self) -> None:
        """Generated terrain map can be used with existing functions."""
        from core.shared.terrain import calculate_movement_cost, get_terrain_at

        floor = FloorTile(
            id="floor_1",
            name="Difficult Floor",
            coords=[HexCoord(q=0, r=0)],
            floor_type="difficult",
        )
        result = compose_terrain_map([floor])

        # Verify terrain map works with existing functions
        terrain_at = get_terrain_at(result.terrain_map, HexCoord(q=0, r=0))
        assert terrain_at is not None
        assert terrain_at.difficult is True

        cost = calculate_movement_cost(1, result.terrain_map, HexCoord(q=0, r=0))
        assert cost == 2  # Difficult terrain costs 2x

    def test_cover_calculation_works(self) -> None:
        """Generated terrain works with cover calculation."""
        from core.shared.terrain import check_hard_cover_available

        obs = Obstacle(
            id="wall_1",
            name="Wall",
            coords=[HexCoord(q=1, r=0)],
            hard_cover_size="size_1",
        )
        result = compose_terrain_map([obs])

        # Check hard cover from adjacent hex
        cover_result = check_hard_cover_available(
            terrain=result.terrain_map,
            attacker_coord=HexCoord(q=5, r=0),
            target_coord=HexCoord(q=0, r=0),  # Adjacent to wall at 1,0
            target_size="size_1",
        )
        assert cover_result.available is True
