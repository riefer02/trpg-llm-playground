"""Tests for special NPC classes (Priority 57).

Tests cover:
- Victory count verification for all 15 special classes
- Template lookup and validation
- Stat scaling for special class types
- Special trait and module mechanics
- Victory point calculation for SITREP resolution
"""

import pytest
from core.npc.special_classes import (
    NPC_SPECIAL_CLASSES,
    VICTORY_COUNTS,
    get_special_class_template,
    get_special_class_by_id,
    get_victory_count,
    calculate_victory_points_from_templates,
    get_ultra_traits,
    get_veteran_traits,
    get_exotic_modules,
    get_commander_traits,
    SPECIAL_HUMAN,
    SPECIAL_INFANTRY_SQUAD_T1,
    SPECIAL_MONSTROSITY_T1,
    SPECIAL_ULTRA_T1,
    SPECIAL_ELITE_T1,
    SPECIAL_GRUNT_T1,
    SPECIAL_VETERAN_T1,
    SPECIAL_EXOTIC_T1,
    SPECIAL_DRONE_T1,
    SPECIAL_MERCENARY_T1,
    SPECIAL_COMMANDER_T1,
    SPECIAL_PIRATE_T1,
    SPECIAL_SPACER_T1,
    SPECIAL_VEHICLE_T1,
    SPECIAL_SHIP_T1,
)
from core.npc.enums import NPCSpecialClass


class TestVictoryCounts:
    """Test victory count values for all special classes per PR2 12769-12770."""

    def test_human_victory_count(self):
        """Human counts as 1.0 victory."""
        assert get_victory_count("human") == 1.0

    def test_infantry_squad_victory_count(self):
        """Infantry Squad counts as 0.25 (4 squads = 1 victory)."""
        assert get_victory_count("infantry_squad") == 0.25

    def test_monstrosity_victory_count(self):
        """Monstrosity counts as 4.0 victories."""
        assert get_victory_count("monstrosity") == 4.0

    def test_ultra_victory_count(self):
        """Ultra counts as 4.0 victories."""
        assert get_victory_count("ultra") == 4.0

    def test_elite_victory_count(self):
        """Elite counts as 0.5 victories."""
        assert get_victory_count("elite") == 0.5

    def test_grunt_victory_count(self):
        """Grunt counts as 0.25 victories."""
        assert get_victory_count("grunt") == 0.25

    def test_veteran_victory_count(self):
        """Veteran counts as 0.25 victories."""
        assert get_victory_count("veteran") == 0.25

    def test_exotic_victory_count(self):
        """Exotic counts as 1.0 victory (default)."""
        assert get_victory_count("exotic") == 1.0

    def test_drone_victory_count(self):
        """Drone counts as 0.5 victories."""
        assert get_victory_count("drone") == 0.5

    def test_mercenary_victory_count(self):
        """Mercenary counts as 1.0 victory."""
        assert get_victory_count("mercenary") == 1.0

    def test_commander_victory_count(self):
        """Commander counts as 2.0 victories."""
        assert get_victory_count("commander") == 2.0

    def test_pirate_victory_count(self):
        """Pirate counts as 0.5 victories."""
        assert get_victory_count("pirate") == 0.5

    def test_spacer_victory_count(self):
        """Spacer counts as 1.0 victory."""
        assert get_victory_count("spacer") == 1.0

    def test_vehicle_victory_count(self):
        """Vehicle counts as 1.5 victories."""
        assert get_victory_count("vehicle") == 1.5

    def test_ship_victory_count(self):
        """Ship counts as 8.0 victories."""
        assert get_victory_count("ship") == 8.0


class TestTemplateLookup:
    """Test template lookup by special class type and ID."""

    def test_get_special_class_template_human(self):
        """Can lookup Human template by special class type."""
        template = get_special_class_template("human")
        assert template is not None
        assert template.special_class == "human"
        assert template.victory_count == 1.0

    def test_get_special_class_template_ultra(self):
        """Can lookup Ultra template by special class type."""
        template = get_special_class_template("ultra")
        assert template is not None
        assert template.special_class == "ultra"
        assert len(template.ultra_traits) == 15

    def test_get_special_class_template_infantry_squad(self):
        """Can lookup Infantry Squad template."""
        template = get_special_class_template("infantry_squad")
        assert template is not None
        assert template.infantry_squad_stats is not None
        assert template.infantry_squad_stats.squad_members == 5

    def test_get_special_class_template_exotic(self):
        """Can lookup Exotic template with modules."""
        template = get_special_class_template("exotic")
        assert template is not None
        assert len(template.exotic_modules) == 7

    def test_get_special_class_template_commander(self):
        """Can lookup Commander template with traits."""
        template = get_special_class_template("commander")
        assert template is not None
        assert len(template.commander_traits) == 5

    def test_get_special_class_template_invalid(self):
        """Returns None for invalid special class type."""
        template = get_special_class_template("invalid_class")
        assert template is None

    def test_get_special_class_by_id_ultra(self):
        """Can lookup Ultra template by ID."""
        template = get_special_class_by_id("special_ultra_t1")
        assert template is not None
        assert template.id == "special_ultra_t1"

    def test_get_special_class_by_id_invalid(self):
        """Returns None for invalid template ID."""
        template = get_special_class_by_id("invalid_id")
        assert template is None


class TestStatScaling:
    """Test stat scaling for special class types."""

    def test_ultra_tier_1_scaling(self):
        """Ultra T1 has +5 bonus HP."""
        assert SPECIAL_ULTRA_T1.bonus_hp == 5
        assert SPECIAL_ULTRA_T1.structure_override == 4
        assert SPECIAL_ULTRA_T1.stress_override == 4

    def test_ultra_tier_1_stats(self):
        """Ultra T1 base stats are correct."""
        assert SPECIAL_ULTRA_T1.stats.base.hp_base == 20
        assert SPECIAL_ULTRA_T1.stats.base.evasion_base == 10
        assert SPECIAL_ULTRA_T1.stats.base.armor_base == 2

    def test_elite_tier_1_scaling(self):
        """Elite T1 has structure=2, stress=2."""
        assert SPECIAL_ELITE_T1.structure_override == 2
        assert SPECIAL_ELITE_T1.stress_override == 2

    def test_grunt_tier_1_scaling(self):
        """Grunt T1 has 1 HP, 1 structure, 1 stress."""
        assert SPECIAL_GRUNT_T1.stats.base.hp_base == 10
        assert SPECIAL_GRUNT_T1.structure_override == 1
        assert SPECIAL_GRUNT_T1.stress_override == 1

    def test_veteran_tier_1_scaling(self):
        """Veteran T1 has +1 structure, +1 stress."""
        assert SPECIAL_VETERAN_T1.structure_override == 2
        assert SPECIAL_VETERAN_T1.stress_override == 2

    def test_drone_tier_1_scaling(self):
        """Drone T1 has +5 bonus HP."""
        assert SPECIAL_DRONE_T1.bonus_hp == 5
        assert SPECIAL_DRONE_T1.stats.base.size == "size_half"

    def test_vehicle_tier_1_scaling(self):
        """Vehicle T1 has correct size and stats."""
        assert SPECIAL_VEHICLE_T1.vehicle_type is not None
        assert "flier" in SPECIAL_VEHICLE_T1.vehicle_type

    def test_ship_tier_1_scaling(self):
        """Ship T1 has min size 4 and +5 bonus HP."""
        assert SPECIAL_SHIP_T1.stats.base.size == "size_4"
        assert SPECIAL_SHIP_T1.bonus_hp == 5

    def test_infantry_squad_stats(self):
        """Infantry Squad has squad tracking."""
        assert SPECIAL_INFANTRY_SQUAD_T1.infantry_squad_stats is not None
        assert SPECIAL_INFANTRY_SQUAD_T1.infantry_squad_stats.squad_members == 5
        assert SPECIAL_INFANTRY_SQUAD_T1.structure_override == 1

    def test_monstrosity_tier_1_stats(self):
        """Monstrosity T1 has biological tag and correct stats."""
        assert "biological" in SPECIAL_MONSTROSITY_T1.tags
        assert SPECIAL_MONSTROSITY_T1.stats.base.size == "size_2"
        assert SPECIAL_MONSTROSITY_T1.stats.base.hp_base == 30


class TestSpecialTraits:
    """Test special trait and module implementations."""

    def test_ultra_all_traits_present(self):
        """All 15 Ultra traits are implemented."""
        traits = get_ultra_traits()
        assert len(traits) == 15
        trait_types = [t.trait_type for t in traits]
        assert "berserker" in trait_types
        assert "devastator" in trait_types
        assert "evasive" in trait_types
        assert "extra_deadly" in trait_types
        assert "fortress" in trait_types
        assert "legion" in trait_types
        assert "limitless" in trait_types
        assert "unstoppable" in trait_types
        assert "sight" in trait_types
        assert "superior_construction" in trait_types
        assert "superior_frame" in trait_types
        assert "superior_reactor" in trait_types
        assert "superior_targeting" in trait_types
        assert "supreme_maintenance" in trait_types
        assert "supreme_skirmisher" in trait_types

    def test_veteran_traits_present(self):
        """Veteran traits are implemented."""
        traits = get_veteran_traits()
        assert len(traits) == 5
        trait_types = [t.trait_type for t in traits]
        assert "deadly" in trait_types
        assert "hardened_target" in trait_types
        assert "limitless" in trait_types
        assert "self_repair" in trait_types
        assert "skirmisher" in trait_types

    def test_exotic_modules_present(self):
        """All 7 Exotic modules are implemented."""
        modules = get_exotic_modules()
        assert len(modules) == 7
        module_types = [m.module_type for m in modules]
        assert "bio_integrated" in module_types
        assert "blinkspace_carver" in module_types
        assert "extrusion" in module_types
        assert "living_weaponry" in module_types
        assert "paracausal_weapon" in module_types
        assert "ouroboros_brand" in module_types
        assert "regenerator" in module_types

    def test_commander_traits_present(self):
        """All 5 Commander traits are implemented."""
        traits = get_commander_traits()
        assert len(traits) == 5
        trait_types = [t.trait_type for t in traits]
        assert "bolster_network" in trait_types
        assert "retribution" in trait_types
        assert "press_on" in trait_types
        assert "reposition" in trait_types
        assert "rank_and_file" in trait_types


class TestVictoryPointCalculation:
    """Test victory point calculation for SITREP resolution."""

    def test_single_ultra_victory_points(self):
        """One Ultra = 4.0 victory points."""
        total = calculate_victory_points_from_templates([SPECIAL_ULTRA_T1])
        assert total == 4.0

    def test_four_grunts_victory_points(self):
        """Four Grunts = 1.0 victory point (4 x 0.25)."""
        grunts = [SPECIAL_GRUNT_T1] * 4
        total = calculate_victory_points_from_templates(grunts)
        assert total == 1.0

    def test_mixed_npcs_victory_points(self):
        """Mixed NPCs calculate correctly."""
        templates = [
            SPECIAL_ULTRA_T1,  # 4.0
            SPECIAL_COMMANDER_T1,  # 2.0
            SPECIAL_ELITE_T1,  # 0.5
            SPECIAL_GRUNT_T1,  # 0.25
        ]
        total = calculate_victory_points_from_templates(templates)
        assert total == 6.75

    def test_empty_list_victory_points(self):
        """Empty list = 0 victory points."""
        total = calculate_victory_points_from_templates([])
        assert total == 0.0

    def test_infantry_squad_aggregation(self):
        """4 Infantry Squads = 1 victory point."""
        squads = [SPECIAL_INFANTRY_SQUAD_T1] * 4
        total = calculate_victory_points_from_templates(squads)
        assert total == 1.0


class TestTemplateValidation:
    """Test that all templates have valid required fields."""

    def test_all_templates_have_ids(self):
        """All special class templates have unique IDs."""
        ids = [t.id for t in NPC_SPECIAL_CLASSES]
        assert len(ids) == len(set(ids))

    def test_all_templates_have_special_class(self):
        """All templates have special_class set."""
        for template in NPC_SPECIAL_CLASSES:
            assert template.special_class is not None

    def test_all_templates_have_victory_count(self):
        """All templates have victory_count > 0."""
        for template in NPC_SPECIAL_CLASSES:
            assert template.victory_count > 0

    def test_all_15_special_classes_exist(self):
        """All 15 special classes are implemented."""
        assert len(NPC_SPECIAL_CLASSES) == 15
        special_classes = {t.special_class for t in NPC_SPECIAL_CLASSES}
        expected = {
            "human",
            "infantry_squad",
            "monstrosity",
            "ultra",
            "elite",
            "grunt",
            "veteran",
            "exotic",
            "drone",
            "mercenary",
            "commander",
            "pirate",
            "spacer",
            "vehicle",
            "ship",
        }
        assert special_classes == expected

    def test_victory_counts_dict_complete(self):
        """VICTORY_COUNTS dict has all 15 special classes."""
        assert len(VICTORY_COUNTS) == 15


class TestInfantrySquadMechanics:
    """Test Infantry Squad specific mechanics."""

    def test_squad_members_range(self):
        """Squad members should be 5-10."""
        assert SPECIAL_INFANTRY_SQUAD_T1.infantry_squad_stats.squad_members == 5

    def test_squad_has_biological_tag(self):
        """Infantry Squad has biological tag."""
        assert "biological" in SPECIAL_INFANTRY_SQUAD_T1.tags
        assert "infantry" in SPECIAL_INFANTRY_SQUAD_T1.tags

    def test_squad_single_structure(self):
        """Infantry Squad has only 1 structure."""
        assert SPECIAL_INFANTRY_SQUAD_T1.structure_override == 1

    def test_squad_has_resistance_effect(self):
        """Infantry Squad has resistance to non-AOE damage."""
        effects = SPECIAL_INFANTRY_SQUAD_T1.effects
        assert len(effects.resistances) > 0


class TestUltraMechanics:
    """Test Ultra specific mechanics."""

    def test_ultra_has_15_traits(self):
        """Ultra has all 15 traits."""
        assert len(SPECIAL_ULTRA_T1.ultra_traits) == 15

    def test_ultra_structure_and_stress(self):
        """Ultra has structure=4, stress=4."""
        assert SPECIAL_ULTRA_T1.structure_override == 4
        assert SPECIAL_ULTRA_T1.stress_override == 4

    def test_ultra_has_bonus_hp(self):
        """Ultra has +5 bonus HP."""
        assert SPECIAL_ULTRA_T1.bonus_hp == 5

    def test_ultra_has_ultra_tag(self):
        """Ultra has ultra tag."""
        assert "ultra" in SPECIAL_ULTRA_T1.tags


class TestVehicleAndShipMechanics:
    """Test Vehicle and Ship specific mechanics."""

    def test_vehicle_has_vehicle_tag(self):
        """Vehicle has vehicle tag."""
        assert "vehicle" in SPECIAL_VEHICLE_T1.tags

    def test_ship_has_ship_and_vehicle_tags(self):
        """Ship has both ship and vehicle tags."""
        assert "ship" in SPECIAL_SHIP_T1.tags
        assert "vehicle" in SPECIAL_SHIP_T1.tags

    def test_ship_minimum_size_4(self):
        """Ship has minimum size 4."""
        assert SPECIAL_SHIP_T1.stats.base.size == "size_4"

    def test_ship_has_bonus_hp(self):
        """Ship has +5 bonus HP."""
        assert SPECIAL_SHIP_T1.bonus_hp == 5
