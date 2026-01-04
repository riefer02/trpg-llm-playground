"""Tests for NPC stat tier scaling."""

import pytest
from core.npc.models import (
    NPCTemplate,
    NPCStats,
    NPCStatsBase,
    NPCTierScaling,
    NPCAbility,
    NPCGear,
)
from core.npc.state import (
    NPCState,
    NPCCombatStats,
    scale_npc_stats,
    convert_to_combat_stats,
)
from core.npc.enums import NPCTier, NPCClass
from core.shared.enums import SizeClass


def _make_test_scaling() -> NPCTierScaling:
    """Create test tier scaling with known values."""
    return NPCTierScaling(
        hp_multiplier=1.5,
        hp_adder_tier_2=10,
        hp_adder_tier_3=20,
        evasion_adder_tier_2=1,
        evasion_adder_tier_3=2,
        e_defense_adder_tier_2=1,
        e_defense_adder_tier_3=2,
        armor_adder_tier_2=0,
        armor_adder_tier_3=1,
        save_adder_tier_2=1,
        save_adder_tier_3=2,
    )


def _make_test_template() -> NPCTemplate:
    """Create a test template with known stats."""
    return NPCTemplate(
        id="test_npc",
        name="Test NPC",
        npc_class="grunt",
        tier="tier_1",
        stats=NPCStats(
            base=NPCStatsBase(
                size="size_1",
                hp_base=10,
                evasion_base=8,
                e_defense_base=8,
                armor_base=0,
                speed_base=4,
                sensor_range=10,
                save_bonus=0,
            ),
            scaling=_make_test_scaling(),
        ),
        abilities=[],
        gear=[],
    )


class TestNPCTierScaling:
    """Tests for NPC tier scaling logic."""

    def test_tier_1_scaling(self) -> None:
        """Tier 1 should have no adders."""
        stats = NPCStats(
            base=NPCStatsBase(hp_base=10),
            scaling=_make_test_scaling(),
        )
        result = scale_npc_stats(stats, "tier_1")
        assert result.hp_max == 10
        assert result.evasion == 8
        assert result.e_defense == 8
        assert result.armor == 0
        assert result.save_bonus == 0

    def test_tier_2_scaling(self) -> None:
        """Tier 2 should apply tier 2 adders and multiplier."""
        stats = NPCStats(
            base=NPCStatsBase(hp_base=10),
            scaling=_make_test_scaling(),
        )
        result = scale_npc_stats(stats, "tier_2")
        assert result.hp_max == 25  # 10 * 1.5 + 10
        assert result.evasion == 9  # 8 + 1
        assert result.e_defense == 9  # 8 + 1
        assert result.armor == 0  # 0 + 0
        assert result.save_bonus == 1  # 0 + 1

    def test_tier_3_scaling(self) -> None:
        """Tier 3 should apply tier 3 adders and multiplier."""
        stats = NPCStats(
            base=NPCStatsBase(hp_base=10),
            scaling=_make_test_scaling(),
        )
        result = scale_npc_stats(stats, "tier_3")
        assert result.hp_max == 35  # 10 * 1.5 + 20
        assert result.evasion == 10  # 8 + 2
        assert result.e_defense == 10  # 8 + 2
        assert result.armor == 1  # 0 + 1
        assert result.save_bonus == 2  # 0 + 2

    def test_size_preserved(self) -> None:
        """Size class should not change with tier."""
        stats = NPCStats(
            base=NPCStatsBase(size="size_2"),
            scaling=_make_test_scaling(),
        )
        for tier in ("tier_1", "tier_2", "tier_3"):
            result = scale_npc_stats(stats, tier)
            assert result.size == "size_2"

    def test_speed_not_affected(self) -> None:
        """Speed should not have tier adders in this scaling."""
        stats = NPCStats(
            base=NPCStatsBase(speed_base=5),
            scaling=_make_test_scaling(),
        )
        for tier in ("tier_1", "tier_2", "tier_3"):
            result = scale_npc_stats(stats, tier)
            assert result.speed == 5

    def test_sensor_affected(self) -> None:
        """Sensor range should have tier 3 adder."""
        stats = NPCStats(
            base=NPCStatsBase(sensor_range=10),
            scaling=_make_test_scaling(),
        )
        tier1 = scale_npc_stats(stats, "tier_1")
        tier2 = scale_npc_stats(stats, "tier_2")
        tier3 = scale_npc_stats(stats, "tier_3")
        assert tier1.sensor_range == 10
        assert tier2.sensor_range == 10  # No tier 2 adder
        assert tier3.sensor_range == 15  # 10 + 5


class TestNPCStateFromTemplate:
    """Tests for creating NPC state from template."""

    def test_create_npc_from_template(self) -> None:
        """NPCState.from_template should create correct state."""
        template = _make_test_template()
        npc = NPCState.from_template(template, "npc_1")
        assert npc.id == "npc_1"
        assert npc.name == "Test NPC"
        assert npc.npc_class == "grunt"
        assert npc.tier == "tier_1"
        assert npc.template_id == "test_npc"
        assert npc.abilities_used == set()

    def test_npc_state_tier_scaling(self) -> None:
        """NPC state should have scaled stats."""
        template = _make_test_template()
        npc = NPCState.from_template(template, "npc_1", name="Custom Name")
        assert npc.stats.hp_max == 10  # Tier 1: 10 * 1.5 = 10
        assert npc.stats.evasion == 8

    def test_npc_state_tier_3(self) -> None:
        """NPC state tier 3 should have higher stats."""
        template = _make_test_template()
        template = NPCTemplate(
            id="test_npc_t3",
            name="Test NPC T3",
            npc_class="boss",
            tier="tier_3",
            stats=NPCStats(
                base=NPCStatsBase(
                    hp_base=10,
                    evasion_base=8,
                    e_defense_base=8,
                    armor_base=0,
                    save_bonus=0,
                ),
                scaling=_make_test_scaling(),
            ),
        )
        npc = NPCState.from_template(template, "npc_1")
        assert npc.stats.hp_max == 35  # 10 * 1.5 + 20
        assert npc.stats.evasion == 10  # 8 + 2
        assert npc.stats.e_defense == 10  # 8 + 2
        assert npc.stats.armor == 1  # 0 + 1

    def test_structure_by_tier(self) -> None:
        """Structure should match tier expectations."""
        template = _make_test_template()
        for tier in ("tier_1", "tier_2", "tier_3"):
            template = NPCTemplate(
                id=f"test_{tier}",
                name=f"Test {tier}",
                npc_class="grunt",
                tier=tier,
                stats=NPCStats(base=NPCStatsBase()),
            )
            npc = NPCState.from_template(template, f"npc_{tier}")
            if tier == "tier_1":
                assert npc.structure_current == 1
            elif tier == "tier_2":
                assert npc.structure_current == 2
            else:
                assert npc.structure_current == 3


class TestConvertToCombatStats:
    """Tests for converting NPCCombatStats to CombatStats dict."""

    def test_convert_all_fields(self) -> None:
        """All fields should convert correctly."""
        npc_stats = NPCCombatStats(
            size="size_2",
            hp_max=25,
            evasion=10,
            e_defense=12,
            armor=2,
            speed=5,
            sensor_range=15,
            tech_attack=1,
            save_bonus=2,
        )
        result = convert_to_combat_stats(npc_stats)
        assert result["size"] == "size_2"
        assert result["hp_max"] == 25
        assert result["evasion"] == 10
        assert result["e_defense"] == 12
        assert result["armor"] == 2
        assert result["speed"] == 5
        assert result["sensor_range"] == 15
        assert result["tech_attack"] == 1


class TestNPCHPProperties:
    """Tests for NPC state HP properties."""

    def test_hp_current_equals_max(self) -> None:
        """NPC HP current should equal max on creation."""
        template = _make_test_template()
        npc = NPCState.from_template(template, "npc_1")
        assert npc.hp_current == npc.stats.hp_max
