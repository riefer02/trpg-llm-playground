"""Tests for pilot talent integration in combat (Phase 32).

Tests that pilot talents are properly:
1. Collected from pilots via collect_pilot_talent_effects()
2. Applied to CombatantState when creating combatants
3. Evaluated during combat for accuracy/difficulty modifiers
"""

import pytest
from core.mech.combat_state import (
    MechCombatScenario,
    CombatantState,
    CombatStats,
    CombatResources,
    CombatSide,
)
from core.mech.grid import HexPosition, HexCoord
from core.mech.combat_helpers import (
    _evaluate_condition,
    _get_talent_accuracy_modifiers,
    _get_talent_check_modifiers,
)
from core.shared.effects import (
    MechanicalEffect,
    AccuracyModifier,
    CheckModifierEffect,
    SpatialCondition,
    AttackContextCondition,
    SizeCondition,
    ConditionGroup,
)
from core.pilot import (
    Pilot,
    Talent,
    TalentDefinition,
    TalentRank,
    collect_pilot_talent_effects,
    get_talent_definition,
)
from core.mech.frame import (
    MechFrameDefinition,
    MechFrameBaseStats,
    FrameTrait,
    CoreSystemDefinition,
    collect_frame_trait_effects,
    get_core_power_effects,
)
from core.mech.combat_execution import (
    execute_action,
    get_available_actions,
    ActionExecutionInput,
)
from core.mech.action_economy import ActionEconomyState
from core.mech.combat_state import CombatTurn


# =============================================================================
# Test Fixtures
# =============================================================================


def make_combatant(
    id: str = "mech_1",
    name: str = "Test Mech",
    side: CombatSide = "players",
    hp_max: int = 10,
    hp_current: int = 10,
    talent_effects: list[MechanicalEffect] | None = None,
    frame_trait_effects: list[MechanicalEffect] | None = None,
    core_power_available: bool = True,
    core_power_active: bool = False,
    core_power_effects: MechanicalEffect | None = None,
    **kwargs,
) -> CombatantState:
    """Create a test combatant with optional talent effects."""
    position = kwargs.pop("position", HexPosition(coord=HexCoord(q=0, r=0), elevation=0))
    return CombatantState(
        id=id,
        name=name,
        side=side,
        kind="mech",
        stats=CombatStats(
            size="size_1",
            hp_max=hp_max,
            evasion=8,
            e_defense=8,
            armor=0,
            speed=4,
            sensor_range=10,
            tech_attack=0,
            grit=0,
        ),
        resources=CombatResources(
            hp_current=hp_current,
            heat_current=0,
            heat_cap=6,
            structure_current=4,
            stress_current=4,
            repairs_remaining=4,
        ),
        position=position,
        talent_effects=talent_effects or [],
        frame_trait_effects=frame_trait_effects or [],
        core_power_available=core_power_available,
        core_power_active=core_power_active,
        core_power_effects=core_power_effects,
        **kwargs,
    )


# =============================================================================
# Test collect_pilot_talent_effects
# =============================================================================


class TestCollectPilotTalentEffects:
    """Tests for collect_pilot_talent_effects() function."""

    def test_no_talents_returns_empty(self):
        """A pilot with no talents returns empty effects list."""
        pilot = Pilot(callsign="TestPilot")
        effects = collect_pilot_talent_effects(pilot)
        assert effects == []

    def test_single_talent_rank_1(self):
        """A pilot with a single rank 1 talent gets rank 1 effects."""
        pilot = Pilot(
            callsign="TestPilot",
            talents=[Talent(talent_id="ace", rank=1)],
        )
        effects = collect_pilot_talent_effects(pilot)
        assert len(effects) == 1
        # ACE rank 1 has check_mods and triggered_effects
        assert len(effects[0].check_mods) > 0 or len(effects[0].triggered_effects) > 0

    def test_single_talent_rank_2_includes_both_ranks(self):
        """A pilot with rank 2 talent gets effects from ranks 1 AND 2."""
        pilot = Pilot(
            callsign="TestPilot",
            talents=[Talent(talent_id="ace", rank=2)],
        )
        effects = collect_pilot_talent_effects(pilot)
        # Should have 2 MechanicalEffects (one per rank)
        assert len(effects) == 2

    def test_multiple_talents(self):
        """A pilot with multiple talents gets effects from all of them."""
        pilot = Pilot(
            callsign="TestPilot",
            talents=[
                Talent(talent_id="ace", rank=1),
                Talent(talent_id="brutal", rank=1),
            ],
        )
        effects = collect_pilot_talent_effects(pilot)
        # Should have 2 effects (one from each talent)
        assert len(effects) == 2

    def test_unknown_talent_id_skipped(self):
        """Unknown talent IDs are gracefully skipped."""
        pilot = Pilot(
            callsign="TestPilot",
            talents=[Talent(talent_id="nonexistent_talent", rank=1)],
        )
        effects = collect_pilot_talent_effects(pilot)
        assert effects == []


# =============================================================================
# Test _evaluate_condition
# =============================================================================


class TestEvaluateCondition:
    """Tests for condition evaluation."""

    def test_none_condition_returns_true(self):
        """None condition always passes."""
        assert _evaluate_condition(None, {}) is True

    def test_string_condition_engaged(self):
        """String condition 'engaged' checks context."""
        assert _evaluate_condition("engaged", {"is_engaged": True}) is True
        assert _evaluate_condition("engaged", {"is_engaged": False}) is False
        assert _evaluate_condition("engaged", {}) is False

    def test_string_condition_while_flying(self):
        """String condition 'while_flying' checks context."""
        assert _evaluate_condition("while_flying", {"is_flying": True}) is True
        assert _evaluate_condition("while_flying", {"is_flying": False}) is False

    def test_string_condition_melee_attack(self):
        """String condition for melee attack."""
        assert _evaluate_condition("melee_attack", {"is_melee": True}) is True
        assert _evaluate_condition("melee_attack", {"is_melee": False}) is False

    def test_string_condition_ranged_attack(self):
        """String condition for ranged attack."""
        assert _evaluate_condition("ranged_attack", {"is_ranged": True}) is True
        assert _evaluate_condition("ranged_attack", {"is_ranged": False}) is False

    def test_unknown_string_condition_returns_false(self):
        """Unknown string conditions return False."""
        assert _evaluate_condition("unknown_condition", {}) is False

    def test_spatial_condition_adjacent(self):
        """SpatialCondition with adjacent relation."""
        cond = SpatialCondition(relation="adjacent", target="ally")
        assert _evaluate_condition(cond, {"is_adjacent": True}) is True
        assert _evaluate_condition(cond, {"is_adjacent": False}) is False

    def test_attack_context_condition(self):
        """AttackContextCondition checks attack types."""
        # applies_to="incoming" means checking incoming attacks, so is_incoming must be True
        cond = AttackContextCondition(attack_types=["ranged"], applies_to="incoming")
        assert _evaluate_condition(cond, {"attack_type": "ranged", "is_incoming": True}) is True
        assert _evaluate_condition(cond, {"attack_type": "melee", "is_incoming": True}) is False

        # Test outgoing attacks
        cond_out = AttackContextCondition(attack_types=["ranged"], applies_to="outgoing")
        assert _evaluate_condition(cond_out, {"attack_type": "ranged", "is_outgoing": True}) is True
        assert _evaluate_condition(cond_out, {"attack_type": "ranged", "is_outgoing": False}) is False

    def test_size_condition_larger(self):
        """SizeCondition with gt comparator."""
        cond = SizeCondition(subject="self", comparator="gt", size="size_1")
        assert _evaluate_condition(cond, {"actor_size": 2}) is True
        assert _evaluate_condition(cond, {"actor_size": 1}) is False

    def test_condition_group_all_of(self):
        """ConditionGroup with all_of requires all conditions."""
        cond = ConditionGroup(all_of=["engaged", "while_flying"])
        # Both must be true
        assert _evaluate_condition(cond, {"is_engaged": True, "is_flying": True}) is True
        assert _evaluate_condition(cond, {"is_engaged": True, "is_flying": False}) is False

    def test_condition_group_any_of(self):
        """ConditionGroup with any_of requires at least one condition."""
        cond = ConditionGroup(any_of=["engaged", "while_flying"])
        # At least one must be true
        assert _evaluate_condition(cond, {"is_engaged": True, "is_flying": False}) is True
        assert _evaluate_condition(cond, {"is_engaged": False, "is_flying": True}) is True
        assert _evaluate_condition(cond, {"is_engaged": False, "is_flying": False}) is False

    def test_condition_group_none_of(self):
        """ConditionGroup with none_of requires no conditions to be true."""
        cond = ConditionGroup(none_of=["engaged"])
        assert _evaluate_condition(cond, {"is_engaged": False}) is True
        assert _evaluate_condition(cond, {"is_engaged": True}) is False


# =============================================================================
# Test _get_talent_accuracy_modifiers
# =============================================================================


class TestGetTalentAccuracyModifiers:
    """Tests for talent accuracy modifier collection."""

    def test_no_talent_effects_returns_zero(self):
        """Combatant with no talent effects gets no modifiers."""
        combatant = make_combatant()
        acc_mod, diff_mod = _get_talent_accuracy_modifiers(combatant, is_ranged=True)
        assert acc_mod == 0
        assert diff_mod == 0

    def test_unconditional_accuracy_modifier(self):
        """Talent with unconditional accuracy bonus is applied."""
        effect = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=1, applies_to="all")]
        )
        combatant = make_combatant(talent_effects=[effect])
        acc_mod, diff_mod = _get_talent_accuracy_modifiers(combatant, is_ranged=True)
        assert acc_mod == 1
        assert diff_mod == 0

    def test_ranged_only_accuracy_modifier(self):
        """Talent with ranged-only accuracy bonus is applied correctly."""
        effect = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=2, applies_to="ranged")]
        )
        combatant = make_combatant(talent_effects=[effect])

        # Should apply for ranged
        acc_mod, diff_mod = _get_talent_accuracy_modifiers(combatant, is_ranged=True)
        assert acc_mod == 2

        # Should NOT apply for melee
        acc_mod, diff_mod = _get_talent_accuracy_modifiers(combatant, is_melee=True)
        assert acc_mod == 0

    def test_melee_only_accuracy_modifier(self):
        """Talent with melee-only accuracy bonus is applied correctly."""
        effect = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=3, applies_to="melee")]
        )
        combatant = make_combatant(talent_effects=[effect])

        # Should apply for melee
        acc_mod, diff_mod = _get_talent_accuracy_modifiers(combatant, is_melee=True)
        assert acc_mod == 3

        # Should NOT apply for ranged
        acc_mod, diff_mod = _get_talent_accuracy_modifiers(combatant, is_ranged=True)
        assert acc_mod == 0

    def test_negative_modifier_becomes_difficulty(self):
        """Negative accuracy values become difficulty modifiers."""
        effect = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=-2, applies_to="all")]
        )
        combatant = make_combatant(talent_effects=[effect])
        acc_mod, diff_mod = _get_talent_accuracy_modifiers(combatant, is_ranged=True)
        assert acc_mod == 0
        assert diff_mod == 2

    def test_conditional_modifier_with_matching_context(self):
        """Conditional modifier applies when condition is met."""
        effect = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=1, applies_to="ranged", condition="engaged")]
        )
        combatant = make_combatant(talent_effects=[effect])

        # With condition met
        acc_mod, _ = _get_talent_accuracy_modifiers(
            combatant, is_ranged=True, context={"is_engaged": True}
        )
        assert acc_mod == 1

        # Without condition met
        acc_mod, _ = _get_talent_accuracy_modifiers(
            combatant, is_ranged=True, context={"is_engaged": False}
        )
        assert acc_mod == 0

    def test_multiple_modifiers_stack(self):
        """Multiple talent effects stack their modifiers."""
        effect1 = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=1, applies_to="all")]
        )
        effect2 = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=2, applies_to="ranged")]
        )
        combatant = make_combatant(talent_effects=[effect1, effect2])
        acc_mod, _ = _get_talent_accuracy_modifiers(combatant, is_ranged=True)
        assert acc_mod == 3

    def test_frame_trait_effects_included(self):
        """Frame trait effects are also included."""
        talent_effect = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=1, applies_to="all")]
        )
        frame_effect = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=2, applies_to="all")]
        )
        combatant = make_combatant(
            talent_effects=[talent_effect],
            frame_trait_effects=[frame_effect],
        )
        acc_mod, _ = _get_talent_accuracy_modifiers(combatant, is_ranged=True)
        # 1 from talent + 2 from frame
        assert acc_mod == 3


# =============================================================================
# Test _get_talent_check_modifiers
# =============================================================================


class TestGetTalentCheckModifiers:
    """Tests for talent check/save modifier collection."""

    def test_no_talent_effects_returns_zero(self):
        """Combatant with no talent effects gets no check modifiers."""
        combatant = make_combatant()
        acc_mod, diff_mod = _get_talent_check_modifiers(combatant, check_type="hull")
        assert acc_mod == 0
        assert diff_mod == 0

    def test_check_modifier_applies_to_correct_type(self):
        """Check modifier only applies to specified check types."""
        effect = MechanicalEffect(
            check_mods=[CheckModifierEffect(value=1, check_types=["systems"])]
        )
        combatant = make_combatant(talent_effects=[effect])

        # Should apply to systems check
        acc_mod, _ = _get_talent_check_modifiers(combatant, check_type="systems")
        assert acc_mod == 1

        # Should NOT apply to hull check
        acc_mod, _ = _get_talent_check_modifiers(combatant, check_type="hull")
        assert acc_mod == 0

    def test_check_modifier_applies_to_correct_kind(self):
        """Check modifier only applies to specified check kinds."""
        effect = MechanicalEffect(
            check_mods=[CheckModifierEffect(value=1, check_types=["hull"], check_kinds=["save"])]
        )
        combatant = make_combatant(talent_effects=[effect])

        # Should apply to hull save
        acc_mod, _ = _get_talent_check_modifiers(combatant, check_type="hull", check_kind="save")
        assert acc_mod == 1

        # Should NOT apply to hull check (non-save)
        acc_mod, _ = _get_talent_check_modifiers(combatant, check_type="hull", check_kind="check")
        assert acc_mod == 0

    def test_universal_check_modifier(self):
        """Check modifier without type restriction applies to all."""
        effect = MechanicalEffect(
            check_mods=[CheckModifierEffect(value=1, check_types=[], check_kinds=[])]
        )
        combatant = make_combatant(talent_effects=[effect])

        acc_mod, _ = _get_talent_check_modifiers(combatant, check_type="hull")
        assert acc_mod == 1

        acc_mod, _ = _get_talent_check_modifiers(combatant, check_type="agility")
        assert acc_mod == 1


# =============================================================================
# Test Real Talents from Compendium
# =============================================================================


class TestRealTalentEffects:
    """Tests using actual talent definitions from the compendium."""

    def test_ace_talent_exists(self):
        """ACE talent is defined in compendium."""
        talent_def = get_talent_definition("ace")
        assert talent_def is not None
        assert talent_def.name == "ACE"
        assert len(talent_def.ranks) == 3

    def test_brutal_talent_exists(self):
        """BRUTAL talent is defined in compendium."""
        talent_def = get_talent_definition("brutal")
        assert talent_def is not None
        assert talent_def.name == "BRUTAL"

    def test_combined_arms_talent_has_accuracy_mods(self):
        """COMBINED ARMS talent has accuracy modifiers."""
        talent_def = get_talent_definition("combined_arms")
        assert talent_def is not None
        # Rank 2 (CQC Training) has +1 accuracy for ranged attacks while engaged
        rank2 = talent_def.get_rank(2)
        assert len(rank2.effects.accuracy_mods) > 0

    def test_combined_arms_accuracy_applied_correctly(self):
        """COMBINED ARMS CQC Training bonus applies when engaged."""
        talent_def = get_talent_definition("combined_arms")
        assert talent_def is not None

        # Create pilot with COMBINED ARMS rank 2
        pilot = Pilot(
            callsign="TestPilot",
            talents=[Talent(talent_id="combined_arms", rank=2)],
        )
        effects = collect_pilot_talent_effects(pilot)

        combatant = make_combatant(talent_effects=effects)

        # Should get +1 accuracy on ranged attacks while engaged
        acc_mod, _ = _get_talent_accuracy_modifiers(
            combatant, is_ranged=True, context={"is_engaged": True}
        )
        assert acc_mod == 1

        # Should NOT get bonus when not engaged
        acc_mod, _ = _get_talent_accuracy_modifiers(
            combatant, is_ranged=True, context={"is_engaged": False}
        )
        assert acc_mod == 0


# =============================================================================
# Test CombatantState Fields
# =============================================================================


class TestCombatantStateFields:
    """Tests for new CombatantState fields."""

    def test_talent_effects_default_empty(self):
        """talent_effects defaults to empty list."""
        combatant = make_combatant()
        assert combatant.talent_effects == []

    def test_frame_trait_effects_default_empty(self):
        """frame_trait_effects defaults to empty list."""
        combatant = make_combatant()
        assert combatant.frame_trait_effects == []

    def test_core_power_available_default_true(self):
        """core_power_available defaults to True."""
        combatant = make_combatant()
        assert combatant.core_power_available is True

    def test_core_power_active_default_false(self):
        """core_power_active defaults to False."""
        combatant = make_combatant()
        assert combatant.core_power_active is False

    def test_core_power_effects_default_none(self):
        """core_power_effects defaults to None."""
        combatant = make_combatant()
        assert combatant.core_power_effects is None


# =============================================================================
# Test Frame Trait Effects (Phase 33)
# =============================================================================


class TestCollectFrameTraitEffects:
    """Tests for collect_frame_trait_effects() function."""

    def test_no_traits_returns_empty(self):
        """A frame with no traits returns empty effects list."""
        frame = MechFrameDefinition(
            id="test_frame",
            name="Test Frame",
            manufacturer="GMS",
            base_stats=MechFrameBaseStats(size="size_1"),
            traits=[],
        )
        effects = collect_frame_trait_effects(frame)
        assert effects == []

    def test_single_trait_returns_its_effect(self):
        """A frame with a single trait returns that trait's effect."""
        trait_effect = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=1, applies_to="melee")]
        )
        frame = MechFrameDefinition(
            id="test_frame",
            name="Test Frame",
            manufacturer="GMS",
            base_stats=MechFrameBaseStats(size="size_1"),
            traits=[FrameTrait(name="Test Trait", effects=trait_effect)],
        )
        effects = collect_frame_trait_effects(frame)
        assert len(effects) == 1
        assert effects[0] == trait_effect

    def test_multiple_traits_returns_all_effects(self):
        """A frame with multiple traits returns all their effects."""
        trait1_effect = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=1, applies_to="melee")]
        )
        trait2_effect = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=2, applies_to="ranged")]
        )
        frame = MechFrameDefinition(
            id="test_frame",
            name="Test Frame",
            manufacturer="GMS",
            base_stats=MechFrameBaseStats(size="size_1"),
            traits=[
                FrameTrait(name="Trait 1", effects=trait1_effect),
                FrameTrait(name="Trait 2", effects=trait2_effect),
            ],
        )
        effects = collect_frame_trait_effects(frame)
        assert len(effects) == 2


class TestGetCorePowerEffects:
    """Tests for get_core_power_effects() function."""

    def test_no_core_system_returns_none(self):
        """A frame without a core system returns None."""
        frame = MechFrameDefinition(
            id="test_frame",
            name="Test Frame",
            manufacturer="GMS",
            base_stats=MechFrameBaseStats(size="size_1"),
            core_system=None,
        )
        effects = get_core_power_effects(frame)
        assert effects is None

    def test_core_system_returns_effects(self):
        """A frame with a core system returns its effects."""
        core_effect = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=2, applies_to="all")]
        )
        frame = MechFrameDefinition(
            id="test_frame",
            name="Test Frame",
            manufacturer="GMS",
            base_stats=MechFrameBaseStats(size="size_1"),
            core_system=CoreSystemDefinition(
                id="test_core",
                name="Test Core System",
                effects=core_effect,
            ),
        )
        effects = get_core_power_effects(frame)
        assert effects == core_effect


# =============================================================================
# Test Activate Core Power Action (Phase 33)
# =============================================================================


class TestActivateCorePower:
    """Tests for activate_core_power action."""

    def test_core_power_available_in_protocols(self):
        """Core power shows up in available protocols when available."""
        core_effect = MechanicalEffect()
        combatant = make_combatant(
            id="mech_1",
            core_power_available=True,
            core_power_active=False,
            core_power_effects=core_effect,
        )
        scenario = MechCombatScenario(combatants=[combatant])
        economy = ActionEconomyState()  # Default state has full/quick actions available

        result = get_available_actions(scenario, "mech_1", economy)
        protocol_ids = [p.action_id for p in result.protocols]
        assert "activate_core_power" in protocol_ids

    def test_core_power_not_available_when_used(self):
        """Core power not in protocols when already used."""
        core_effect = MechanicalEffect()
        combatant = make_combatant(
            id="mech_1",
            core_power_available=False,  # Already used
            core_power_active=False,
            core_power_effects=core_effect,
        )
        scenario = MechCombatScenario(combatants=[combatant])
        economy = ActionEconomyState()

        result = get_available_actions(scenario, "mech_1", economy)
        protocol_ids = [p.action_id for p in result.protocols]
        assert "activate_core_power" not in protocol_ids

    def test_core_power_not_available_without_effects(self):
        """Core power not in protocols when no core power effects."""
        combatant = make_combatant(
            id="mech_1",
            core_power_available=True,
            core_power_active=False,
            core_power_effects=None,  # No core power
        )
        scenario = MechCombatScenario(combatants=[combatant])
        economy = ActionEconomyState()

        result = get_available_actions(scenario, "mech_1", economy)
        protocol_ids = [p.action_id for p in result.protocols]
        assert "activate_core_power" not in protocol_ids

    def test_activate_core_power_success(self):
        """Activating core power sets state correctly."""
        core_effect = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=1, applies_to="all")]
        )
        combatant = make_combatant(
            id="mech_1",
            core_power_available=True,
            core_power_active=False,
            core_power_effects=core_effect,
        )
        scenario = MechCombatScenario(combatants=[combatant])
        economy = ActionEconomyState()
        turn = CombatTurn(actor_id="mech_1")

        action_input = ActionExecutionInput(
            actor_id="mech_1",
            action_id="activate_core_power",
            action_type="protocol",
        )

        new_scenario, new_turn, new_economy, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success
        updated_combatant = next(c for c in new_scenario.combatants if c.id == "mech_1")
        assert updated_combatant.core_power_available is False
        assert updated_combatant.core_power_active is True

    def test_activate_core_power_already_used(self):
        """Cannot activate core power if already used."""
        core_effect = MechanicalEffect()
        combatant = make_combatant(
            id="mech_1",
            core_power_available=False,  # Already used
            core_power_active=False,
            core_power_effects=core_effect,
        )
        scenario = MechCombatScenario(combatants=[combatant])
        economy = ActionEconomyState()
        turn = CombatTurn(actor_id="mech_1")

        action_input = ActionExecutionInput(
            actor_id="mech_1",
            action_id="activate_core_power",
            action_type="protocol",
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert not result.success
        assert result.error is not None and "already used" in result.error

    def test_activate_core_power_already_active(self):
        """Cannot activate core power if already active."""
        core_effect = MechanicalEffect()
        combatant = make_combatant(
            id="mech_1",
            core_power_available=True,
            core_power_active=True,  # Already active
            core_power_effects=core_effect,
        )
        scenario = MechCombatScenario(combatants=[combatant])
        economy = ActionEconomyState()
        turn = CombatTurn(actor_id="mech_1")

        action_input = ActionExecutionInput(
            actor_id="mech_1",
            action_id="activate_core_power",
            action_type="protocol",
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert not result.success
        assert result.error is not None and "already active" in result.error
