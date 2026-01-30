"""Tests for narrative primitives (skill challenges, combat constraints, etc.)."""

import pytest
from core.shared import (
    SkillCheck,
    RollModifiers,
    AccuracyDifficulty,
    DifficultyModifier,
    NarrativeCombatConstraints,
    NarrativeGoalOutcome,
    NarrativeGoalCondition,
    NarrativeGoal,
    NarrativeGoalTracker,
    NarrativeScenarioSettings,
    NarrativeComplication,
    NarrativeCombatState,
    NarrativeResolutionRequirement,
    DEFAULT_NARRATIVE_SCENARIO_SETTINGS,
    PrecedenceRule,
    SKILL_CHALLENGE_TYPES,
    SKILL_CHALLENGE_TYPE_BY_ID,
    IndividualCheckResult,
    SkillChallengeResult,
    SkillChallengeDefinition,
    SkillChallengeUse,
    resolve_skill_challenge,
    compute_check_success,
    add_narrative_complication,
    resolve_narrative_complication,
    add_narrative_goal,
    resolve_narrative_goal_check,
    DIFFICULT_TIER_RULE,
    NARRATIVE_TIER_RULES_EXTENDED,
    is_difficult,
    get_narrative_tier_rule_extended,
    Consequence,
    ConsequenceAssignment,
    generate_harm_consequence,
    generate_time_consequence,
    generate_resource_consequence,
    generate_collateral_consequence,
    generate_position_consequence,
    generate_effect_consequence,
    assign_consequence,
    roll_1d3,
    roll_harm,
    roll_severity,
    ExtendedSkillChallengeUse,
    create_extended_challenge,
    resolve_extended_challenge_phase,
    finalize_extended_challenge,
)


class TestDifficultyModifier:
    """Tests for DifficultyModifier model."""

    def test_create_default(self):
        """Default difficulty is +1."""
        mod = DifficultyModifier()
        assert mod.value == 1
        assert mod.reason == ""

    def test_create_with_values(self):
        """Can create with custom values."""
        mod = DifficultyModifier(value=2, reason="extremely difficult")
        assert mod.value == 2
        assert mod.reason == "extremely difficult"

    def test_repr(self):
        """String representation includes key info."""
        mod = DifficultyModifier(value=1, reason="dark room")
        assert "1" in str(mod)
        assert "dark room" in str(mod)


class TestSkillCheckWithDifficulty:
    """Tests for SkillCheck with is_difficult flag."""

    def test_create_standard_check(self):
        """Standard check without difficulty."""
        check = SkillCheck(target=10)
        assert check.target == 10
        assert check.is_difficult is False

    def test_create_difficult_check(self):
        """Check marked as difficult."""
        check = SkillCheck(target=10, is_difficult=True)
        assert check.is_difficult is True

    def test_check_with_modifiers(self):
        """Check with accuracy/difficulty modifiers."""
        check = SkillCheck(
            target=10,
            modifiers=RollModifiers(
                accuracy_difficulty=AccuracyDifficulty(accuracy=1, difficulty=0)
            ),
            is_difficult=True,
        )
        assert check.modifiers.accuracy_difficulty.accuracy == 1
        assert check.is_difficult is True


class TestNarrativeCombatConstraints:
    """Tests for NarrativeCombatConstraints per PR2 rules."""

    def test_default_constraints(self):
        """Default constraints enforce PR2 rules."""
        constraints = NarrativeCombatConstraints()
        assert constraints.harm_only_on_risky_complication is True
        assert constraints.no_attack_rolls is True
        assert constraints.no_npc_turns is True
        assert constraints.no_npc_hp_tracking is True

    def test_custom_constraints(self):
        """Can customize constraints."""
        constraints = NarrativeCombatConstraints(
            harm_only_on_risky_complication=False,
            allow_granted_harm_on_standard=True,
        )
        assert constraints.harm_only_on_risky_complication is False
        assert constraints.allow_granted_harm_on_standard is True

    def test_combat_scenario_constraints(self):
        """Full combat scenario with custom constraints."""
        constraints = NarrativeCombatConstraints(
            no_attack_rolls=True,
            no_npc_turns=True,
            no_npc_hp_tracking=True,
            npc_goals_require_skill_checks=True,
        )
        assert constraints.no_attack_rolls is True
        assert constraints.npc_goals_require_skill_checks is True


class TestNarrativeScenarioSettings:
    """Tests for NarrativeScenarioSettings."""

    def test_default_settings(self):
        """Default settings have expected values."""
        settings = DEFAULT_NARRATIVE_SCENARIO_SETTINGS
        assert settings.environment == "urban"
        assert settings.is_combat is False
        assert settings.combat_constraints is None
        assert settings.time_pressure == "none"
        assert settings.npc_disposition == "neutral"

    def test_combat_scenario(self):
        """Create a combat scenario with constraints."""
        constraints = NarrativeCombatConstraints()
        settings = NarrativeScenarioSettings(
            environment="space_station",
            is_combat=True,
            combat_constraints=constraints,
            time_pressure="urgent",
            npc_disposition="hostile",
            has_allies_present=True,
            has_enemies_present=True,
        )
        assert settings.is_combat is True
        assert settings.combat_constraints is not None
        assert settings.time_pressure == "urgent"
        assert settings.npc_disposition == "hostile"

    def test_environments(self):
        """Test various environment types."""
        for env in [
            "urban",
            "wilderness",
            "space_station",
            "underwater",
            "underground",
        ]:
            settings = NarrativeScenarioSettings(environment=env)
            assert settings.environment == env


class TestPrecedenceRule:
    """Tests for PrecedenceRule model."""

    def test_create_precedence_rule(self):
        """Create a precedence rule."""
        rule = PrecedenceRule(
            rule_id="specific_rule",
            overrides_rule_id="general_rule",
            precedence_level=2,
            context_description="When in space environments",
        )
        assert rule.rule_id == "specific_rule"
        assert rule.overrides_rule_id == "general_rule"
        assert rule.precedence_level == 2


class TestSkillChallengeTypes:
    """Tests for skill challenge type definitions."""

    def test_challenge_types_exist(self):
        """Challenge types are defined."""
        assert len(SKILL_CHALLENGE_TYPES) >= 5

    def test_combat_type_exists(self):
        """Combat type is available."""
        combat = SKILL_CHALLENGE_TYPE_BY_ID.get("combat")
        assert combat is not None
        assert combat.name == "Combat"

    def test_social_type_exists(self):
        """Social type is available."""
        social = SKILL_CHALLENGE_TYPE_BY_ID.get("social")
        assert social is not None

    def test_all_types_have_required_fields(self):
        """All types have valid IDs and names."""
        for challenge_type in SKILL_CHALLENGE_TYPES:
            assert challenge_type.id
            assert challenge_type.name
            assert challenge_type.id == challenge_type.id.lower().replace(" ", "_")


class TestSkillChallengeDefinition:
    """Tests for SkillChallengeDefinition."""

    def test_create_combat_challenge(self):
        """Create a combat skill challenge definition."""
        challenge = SkillChallengeDefinition(
            id="stealth_infiltration",
            name="Stealth Infiltration",
            challenge_type="infiltration",
            description="Sneak past the guards undetected",
            default_tier="standard",
            participant_count_min=1,
            participant_count_max=4,
            allows_help=True,
            allows_push=True,
        )
        assert challenge.id == "stealth_infiltration"
        assert challenge.challenge_type == "infiltration"
        assert challenge.participant_count_min == 1

    def test_default_values(self):
        """Default values are set correctly."""
        challenge = SkillChallengeDefinition(
            id="test_challenge",
            name="Test",
            challenge_type="combat",
            description="Test description",
        )
        assert challenge.target_difficulty == 10
        assert challenge.default_tier == "standard"
        assert challenge.allows_help is True
        assert challenge.allows_push is True

    def test_time_constrained_challenge(self):
        """Challenge with time constraint."""
        challenge = SkillChallengeDefinition(
            id="timed_escape",
            name="Timed Escape",
            challenge_type="chase",
            description="Escape before the doors close",
            time_constraint_turns=3,
        )
        assert challenge.time_constraint_turns == 3


class TestIndividualCheckResult:
    """Tests for IndividualCheckResult."""

    def test_create_check_result(self):
        """Create a successful check result."""
        result = IndividualCheckResult(
            participant_id="pilot_1",
            trigger_used="move_unseen",
            skill_context="agility",
            roll_result=15,
            modifiers_applied="+2 from cover",
            difficulty_modifier=0,
            total_result=15,
            is_success=True,
        )
        assert result.participant_id == "pilot_1"
        assert result.is_success is True
        assert result.total_result == 15

    def test_failed_check_with_consequence(self):
        """Failed check with consequence on risky roll."""
        result = IndividualCheckResult(
            participant_id="pilot_2",
            trigger_used="charm",
            skill_context="hull",
            roll_result=8,
            modifiers_applied="",
            difficulty_modifier=1,
            total_result=7,
            is_success=False,
            consequence_suffered=True,
            consequence_description="Guard becomes suspicious",
        )
        assert result.is_success is False
        assert result.consequence_suffered is True

    def test_skill_types(self):
        """All skill types can be used."""
        for skill in ["hull", "agility", "systems", "engineering"]:
            result = IndividualCheckResult(
                participant_id="test",
                trigger_used="test",
                skill_context=skill,  # type: ignore
                roll_result=10,
                total_result=10,
                is_success=True,
            )
            assert result.skill_context == skill


class TestSkillChallengeResult:
    """Tests for SkillChallengeResult."""

    def test_clear_success(self):
        """Challenge with clear success."""
        result = SkillChallengeResult(
            total_participants=4,
            success_count=3,
            failure_count=1,
            is_success=True,
            required_for_success=3,
        )
        assert result.is_success is True
        assert result.was_tie is False

    def test_clear_failure(self):
        """Challenge with clear failure."""
        result = SkillChallengeResult(
            total_participants=4,
            success_count=1,
            failure_count=3,
            is_success=False,
            required_for_success=3,
        )
        assert result.is_success is False

    def test_tie_result(self):
        """Challenge resulting in tie."""
        result = SkillChallengeResult(
            total_participants=4,
            success_count=2,
            failure_count=2,
            is_success=False,
            required_for_success=3,
            was_tie=True,
            tie_roll_result=0,
        )
        assert result.was_tie is True
        assert result.tie_roll_result == 0

    def test_tie_with_success(self):
        """Tie resolved as success (50% chance)."""
        result = SkillChallengeResult(
            total_participants=4,
            success_count=2,
            failure_count=2,
            is_success=True,
            required_for_success=3,
            was_tie=True,
            tie_roll_result=1,
        )
        assert result.was_tie is True
        assert result.is_success is True


class TestSkillChallengeUse:
    """Tests for SkillChallengeUse and resolution."""

    def test_create_challenge_use(self):
        """Create a skill challenge in progress."""
        definition = SkillChallengeDefinition(
            id="test",
            name="Test",
            challenge_type="combat",
            description="Test",
        )
        use = SkillChallengeUse(
            definition=definition,
            participant_ids=["pilot_1", "pilot_2"],
        )
        assert len(use.participant_ids) == 2
        assert use.resolution is None

    def test_resolve_simple_challenge(self):
        """Resolve a simple challenge with clear results."""
        definition = SkillChallengeDefinition(
            id="test_challenge",
            name="Test Challenge",
            challenge_type="combat",
            description="Test description",
        )
        use = SkillChallengeUse(
            definition=definition,
            participant_ids=["pilot_1", "pilot_2", "pilot_3"],
            individual_checks=[
                IndividualCheckResult(
                    participant_id="pilot_1",
                    trigger_used="attack",
                    skill_context="hull",
                    roll_result=15,
                    total_result=15,
                    is_success=True,
                ),
                IndividualCheckResult(
                    participant_id="pilot_2",
                    trigger_used="hide",
                    skill_context="agility",
                    roll_result=8,
                    total_result=8,
                    is_success=False,
                ),
                IndividualCheckResult(
                    participant_id="pilot_3",
                    trigger_used="hack",
                    skill_context="systems",
                    roll_result=14,
                    total_result=14,
                    is_success=True,
                ),
            ],
        )

        result = resolve_skill_challenge(use)
        assert result.success_count == 2
        assert result.failure_count == 1
        assert result.is_success is True
        assert result.required_for_success == 2

    def test_resolve_tie_challenge(self):
        """Resolve a challenge that results in a tie."""
        definition = SkillChallengeDefinition(
            id="test_challenge",
            name="Test",
            challenge_type="social",
            description="Test",
        )
        use = SkillChallengeUse(
            definition=definition,
            participant_ids=["pilot_1", "pilot_2"],
            individual_checks=[
                IndividualCheckResult(
                    participant_id="pilot_1",
                    trigger_used="charm",
                    skill_context="hull",
                    roll_result=12,
                    total_result=12,
                    is_success=True,
                ),
                IndividualCheckResult(
                    participant_id="pilot_2",
                    trigger_used="read_situation",
                    skill_context="systems",
                    roll_result=9,
                    total_result=9,
                    is_success=False,
                ),
            ],
        )

        result = resolve_skill_challenge(use)
        assert result.was_tie is True
        assert result.tie_roll_result in [0, 1]

    def test_resolve_no_checks(self):
        """Challenge with no individual checks fails."""
        definition = SkillChallengeDefinition(
            id="test",
            name="Test",
            challenge_type="combat",
            description="Test",
        )
        use = SkillChallengeUse(
            definition=definition,
            participant_ids=["pilot_1"],
            individual_checks=[],
        )

        result = resolve_skill_challenge(use)
        assert result.is_success is False
        assert "No checks were made" in result.overall_consequences[0]


class TestComputeCheckSuccess:
    """Tests for compute_check_success function."""

    def test_standard_success(self):
        """Standard check that succeeds."""
        success, consequence, total = compute_check_success(
            roll_result=12,
            target=10,
            modifiers=0,
            tier="standard",
        )
        assert success is True
        assert consequence is False
        assert total == 12

    def test_standard_failure(self):
        """Standard check that fails."""
        success, consequence, total = compute_check_success(
            roll_result=8,
            target=10,
            modifiers=0,
            tier="standard",
        )
        assert success is False
        assert consequence is False
        assert total == 8

    def test_risky_check_success_no_consequence(self):
        """Risky check that succeeds without consequence (20+)."""
        success, consequence, total = compute_check_success(
            roll_result=20,
            target=10,
            modifiers=0,
            tier="risky",
        )
        assert success is True
        assert consequence is False
        assert total == 20

    def test_risky_check_success_with_consequence(self):
        """Risky check that succeeds but with consequence (10-19)."""
        success, consequence, total = compute_check_success(
            roll_result=15,
            target=10,
            modifiers=0,
            tier="risky",
        )
        assert success is True
        assert consequence is True
        assert total == 15

    def test_risky_check_failure_with_consequence(self):
        """Risky check that fails with consequence."""
        success, consequence, total = compute_check_success(
            roll_result=8,
            target=10,
            modifiers=0,
            tier="risky",
        )
        assert success is False
        assert consequence is True
        assert total == 8

    def test_heroic_success(self):
        """Heroic check that succeeds (only on 20+)."""
        success, consequence, total = compute_check_success(
            roll_result=20,
            target=10,
            modifiers=0,
            tier="heroic",
        )
        assert success is True
        assert consequence is False
        assert total == 20

    def test_heroic_failure(self):
        """Heroic check that fails (below 20)."""
        success, consequence, total = compute_check_success(
            roll_result=19,
            target=10,
            modifiers=0,
            tier="heroic",
        )
        assert success is False
        assert consequence is True
        assert total == 19

    def test_difficult_check(self):
        """Check with difficulty modifier."""
        success, consequence, total = compute_check_success(
            roll_result=12,
            target=10,
            modifiers=0,
            difficulty_modifier=1,
            tier="standard",
        )
        assert success is True
        assert total == 11

    def test_difficult_failure(self):
        """Check that fails due to difficulty modifier."""
        success, consequence, total = compute_check_success(
            roll_result=10,
            target=10,
            modifiers=0,
            difficulty_modifier=1,
            tier="standard",
        )
        assert success is False
        assert total == 9

    def test_with_accuracy_modifiers(self):
        """Check with accuracy modifiers."""
        success, consequence, total = compute_check_success(
            roll_result=9,
            target=10,
            modifiers=2,
            tier="standard",
        )
        assert success is True
        assert total == 11

    def test_with_accuracy_and_difficulty(self):
        """Check with both accuracy and difficulty."""
        success, consequence, total = compute_check_success(
            roll_result=10,
            target=10,
            modifiers=2,
            difficulty_modifier=1,
            tier="standard",
        )
        assert success is True
        assert total == 11


class TestNarrativeGoalOutcome:
    """Tests for NarrativeGoalOutcome."""

    def test_successful_goal_no_harm(self):
        """Goal accomplished without harm."""
        outcome = NarrativeGoalOutcome(
            goal_description="Knock out the guard",
            success=True,
            tier_attained="standard",
            harm_involved=True,
            harm_suffered=False,
        )
        assert outcome.success is True
        assert outcome.harm_suffered is False

    def test_successful_goal_with_complication(self):
        """Goal accomplished but pilot suffers harm on risky roll."""
        outcome = NarrativeGoalOutcome(
            goal_description="Disarm the bomb",
            success=True,
            tier_attained="risky",
            harm_involved=True,
            harm_suffered=True,
            complication_description="Slight cut from wire",
        )
        assert outcome.success is True
        assert outcome.harm_suffered is True
        assert outcome.complication_description is not None

    def test_failed_goal(self):
        """Goal not accomplished."""
        outcome = NarrativeGoalOutcome(
            goal_description="Sneak past cameras",
            success=False,
            tier_attained="standard",
            harm_involved=False,
        )
        assert outcome.success is False


class TestNarrativeGoalTracker:
    """Tests for narrative goal tracking helpers."""

    def test_goal_defaults_successes_required(self):
        """Goal defaults successes_required to condition count."""
        goal = NarrativeGoal(
            id="goal_break_in",
            description="Break into the vault",
            success_conditions=[
                NarrativeGoalCondition(
                    id="goal_check",
                    condition_type="skill_check",
                    description="Pass the lockpick check",
                    required_skill="systems",
                ),
                NarrativeGoalCondition(
                    id="goal_escape",
                    condition_type="position_reached",
                    description="Reach the inner door",
                ),
            ],
        )
        assert goal.successes_required == 2
        assert goal.repeat_requires_change is True

    def test_goal_tracker_success(self):
        """Successful check completes the goal."""
        goal = NarrativeGoal(
            id="goal_disarm",
            description="Disarm the bomb",
            harm_involved=False,
        )
        tracker = add_narrative_goal(NarrativeGoalTracker(), goal)
        tracker, outcome = resolve_narrative_goal_check(
            tracker,
            goal_id="goal_disarm",
            roll_result=15,
            tier="standard",
            action_description="Cut the correct wire",
        )
        assert outcome.success is True
        assert tracker.goals[0].status == "completed"
        assert tracker.goals[0].attempts == 1

    def test_goal_tracker_repeat_requires_change(self):
        """Failed checks require circumstances to change."""
        goal = NarrativeGoal(
            id="goal_escape",
            description="Escape the patrol",
        )
        tracker = add_narrative_goal(NarrativeGoalTracker(), goal)
        tracker, outcome = resolve_narrative_goal_check(
            tracker,
            goal_id="goal_escape",
            roll_result=4,
            tier="standard",
        )
        assert outcome.success is False
        with pytest.raises(ValueError):
            resolve_narrative_goal_check(
                tracker,
                goal_id="goal_escape",
                roll_result=12,
                tier="standard",
                circumstances_changed=False,
            )

    def test_goal_tracker_failure_limit(self):
        """Failure limit marks the goal as failed."""
        goal = NarrativeGoal(
            id="goal_hold",
            description="Hold the line",
            failure_limit=1,
        )
        tracker = add_narrative_goal(NarrativeGoalTracker(), goal)
        tracker, outcome = resolve_narrative_goal_check(
            tracker,
            goal_id="goal_hold",
            roll_result=2,
            tier="standard",
        )
        assert outcome.success is False
        assert tracker.goals[0].status == "failed"


class TestNarrativeCombatState:
    """Tests for narrative complication state tracking."""

    def test_complication_defaults_and_requirements(self):
        """Complications track requirements and default flags."""
        complication = NarrativeComplication(
            id="complication_1",
            complication_type="harm",
            description="Stray shot grazes the pilot",
            harm_damage=2,
            resolution_requirements=[
                NarrativeResolutionRequirement(
                    requirement_type="skill_check",
                    description="Find cover and stabilize",
                    required_tier="risky",
                    required_skill="hull",
                )
            ],
        )
        assert complication.severity == "minor"
        assert complication.established_before_roll is True
        assert complication.harm_damage == 2
        assert complication.resolution_requirements[0].required_skill == "hull"

    def test_harm_damage_requires_harm_type(self):
        """Non-harm complications cannot set harm damage."""
        with pytest.raises(ValueError):
            NarrativeComplication(
                id="complication_2",
                complication_type="time",
                description="Delayed extraction",
                harm_damage=1,
            )

    def test_add_and_resolve_complication(self):
        """Complications can be added and resolved by ID."""
        complication = NarrativeComplication(
            id="complication_3",
            complication_type="position",
            description="Pinned behind the barricade",
        )
        state = NarrativeCombatState(scene_id="scene_1")
        state = add_narrative_complication(state, complication)
        assert len(state.complications) == 1
        assert state.complications[0].status == "active"

        state = resolve_narrative_complication(
            state,
            complication_id="complication_3",
            resolution_notes="Flanked the opponent",
            resolved_by="pilot_1",
        )
        assert state.complications[0].status == "resolved"
        assert state.complications[0].resolved_by == "pilot_1"

    def test_resolve_unknown_complication_raises(self):
        """Resolving a missing complication raises."""
        state = NarrativeCombatState()
        with pytest.raises(ValueError):
            resolve_narrative_complication(state, complication_id="missing")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestNarrativeCheckTierExtended:
    """Tests for extended narrative check tiers (difficult)."""

    def test_difficult_tier_exists(self):
        """Difficult tier is available in extended tiers."""
        difficult_rule = get_narrative_tier_rule_extended("difficult")
        assert difficult_rule is not None
        assert difficult_rule.tier == "difficult"

    def test_difficult_tier_properties(self):
        """Difficult tier has correct properties."""
        assert DIFFICULT_TIER_RULE.tier == "difficult"
        assert DIFFICULT_TIER_RULE.success_threshold == 10
        assert DIFFICULT_TIER_RULE.consequence_threshold is None
        assert DIFFICULT_TIER_RULE.allows_push is True

    def test_all_tiers_in_extended_rules(self):
        """All tiers including difficult are in extended rules."""
        tier_names = {rule.tier for rule in NARRATIVE_TIER_RULES_EXTENDED}
        assert tier_names == {"standard", "difficult", "risky", "heroic"}

    def test_is_difficult_helper(self):
        """is_difficult helper works correctly."""
        assert is_difficult("difficult") is True
        assert is_difficult("standard") is False
        assert is_difficult("risky") is False
        assert is_difficult("heroic") is False

    def test_get_tier_rule_extended_lookup(self):
        """Extended tier rule lookup works for all tiers."""
        for tier in ["standard", "difficult", "risky", "heroic"]:
            rule = get_narrative_tier_rule_extended(tier)  # type: ignore
            assert rule is not None
            assert rule.tier == tier


class TestConsequence:
    """Tests for consequence system."""

    def test_generate_minor_harm_consequence(self):
        """Minor harm consequence generates correctly."""
        consequence = generate_harm_consequence(
            severity="minor",
            context="while sneaking past guards",
        )
        assert consequence.consequence_type == "harm"
        assert consequence.severity == "minor"
        assert consequence.harm_damage in [1, 2]
        assert "minor" in consequence.description.lower()

    def test_generate_major_harm_consequence(self):
        """Major harm consequence generates correctly."""
        consequence = generate_harm_consequence(
            severity="major",
            context="during combat",
        )
        assert consequence.consequence_type == "harm"
        assert consequence.severity == "major"
        assert consequence.harm_damage in [3, 4]

    def test_generate_lethal_harm_consequence(self):
        """Lethal harm consequence generates correctly."""
        consequence = generate_harm_consequence(
            severity="lethal",
            context="in the explosion",
        )
        assert consequence.consequence_type == "harm"
        assert consequence.severity == "lethal"
        assert consequence.harm_damage in [5, 6]

    def test_generate_time_consequence(self):
        """Time consequence generates correctly."""
        consequence = generate_time_consequence(
            time_cost=3,
            time_unit="hours",
            context="hacking the terminal",
        )
        assert consequence.consequence_type == "time"
        assert consequence.time_cost == 3
        assert "3" in consequence.description
        assert "hours" in consequence.description

    def test_generate_resource_consequence(self):
        """Resource consequence generates correctly."""
        consequence = generate_resource_consequence(
            resource_type="ammo",
            amount=2,
            context="during firefight",
        )
        assert consequence.consequence_type == "resources"
        assert consequence.resource_type == "ammo"
        assert consequence.resource_amount == 2

    def test_generate_collateral_consequence(self):
        """Collateral consequence generates correctly."""
        consequence = generate_collateral_consequence(
            affected_target="civilian bystander",
            harm_description="is caught in the crossfire",
        )
        assert consequence.consequence_type == "collateral"
        assert consequence.affected_target == "civilian bystander"

    def test_generate_position_consequence(self):
        """Position consequence generates correctly."""
        consequence = generate_position_consequence(
            position_change="pinned down behind cover",
            context="during the firefight",
        )
        assert consequence.consequence_type == "position"
        assert "pinned down" in consequence.description

    def test_generate_effect_consequence(self):
        """Effect consequence generates correctly."""
        consequence = generate_effect_consequence(
            effect_reduction="door only opens partially",
            context="trying to breach",
        )
        assert consequence.consequence_type == "effect"
        assert "reduced" in consequence.description.lower()

    def test_consequence_id_generation(self):
        """Consequences generate unique IDs when context differs."""
        c1 = generate_harm_consequence("minor", "context1", consequence_id="conseq_1")
        c2 = generate_harm_consequence("minor", "context2", consequence_id="conseq_2")
        assert c1.id == "conseq_1"
        assert c2.id == "conseq_2"

    def test_harm_consequence_requires_harm_type(self):
        """Non-harm consequences cannot have harm_damage."""
        with pytest.raises(ValueError):
            Consequence(
                id="test",
                consequence_type="time",
                description="test",
                harm_damage=2,
            )

    def test_severity_only_for_harm(self):
        """Severity is only valid for harm consequences."""
        with pytest.raises(ValueError):
            Consequence(
                id="test",
                consequence_type="time",
                description="test",
                severity="minor",
            )


class TestConsequenceAssignment:
    """Tests for consequence assignment."""

    def test_assign_harm_consequence(self):
        """Can assign harm consequence to check result."""
        result = IndividualCheckResult(
            participant_id="pilot_1",
            trigger_used="attack",
            skill_context="hull",
            roll_result=8,
            total_result=7,
            is_success=False,
        )
        assignment = assign_consequence(
            result,
            consequence_type="harm",
            severity="minor",
        )
        assert assignment.consequence is not None
        assert assignment.consequence.consequence_type == "harm"
        assert assignment.consequence.severity == "minor"

    def test_assign_time_consequence(self):
        """Can assign time consequence to check result."""
        result = IndividualCheckResult(
            participant_id="pilot_1",
            trigger_used="hack",
            skill_context="systems",
            roll_result=9,
            total_result=9,
            is_success=False,
        )
        assignment = assign_consequence(
            result,
            consequence_type="time",
        )
        assert assignment.consequence is not None
        assert assignment.consequence.consequence_type == "time"

    def test_assign_position_consequence(self):
        """Can assign position consequence."""
        result = IndividualCheckResult(
            participant_id="pilot_1",
            trigger_used="charm",
            skill_context="hull",
            roll_result=15,
            total_result=15,
            is_success=True,
        )
        assignment = assign_consequence(
            result,
            consequence_type="position",
            description="pinned down behind cover",
        )
        assert assignment.consequence is not None
        assert assignment.consequence.consequence_type == "position"

    def test_assignment_tracks_roll_total(self):
        """Assignment tracks the roll total when applied."""
        result = IndividualCheckResult(
            participant_id="pilot_1",
            trigger_used="spot",
            skill_context="agility",
            roll_result=7,
            total_result=7,
            is_success=False,
        )
        assignment = assign_consequence(
            result,
            consequence_type="harm",
            severity="minor",
        )
        assert assignment.applied_at_roll == 7


class TestRoll1d3:
    """Tests for 1d3 utility function."""

    def test_roll_1d3_returns_1_to_3(self):
        """1d3 always returns 1, 2, or 3."""
        for _ in range(100):
            result = roll_1d3()
            assert 1 <= result <= 3

    def test_roll_harm_with_specific_roll(self):
        """roll_harm converts d6 roll correctly."""
        assert roll_harm(1) == 1
        assert roll_harm(2) == 1
        assert roll_harm(3) == 2
        assert roll_harm(4) == 2
        assert roll_harm(5) == 3
        assert roll_harm(6) == 3

    def test_roll_severity_mapping(self):
        """roll_severity maps d6 to severity correctly."""
        assert roll_severity(1) == "minor"
        assert roll_severity(2) == "minor"
        assert roll_severity(3) == "major"
        assert roll_severity(4) == "major"
        assert roll_severity(5) == "lethal"
        assert roll_severity(6) == "lethal"


class TestExtendedSkillChallenge:
    """Tests for extended skill challenges."""

    def test_create_extended_challenge(self):
        """Can create extended challenge with phases."""
        definition = SkillChallengeDefinition(
            id="test_extended",
            name="Test Extended",
            challenge_type="combat",
            description="Test",
        )
        challenge = create_extended_challenge(
            definition=definition,
            phase_descriptions=[
                "Phase 1: Breach the perimeter",
                "Phase 2: Secure the objective",
                "Phase 3: Extract safely",
            ],
            participant_ids=["pilot_1", "pilot_2", "pilot_3"],
        )
        assert len(challenge.phases) == 3
        assert challenge.phases[0].phase_number == 1
        assert challenge.phases[2].phase_number == 3

    def test_extended_challenge_defaults(self):
        """Extended challenge has correct defaults."""
        definition = SkillChallengeDefinition(
            id="test",
            name="Test",
            challenge_type="social",
            description="Test",
        )
        challenge = create_extended_challenge(
            definition=definition,
            phase_descriptions=["Single phase"],
            participant_ids=["pilot_1"],
        )
        assert challenge.current_phase_index == 0
        assert challenge.phases[0].required_successes == 1

    def test_resolve_extended_phase(self):
        """Can resolve a single phase of extended challenge."""
        definition = SkillChallengeDefinition(
            id="test",
            name="Test",
            challenge_type="combat",
            description="Test",
        )
        challenge = create_extended_challenge(
            definition=definition,
            phase_descriptions=["Phase 1"],
            participant_ids=["pilot_1", "pilot_2"],
        )

        phase_checks = [
            IndividualCheckResult(
                participant_id="pilot_1",
                trigger_used="attack",
                skill_context="hull",
                roll_result=15,
                total_result=15,
                is_success=True,
            ),
            IndividualCheckResult(
                participant_id="pilot_2",
                trigger_used="hide",
                skill_context="agility",
                roll_result=10,
                total_result=10,
                is_success=True,
            ),
        ]

        updated_challenge, phase_result = resolve_extended_challenge_phase(
            challenge, phase_checks
        )
        assert phase_result.is_success is True
        assert phase_result.success_count == 2
        assert phase_result.failure_count == 0

    def test_finalize_extended_challenge_success(self):
        """Can finalize extended challenge with all phases won."""
        definition = SkillChallengeDefinition(
            id="test",
            name="Test",
            challenge_type="combat",
            description="Test",
        )
        challenge = create_extended_challenge(
            definition=definition,
            phase_descriptions=["Phase 1", "Phase 2", "Phase 3"],
            participant_ids=["pilot_1"],
        )

        for i, phase in enumerate(challenge.phases):
            check = IndividualCheckResult(
                participant_id="pilot_1",
                trigger_used="test",
                skill_context="hull",
                roll_result=15,
                total_result=15,
                is_success=True,
            )
            challenge, _ = resolve_extended_challenge_phase(challenge, [check], i)

        finalized, outcome = finalize_extended_challenge(challenge)
        assert outcome.is_success is True
        assert outcome.phases_won == 3
        assert outcome.phases_lost == 0

    def test_finalize_extended_challenge_failure(self):
        """Can finalize extended challenge with failure."""
        definition = SkillChallengeDefinition(
            id="test",
            name="Test",
            challenge_type="combat",
            description="Test",
        )
        challenge = create_extended_challenge(
            definition=definition,
            phase_descriptions=["Phase 1", "Phase 2", "Phase 3"],
            participant_ids=["pilot_1"],
        )

        for i, phase in enumerate(challenge.phases):
            check = IndividualCheckResult(
                participant_id="pilot_1",
                trigger_used="test",
                skill_context="hull",
                roll_result=5,
                total_result=5,
                is_success=False,
            )
            challenge, _ = resolve_extended_challenge_phase(challenge, [check], i)

        finalized, outcome = finalize_extended_challenge(challenge)
        assert outcome.is_success is False
        assert outcome.phases_won == 0
        assert outcome.phases_lost == 3

    def test_finalize_extended_challenge_tie(self):
        """Tie in extended challenge resolved with 50% chance."""
        definition = SkillChallengeDefinition(
            id="test",
            name="Test",
            challenge_type="combat",
            description="Test",
        )
        challenge = create_extended_challenge(
            definition=definition,
            phase_descriptions=["Phase 1", "Phase 2", "Phase 3", "Phase 4"],
            participant_ids=["pilot_1"],
        )

        for i, phase in enumerate(challenge.phases):
            is_success = i % 2 == 0  # Win, Loss, Win, Loss
            check = IndividualCheckResult(
                participant_id="pilot_1",
                trigger_used="test",
                skill_context="hull",
                roll_result=15 if is_success else 5,
                total_result=15 if is_success else 5,
                is_success=is_success,
            )
            challenge, _ = resolve_extended_challenge_phase(challenge, [check], i)

        finalized, outcome = finalize_extended_challenge(challenge)
        assert outcome.was_tie is True
        assert outcome.phases_won == 2
        assert outcome.phases_lost == 2

    def test_finalize_requires_all_phases_resolved(self):
        """Finalize fails if phases are unresolved."""
        definition = SkillChallengeDefinition(
            id="test",
            name="Test",
            challenge_type="combat",
            description="Test",
        )
        challenge = create_extended_challenge(
            definition=definition,
            phase_descriptions=["Phase 1", "Phase 2"],
            participant_ids=["pilot_1"],
        )

        with pytest.raises(ValueError):
            finalize_extended_challenge(challenge)

    def test_extended_challenge_requires_phases(self):
        """Cannot finalize challenge with no phases."""
        definition = SkillChallengeDefinition(
            id="test",
            name="Test",
            challenge_type="combat",
            description="Test",
        )
        challenge = ExtendedSkillChallengeUse(
            definition=definition,
            phases=[],
            participant_ids=["pilot_1"],
        )

        with pytest.raises(ValueError):
            finalize_extended_challenge(challenge)


class TestComputeCheckSuccessExtended:
    """Tests for compute_check_success with extended tiers."""

    def test_difficult_check_with_success(self):
        """Difficult check that succeeds."""
        success, consequence, total = compute_check_success(
            roll_result=14,
            target=10,
            modifiers=0,
            difficulty_modifier=1,
            tier="difficult",
        )
        assert success is True
        assert consequence is False
        assert total == 13

    def test_difficult_check_failure(self):
        """Difficult check that fails due to difficulty."""
        success, consequence, total = compute_check_success(
            roll_result=10,
            target=10,
            modifiers=0,
            difficulty_modifier=1,
            tier="difficult",
        )
        assert success is False
        assert total == 9

    def test_difficult_no_automatic_consequence(self):
        """Difficult checks don't have automatic consequences on success."""
        success, consequence, total = compute_check_success(
            roll_result=10,
            target=10,
            modifiers=0,
            difficulty_modifier=1,
            tier="difficult",
        )
        assert consequence is False


class TestIntegrationExtended:
    """Integration tests for extended features."""

    def test_full_consequence_workflow(self):
        """Test complete consequence generation and assignment."""
        result = IndividualCheckResult(
            participant_id="pilot_1",
            trigger_used="attack",
            skill_context="hull",
            roll_result=7,
            difficulty_modifier=1,
            total_result=6,
            is_success=False,
            consequence_suffered=True,
        )

        harm_consequence = generate_harm_consequence(
            severity="minor",
            context="grazed by return fire",
        )

        assignment = ConsequenceAssignment(
            check_result=result,
            consequence=harm_consequence,
            gm_notes="GM: Standard harm for failed attack under fire",
            applied_at_roll=result.total_result,
        )

        assert assignment.consequence is not None
        assert assignment.consequence.harm_damage in [1, 2]
        assert "minor" in assignment.consequence.description.lower()

    def test_extended_challenge_with_consequences(self):
        """Test extended challenge where failures produce consequences."""
        definition = SkillChallengeDefinition(
            id="infiltration",
            name="Stealth Infiltration",
            challenge_type="infiltration",
            description="Sneak into the facility",
        )
        challenge = create_extended_challenge(
            definition=definition,
            phase_descriptions=[
                "Get past the outer guards",
                "Bypass the security checkpoint",
                "Reach the objective undetected",
            ],
            participant_ids=["pilot_1", "pilot_2"],
        )

        phase1_checks = [
            IndividualCheckResult(
                participant_id="pilot_1",
                trigger_used="move_unseen",
                skill_context="agility",
                roll_result=14,
                total_result=14,
                is_success=True,
            ),
            IndividualCheckResult(
                participant_id="pilot_2",
                trigger_used="charm",
                skill_context="hull",
                roll_result=12,
                total_result=12,
                is_success=True,
            ),
        ]

        updated_challenge, phase1_result = resolve_extended_challenge_phase(
            challenge, phase1_checks, phase_index=0
        )

        assert phase1_result.is_success is True
        assert phase1_result.success_count == 2
        assert len(phase1_result.overall_consequences) == 0

        for i in [1, 2]:
            phase_checks = [
                IndividualCheckResult(
                    participant_id="pilot_1",
                    trigger_used="test",
                    skill_context="hull",
                    roll_result=15,
                    total_result=15,
                    is_success=True,
                ),
            ]
            updated_challenge, _ = resolve_extended_challenge_phase(
                updated_challenge, phase_checks, phase_index=i
            )

        finalized, outcome = finalize_extended_challenge(updated_challenge)
        assert outcome.total_phases == 3
        assert outcome.phases_won == 3
        assert outcome.phases_lost == 0
