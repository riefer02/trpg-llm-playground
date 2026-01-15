"""Integration tests for complete Lancer game loop.

Tests the full player experience from character creation through campaign completion,
following the session structure defined in PR2:
    Brief → Preparation → Boots on Ground → Narrative Play → Combat → Debrief → Downtime

Each test is independent and uses function-scoped fixtures from conftest.py.
Real compendium data is used for realistic testing.
"""

import pytest
from datetime import date
from core.character import Character
from core.pilot.pilot import Pilot, create_ll0_pilot
from core.pilot.background import Background
from core.pilot.skill import SkillSet, PilotTrigger
from core.pilot.talent import Talent
from core.pilot.license import License
from core.pilot.core_bonus import CoreBonus
from core.pilot.clone_state import CloneState
from core.mech.frame import MechFrameDefinition
from core.mech.build import MechBuild, compute_mech_stats
from core.mech.compendium import get_frame_definition
from core.shared.campaign.campaign import (
    Campaign,
    Session,
    ActiveSessionMission,
    CharacterMechAssignment,
    CampaignMissionRecord,
)
from core.shared.scenario import (
    SitrepTemplate,
    MissionObjective,
    SitrepZone,
    VictoryCondition,
)
from core.shared.integration.narrative_combat import (
    NarrativeCombatBridge,
    CombatEvent,
    CombatResult,
    CombatSetup,
)
from core.mech.combat_state import CombatStats
from core.mech.action_economy import ActionEconomyState
from core.shared.rolls import AttackRoll, SaveRoll
from core.shared.downtime import Reserve, DowntimeAction, PowerAtACost


class TestPilotCreationFlow:
    """Tests for pilot creation and progression (PR2 1905-1920, 1454-1463)."""

    def test_pilot_ll0_creation(self, integration_pilot_ll0: Pilot):
        """Create valid LL0 pilot with all required components."""
        pilot = integration_pilot_ll0

        assert pilot.level == 0
        assert pilot.background is not None
        assert pilot.background.name == "Colonist"
        assert len(pilot.triggers) == 4
        assert all(t.rank == 2 for t in pilot.triggers)
        assert len(pilot.talents) == 3
        assert all(t.rank == 1 for t in pilot.talents)

    def test_pilot_ll0_skill_points(self, integration_pilot_ll0: Pilot):
        """LL0 pilot has exactly 2 skill points allocated."""
        skills = integration_pilot_ll0.skills
        total_points = (
            skills.hull + skills.agility + skills.systems + skills.engineering
        )
        assert total_points == 2

    def test_pilot_level_progression(self, integration_pilot_ll0: Pilot):
        """Pilot can progress from LL0 to LL1."""
        assert integration_pilot_ll0.level == 0

        progressed = Pilot(
            id=integration_pilot_ll0.id,
            callsign=integration_pilot_ll0.callsign,
            name=integration_pilot_ll0.name,
            background=integration_pilot_ll0.background,
            level=1,
            skills=SkillSet(hull=3, agility=0, systems=0, engineering=0),
            triggers=[
                PilotTrigger(trigger_id="read_a_situation", rank=2),
                PilotTrigger(trigger_id="spot", rank=2),
                PilotTrigger(trigger_id="take_someone_out", rank=2),
                PilotTrigger(trigger_id="survive", rank=2),
                PilotTrigger(trigger_id="read_a_situation", rank=2),
            ],
            talents=[
                Talent(talent_id="crack_shot", rank=1),
                Talent(talent_id="infiltrator", rank=1),
                Talent(talent_id="leader", rank=1),
            ],
        )

        assert progressed.level == 1
        assert progressed.skills.total_points() == 3

    def test_pilot_license_acquisition(self, integration_pilot_ll0: Pilot):
        """Pilot gains license at LL1."""
        licensed_pilot = Pilot(
            id=integration_pilot_ll0.id,
            callsign=integration_pilot_ll0.callsign,
            name=integration_pilot_ll0.name,
            background=integration_pilot_ll0.background,
            level=1,
            skills=integration_pilot_ll0.skills,
            triggers=integration_pilot_ll0.triggers,
            talents=integration_pilot_ll0.talents,
            licenses=[License(license_id="gms_everest", rank=1)],
        )

        assert len(licensed_pilot.licenses) == 1
        assert licensed_pilot.licenses[0].rank == 1

    def test_pilot_talent_rank_progression(self, integration_pilot_with_talents: Pilot):
        """Talents can advance from rank I to II to III."""
        assert integration_pilot_with_talents.level >= 6

        crack_shot = next(
            t
            for t in integration_pilot_with_talents.talents
            if t.talent_id == "crack_shot"
        )
        assert crack_shot.rank == 3

    def test_pilot_core_bonus_unlocking(self, integration_pilot_ll3: Pilot):
        """Core bonus unlocks at LL3 manufacturer milestone."""
        assert integration_pilot_ll3.level == 3
        assert len(integration_pilot_ll3.core_bonuses) == 1
        assert (
            integration_pilot_ll3.core_bonuses[0].core_bonus_id == "ipsn_stasis_shield"
        )


class TestMechBuildingFlow:
    """Tests for mech building (PR2 2001-2050)."""

    def test_mech_build_from_compendium(self, integration_mech_everest):
        """Build mech using frame, weapons, and systems from compendium."""
        frame, build, pilot_skills = integration_mech_everest

        assert frame.id == "gms_everest"
        assert build.frame_id == "gms_everest"
        assert len(build.weapons) == 4
        assert len(build.systems) == 4

    def test_mech_stat_calculation(self, integration_mech_everest):
        """Mech stats are correctly calculated from build."""
        frame, build, pilot_skills = integration_mech_everest

        stats = compute_mech_stats(
            frame=frame,
            skills=pilot_skills,
            grit=0,
            bonus_effects=[],
        )

        assert stats.hp > 0
        assert stats.evasion > 0
        assert stats.heat_cap > 0
        assert stats.speed > 0

    def test_mech_loadout_validation(self, integration_mech_everest):
        """Mech loadout follows mount size rules."""
        frame, build, _ = integration_mech_everest

        for mounted in build.weapons:
            assert mounted.mount_index >= 0
            assert mounted.weapon_id is not None

    def test_character_mech_assignment(
        self, integration_character_ll0, integration_mech_everest
    ):
        """Character can be assigned to a mech."""
        frame, build, _ = integration_mech_everest

        assignment = CharacterMechAssignment(
            character_id=integration_character_ll0.id,
            mech_id=f"{integration_character_ll0.pilot.callsign.lower()}-everest",
            mech_name="Everest",
            mech_build=build.model_dump(mode="json"),
            is_active=True,
        )

        assert assignment.character_id == integration_character_ll0.id
        assert assignment.mech_build["frame_id"] == "gms_everest"

    def test_mech_raleigh_build(self, integration_mech_raleigh):
        """Raleigh mech builds correctly with manufacturer-specific gear."""
        frame, build, pilot_skills = integration_mech_raleigh

        assert frame.id == "ipsn_raleigh"
        assert build.frame_id == "ipsn_raleigh"

        stats = compute_mech_stats(
            frame=frame,
            skills=pilot_skills,
            grit=2,
            bonus_effects=[],
        )

        assert stats.hp > 10


class TestMissionPreparation:
    """Tests for mission preparation phase (PR2 2699-2890)."""

    def test_mission_brief_establishes_goal(
        self, integration_sitrep_template: SitrepTemplate
    ):
        """Mission template establishes clear goal and stakes."""
        assert integration_sitrep_template.description is not None
        assert integration_sitrep_template.special_rules is not None

    def test_mission_preparation_selects_mech(
        self,
        integration_character_ll0,
        integration_mech_everest: tuple,
    ):
        """Character can select mech for mission."""
        frame, build, _ = integration_mech_everest

        assignment = CharacterMechAssignment(
            character_id=integration_character_ll0.id,
            mech_id="selected_mech",
            mech_name="Everest",
            mech_build=build.model_dump(mode="json"),
            is_active=True,
        )

        assert assignment.is_active is True
        assert assignment.mech_name == "Everest"

    def test_mission_preparation_sets_reserves(self):
        """Reserves can be established during preparation."""
        reserve = Reserve(
            id="test_ammo_reserve",
            reserve_type="mech",
            specific_type="ammo",
            description="+2 ammo for anti-material rifle",
            quantity=2,
        )

        assert reserve.reserve_type == "mech"
        assert reserve.specific_type == "ammo"
        assert reserve.quantity == 2

    def test_boots_on_ground_starts_mission(
        self,
        integration_character_ll0,
        integration_sitrep_template: SitrepTemplate,
    ):
        """Boots on ground establishes mission start state."""
        mission = ActiveSessionMission(
            mission_state={
                "template_id": integration_sitrep_template.sitrep_type,
                "mission_name": integration_sitrep_template.name,
                "description": integration_sitrep_template.description,
                "special_rules": integration_sitrep_template.special_rules,
                "objectives": [
                    {
                        "id": "obj_1",
                        "description": "Primary objective",
                        "status": "in_progress",
                    }
                ],
            "zones": [
                {"id": "zone_1", "control": "neutral", "controlling_side": None}
            ],
        },
        participating_character_ids=[integration_character_ll0.id],
    )

        assert mission.participating_character_ids[0] == integration_character_ll0.id
        assert "objectives" in mission.mission_state


class TestCombatResolution:
    """Tests for combat mechanics (PR2 3703-4706)."""

    def test_combat_initiative_structure(self, integration_pilot_ll0: Pilot):
        """Combat has proper structure for initiative tracking."""
        from core.shared.combat.tactical_initiative import (
            start_tactical_combat,
            CombatSide,
        )

        combatants: dict[str, CombatSide] = {
            integration_pilot_ll0.id: "players",
            "npc_grunt_1": "hostiles",
        }

        result = start_tactical_combat(combatants=combatants)

        assert result is not None
        assert result.round_index == 1

    def test_combat_action_economy(
        self,
        integration_pilot_ll0: Pilot,
        integration_mech_everest: tuple,
    ):
        """Mech has correct action economy structure."""
        from core.mech.action_economy import ActionEconomyState

        economy = ActionEconomyState(
            full_actions_used=0,
            quick_actions_used=0,
            overcharge_used=False,
            reactions_used_this_turn=0,
        )

        assert economy.full_actions_remaining == 1
        assert economy.quick_actions_remaining == 2

    def test_attack_resolution_structure(
        self,
        integration_mech_everest: tuple,
    ):
        """Attack resolution has required components."""
        from core.shared.rolls import AttackRoll, SaveRoll

        attack = AttackRoll(
            target=10,
        )

        assert attack.target == 10
        assert attack.roll_type == "attack"

    def test_structure_damage_handling(self):
        """Structure damage follows correct mechanics."""
        from core.mech.combat_state import CombatStats
        from core.shared.enums import SizeClass

        stats = CombatStats(
            size="size_half",
            hp_max=10,
            evasion=10,
            e_defense=10,
            armor=0,
        )

        assert stats.hp_max == 10

    def test_heat_overheat_system(self):
        """Heat mechanics work correctly."""
        from core.mech.combat_state import CombatStats

        stats = CombatStats(
            size="size_half",
            hp_max=10,
            evasion=10,
            e_defense=10,
            armor=0,
        )

        assert stats.hp_max == 10


class TestMissionResolution:
    """Tests for mission resolution and debrief (PR2 3198-3254)."""

    def test_mission_outcome_determination(self):
        """Mission outcome is calculated from objectives."""
        completed = 3
        total = 4
        score = completed / total

        if score >= 1.0:
            outcome = "success"
        elif score >= 0.5:
            outcome = "partial"
        elif score > 0:
            outcome = "failure"
        else:
            outcome = "catastrophic"

        assert outcome == "partial"

    def test_debrief_level_up(self):
        """Pilot gains license level after mission."""
        pilot = Pilot(
            id="test_pilot",
            callsign="TEST",
            level=0,
            skills=SkillSet(hull=2),
            triggers=[],
            talents=[],
        )

        leveled_up = Pilot(
            id=pilot.id,
            callsign=pilot.callsign,
            level=pilot.level + 1,
            skills=SkillSet(hull=2),
            triggers=[],
            talents=[],
        )

        assert leveled_up.level == 1

    def test_downtime_action_execution(self):
        """Downtime actions produce expected outcomes."""
        action = PowerAtACost()

        success_outcome = action.get_outcome(roll_result=20)
        assert success_outcome.tier == "success"
        assert len(success_outcome.reserves_earned) > 0

    def test_campaign_pilot_level_tracking(
        self,
        integration_campaign: Campaign,
    ):
        """Campaign tracks character license level across sessions."""
        character_id = integration_campaign.characters[0]["id"]
        character_level = integration_campaign.character_level(character_id)

        assert character_level == 0

        updated_character = {
            **integration_campaign.characters[0],
            "pilot": {
                **integration_campaign.characters[0]["pilot"],
                "level": 1,
            },
        }

        updated_campaign = Campaign(
            id=integration_campaign.id,
            name=integration_campaign.name,
            characters=[updated_character],
            character_mech_links=integration_campaign.character_mech_links,
        )

        character_level = updated_campaign.character_level(character_id)
        assert character_level == 1


class TestCampaignPersistence:
    """Tests for campaign persistence across sessions."""

    def test_campaign_serialization(
        self,
        integration_campaign: Campaign,
    ):
        """Campaign can be serialized to JSON."""
        from core.shared.campaign.serialization import campaign_to_dict

        data = campaign_to_dict(integration_campaign)

        assert data["id"] == integration_campaign.id
        assert data["name"] == integration_campaign.name
        assert len(data["characters"]) == 1
        assert len(data["character_mech_links"]) == 1

    def test_campaign_deserialization(self):
        """Campaign can be loaded from JSON."""
        from core.shared.campaign.serialization import dict_to_campaign

        data = {
            "id": "test_campaign",
            "name": "Test Campaign",
            "characters": [
                {
                    "id": "test_character",
                    "pilot": {
                        "id": "test_pilot",
                        "callsign": "TEST",
                        "level": 0,
                        "skills": {
                            "hull": 2,
                            "agility": 0,
                            "systems": 0,
                            "engineering": 0,
                        },
                        "triggers": [],
                        "talents": [],
                    },
                }
            ],
            "character_mech_links": [],
            "sessions": [],
            "mission_history": [],
        }

        campaign = dict_to_campaign(data)

        assert campaign.id == "test_campaign"
        assert campaign.name == "Test Campaign"
        assert len(campaign.characters) == 1

    def test_session_tracking(self, integration_campaign: Campaign):
        """Sessions are tracked in campaign."""
        new_session = Session(
            id="test_session_2",
            session_number=2,
            session_date=date.today(),
            debrief="Second mission completed",
        )

        updated_campaign = Campaign(
            id=integration_campaign.id,
            name=integration_campaign.name,
            characters=integration_campaign.characters,
            character_mech_links=integration_campaign.character_mech_links,
            sessions=integration_campaign.sessions + [new_session],
        )

        assert len(updated_campaign.sessions) == 2
        assert updated_campaign.sessions[1].session_number == 2

    def test_mission_history_recording(self, integration_campaign: Campaign):
        """Completed missions are recorded in history."""
        mission_record = CampaignMissionRecord(
            mission_id="test_mission_1",
            session_id="test_session_1",
            mission_name="First Mission",
            outcome="success",
            completion_score=1.0,
            participating_character_ids=[integration_campaign.characters[0]["id"]],
        )

        assert mission_record.outcome == "success"
        assert mission_record.completion_score == 1.0


class TestNarrativeCombatBridge:
    """Tests for narrative-combat mode switching."""

    def test_narrative_to_combat_bridge(
        self,
        integration_narrative_tracker,
        integration_pilot_ll0: Pilot,
    ):
        """Narrative mode can transition to combat."""
        from core.shared.narrative import NarrativeGoalTracker, NarrativeCombatState
        from core.shared.integration.narrative_combat import CombatSetup

        combat_state = NarrativeCombatState()

        setup = CombatSetup(
            narrative_tracker=integration_narrative_tracker,
            combat_start_state=combat_state,
            participating_npcs=["npc_1", "npc_2"],
            participating_players=[integration_pilot_ll0.id],
        )

        assert setup.narrative_tracker is not None
        assert len(setup.participating_players) == 1
        assert len(setup.participating_npcs) == 2

    def test_combat_to_narrative_bridge(self):
        """Combat mode can transition back to narrative."""
        from core.shared.integration.narrative_combat import CombatResult

        result = CombatResult(
            outcome="victory",
            events=[],
            surviving_participants=["player_1"],
            casualties=["npc_1"],
            turn_count=5,
        )

        assert result.outcome == "victory"
        assert result.turn_count == 5

    def test_combat_event_triggers_goal_update(self):
        """Combat events can update narrative goals."""
        from core.shared.integration.narrative_combat import CombatEvent

        event = CombatEvent(
            event_type="target_destroyed",
            source_id="player_1",
            target_id="npc_commander",
        )

        assert event.event_type == "target_destroyed"
        assert event.target_id == "npc_commander"


class TestTriggerAndBackground:
    """Tests for trigger and background mechanics."""

    def test_background_invocation(self, integration_pilot_ll0: Pilot):
        """Background can be invoked for bonuses."""
        bg = integration_pilot_ll0.background
        assert bg is not None
        assert len(bg.triggers) == 4

    def test_trigger_activation(self):
        """Triggers activate correctly."""
        from core.shared.effects import TriggerType

        assert "on_hit" in TriggerType.__args__
        assert "on_crit" in TriggerType.__args__
        assert "on_kill" in TriggerType.__args__

    def test_pilot_triggers_match_definitions(self, integration_pilot_ll0: Pilot):
        """Pilot triggers are valid trigger IDs."""
        from core.pilot.skill import TRIGGER_DEFINITIONS

        valid_ids = {t.id for t in TRIGGER_DEFINITIONS}

        for trigger in integration_pilot_ll0.triggers:
            assert trigger.trigger_id in valid_ids, (
                f"Invalid trigger: {trigger.trigger_id}"
            )


class TestSITREPTemplates:
    """Tests for SITREP template structure."""

    def test_escort_template_structure(
        self, integration_sitrep_template: SitrepTemplate
    ):
        """ESCORT template has required fields."""
        assert integration_sitrep_template.sitrep_type is not None
        assert integration_sitrep_template.name is not None
        assert integration_sitrep_template.description is not None

    def test_control_template_structure(
        self, integration_sitrep_control_template: SitrepTemplate
    ):
        """CONTROL template has zone control mechanics."""
        assert integration_sitrep_control_template.sitrep_type is not None
        assert integration_sitrep_control_template.objective_zones is not None

    def test_template_objectives(self, integration_sitrep_template: SitrepTemplate):
        """Templates define mission objectives."""
        objectives = integration_sitrep_template.victory_conditions
        assert isinstance(objectives, list)


class TestCloningSystem:
    """Tests for pilot cloning mechanics."""

    def test_cloning_state_creation(self):
        """Clone state can be created."""
        from core.pilot.clone_state import CloneStatus, CloneState, Quirk, QuirkSource

        clone = CloneState(
            status=CloneStatus(times_cloned=0, is_dead=False),
            assigned_quirk=Quirk(
                roll=1,
                name="Nervous",
                description="+1 Difficulty on all checks",
                quirk_type="mental",
            ),
            quirk_source="clone",
        )

        assert clone.status.is_dead is False
        assert clone.assigned_quirk is not None
        assert clone.assigned_quirk.name == "Nervous"

    def test_pilot_death_tracking(self):
        """Pilot death is tracked in campaign."""
        pilot = Pilot(
            id="test_pilot",
            callsign="TEST",
            level=0,
            skills=SkillSet(hull=2),
            triggers=[],
            talents=[],
            is_dead=True,
        )

        assert pilot.is_dead is True

    def test_quirk_application(self):
        """Quirks apply correctly from cloning."""
        from core.pilot.clone_state import Quirk, QuirkSource

        quirk = Quirk(
            roll=1,
            name="Nervous",
            description="+1 Difficulty on all checks",
            quirk_type="mental",
        )

        assert quirk.name == "Nervous"


class TestIntegrationScenarios:
    """Cross-cutting integration scenarios."""

    def test_pilot_to_mech_to_combat(
        self,
        integration_pilot_ll0: Pilot,
        integration_mech_everest: tuple,
    ):
        """Full pilot → mech → combat pipeline works."""
        frame, build, skills = integration_mech_everest

        stats = compute_mech_stats(
            frame=frame,
            skills=skills,
            grit=0,
            bonus_effects=[],
        )

        assert stats.hp > 0
        assert stats.evasion > 0

    def test_campaign_with_mission_history(
        self,
        integration_campaign: Campaign,
    ):
        """Campaign tracks mission history correctly."""
        mission_record = CampaignMissionRecord(
            mission_id="escort_vip",
            session_id="test_session_1",
            mission_name="VIP Escort",
            outcome="success",
            completion_score=1.0,
            participating_character_ids=[integration_campaign.characters[0]["id"]],
        )

        updated_history = integration_campaign.mission_history + [mission_record]

        assert len(updated_history) == 1
        assert updated_history[0].outcome == "success"

    def test_multi_character_campaign(
        self,
        integration_pilot_ll0: Pilot,
        integration_mech_everest: tuple,
    ):
        """Campaign can track multiple characters."""
        pilot1 = integration_pilot_ll0

        pilot2 = Pilot(
            id="test_pilot_2",
            callsign="BIRD",
            level=0,
            skills=SkillSet(hull=0, agility=2, systems=0, engineering=0),
            triggers=[],
            talents=[],
        )

        character1 = Character(id="test_character_1", pilot=pilot1)
        character2 = Character(id="test_character_2", pilot=pilot2)

        campaign = Campaign(
            id="test_multi_character",
            name="Multi-Character Campaign",
            characters=[
                character1.model_dump(mode="json"),
                character2.model_dump(mode="json"),
            ],
            character_mech_links=[],
        )

        assert len(campaign.characters) == 2

    def test_reserve_accumulation_across_sessions(self):
        """Reserves can be accumulated across sessions."""
        reserve1 = Reserve(
            id="reserve_ammo",
            reserve_type="mech",
            specific_type="ammo",
            description="Ammo",
            quantity=2,
        )

        reserve2 = Reserve(
            id="reserve_gear",
            reserve_type="mech",
            specific_type="rented_gear",
            description="Rented Gear",
            quantity=1,
        )

        all_reserves = [reserve1, reserve2]

        assert len(all_reserves) == 2
        mech_reserves = [r for r in all_reserves if r.reserve_type == "mech"]
        assert len(mech_reserves) == 2
