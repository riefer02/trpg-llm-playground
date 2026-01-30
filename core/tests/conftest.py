"""Integration test fixtures for Lancer TTRPG mechanical system.

These fixtures provide reusable, deterministic test data for integration tests.
All fixtures are function-scoped (fresh instance per test) and use real
compendium data for realistic testing.

Fixture Hierarchy:
    integration_campaign (root fixture)
    ├── integration_pilot_ll0
    │   └── skills, triggers, talents, background
    ├── integration_mech_everest
    │   ├── frame, build, skills
    │   └── weapons, systems
    └── integration_active_session
        └── integration_sitrep_template

Usage:
    def test_pilot_can_build_mech(integration_pilot_ll0, integration_mech_everest):
        frame, build, pilot_skills = integration_mech_everest
        assert frame.id == "gms_everest"
        assert build.frame_id == "gms_everest"
"""

import pytest
from uuid import uuid4
from typing import TYPE_CHECKING

from core.character import Character, MechConfiguration
from core.pilot.background import Background
from core.pilot.pilot import Pilot
from core.pilot.skill import SkillSet, PilotTrigger
from core.pilot.talent import Talent
from core.pilot.gear import PilotLoadout
from core.mech.frame import MechFrameDefinition
from core.mech.build import MechBuild, build_mech_from_compendium
from core.mech.compendium import get_frame_definition
from core.shared.campaign.campaign import (
    Campaign,
    Session,
    ActiveSessionMission,
    CharacterMechAssignment,
)
from core.shared.scenario import (
    SitrepTemplate,
    ESCORT_TEMPLATE,
    CONTROL_TEMPLATE,
)
from core.shared.narrative import (
    NarrativeGoal,
    NarrativeGoalTracker,
    NarrativeGoalState,
)

if TYPE_CHECKING:
    pass


def _fixed_id(prefix: str) -> str:
    """Generate deterministic ID for tests."""
    return f"test_{prefix}_{uuid4().hex[:8]}"


@pytest.fixture
def integration_pilot_ll0() -> Pilot:
    """Standard LL0 pilot for integration testing.

    Purpose: Valid pilot ready for first mech assignment.
    Usage: Tests requiring pilot without mechs.

    Creates pilot matching book example (PR2 2247-2260):
    - Background: Colonist
    - Skills: HULL 2, others 0
    - Triggers: 4 at +2 each
    - Talents: 3 rank I talents
    """
    return Pilot(
        id="test_pilot_oda_ll0",
        callsign="ODA",
        name="",
        background=Background(
            id="background_colonist",
            name="Colonist",
            triggers=["read_a_situation", "spot", "take_someone_out", "survive"],
        ),
        level=0,
        skills=SkillSet(hull=2, agility=0, systems=0, engineering=0),
        triggers=[
            PilotTrigger(trigger_id="read_a_situation", rank=2),
            PilotTrigger(trigger_id="spot", rank=2),
            PilotTrigger(trigger_id="take_someone_out", rank=2),
            PilotTrigger(trigger_id="survive", rank=2),
        ],
        talents=[
            Talent(talent_id="crack_shot", rank=1),
            Talent(talent_id="infiltrator", rank=1),
            Talent(talent_id="leader", rank=1),
        ],
        pilot_gear=PilotLoadout(
            clothing="flight_suit",
            armor="light_hardsuit",
            weapons=["signature_weapon_combat"],
            gear=["extra_rations", "cooking_gear"],
        ),
    )


@pytest.fixture
def integration_pilot_ll3() -> Pilot:
    """LL3 pilot for advanced testing.

    Purpose: Pilot with licenses, core bonus, higher-level progression.
    Usage: Tests requiring pilot with full LL3 capabilities.
    """
    from core.pilot.talent import Talent
    from core.pilot.license import License
    from core.pilot.core_bonus import CoreBonus

    return Pilot(
        id="test_pilot_oda_ll3",
        callsign="ODA",
        name="",
        background=Background(
            id="background_colonist",
            name="Colonist",
            triggers=["read_a_situation", "spot", "take_someone_out", "survive"],
        ),
        level=3,
        skills=SkillSet(hull=5, agility=0, systems=0, engineering=0),
        triggers=[
            PilotTrigger(trigger_id="read_a_situation", rank=2),
            PilotTrigger(trigger_id="spot", rank=2),
            PilotTrigger(trigger_id="take_someone_out", rank=2),
            PilotTrigger(trigger_id="survive", rank=2),
            PilotTrigger(trigger_id="read_a_situation", rank=2),
            PilotTrigger(trigger_id="spot", rank=2),
        ],
        talents=[
            Talent(talent_id="crack_shot", rank=2),
            Talent(talent_id="infiltrator", rank=2),
            Talent(talent_id="leader", rank=2),
            Talent(talent_id="crack_shot", rank=1),
        ],
        licenses=[
            License(license_id="ipsn_raleigh", rank=3),
        ],
        core_bonuses=[
            CoreBonus(core_bonus_id="ipsn_stasis_shield"),
        ],
        pilot_gear=PilotLoadout(
            clothing="flight_suit",
            armor="light_hardsuit",
            weapons=["signature_weapon_combat"],
            gear=["extra_rations", "cooking_gear"],
        ),
    )


@pytest.fixture
def integration_mech_everest(
    integration_pilot_ll0: Pilot,
) -> tuple[MechFrameDefinition, MechBuild, SkillSet]:
    """Standard GMS Everest mech for integration testing.

    Purpose: Basic mech build matching pilot skills.
    Dependencies: Uses pilot's skill set for stat calculation.
    """
    frame = get_frame_definition("gms_everest")
    if not frame:
        raise ValueError("GMS Everest frame definition not found in compendium")

    build = build_mech_from_compendium(
        frame_id=frame.id,
        weapon_mounts=[
            (0, "anti_material_rifle"),
            (1, "assault_rifle"),
            (2, "tactical_knife"),
            (2, "tactical_knife"),
        ],
        system_ids=[
            "gms_hex_charges",
            "gms_jump_jet_burst",
            "personalizations",
            "gms_custom_paint_job",
        ],
    )

    return frame, build, integration_pilot_ll0.skills


@pytest.fixture
def integration_character_ll0(
    integration_pilot_ll0: Pilot,
    integration_mech_everest: tuple[MechFrameDefinition, MechBuild, SkillSet],
) -> Character:
    """Character with pilot + Everest mech for campaign testing."""
    frame, build, _ = integration_mech_everest
    mech = MechConfiguration(
        id="test_mech_everest",
        name="Everest",
        frame_id=frame.id,
        build=build,
    )
    return Character(
        id="test_character_oda",
        pilot=integration_pilot_ll0,
        mechs=[mech],
        active_mech_id=mech.id,
    )


@pytest.fixture
def integration_mech_raleigh(
    integration_pilot_ll3: Pilot,
) -> tuple[MechFrameDefinition, MechBuild, SkillSet]:
    """IPS-N Raleigh mech for advanced testing.

    Purpose: LL3 mech with manufacturer-specific gear.
    Dependencies: Uses pilot LL3 skills.
    """
    frame = get_frame_definition("ipsn_raleigh")
    if not frame:
        raise ValueError("IPS-N Raleigh frame definition not found in compendium")

    build = build_mech_from_compendium(
        frame_id=frame.id,
        weapon_mounts=[
            (0, "anti_material_rifle"),
            (1, "assault_rifle"),
            (2, "hand_cannon"),
            (2, "hand_cannon"),
        ],
        system_ids=[
            "gms_hex_charges",
            "ipsn_breaching_charges",
            "gms_jump_jet_burst",
            "gms_custom_paint_job",
        ],
    )

    return frame, build, integration_pilot_ll3.skills


@pytest.fixture
def integration_sitrep_template() -> SitrepTemplate:
    """Standard ESCORT-type SITREP template for testing.

    Purpose: Mission template with clear objectives and zones.
    Usage: Tests requiring mission structure.
    """
    return ESCORT_TEMPLATE


@pytest.fixture
def integration_sitrep_control_template() -> SitrepTemplate:
    """CONTROL-type SITREP template for testing.

    Purpose: Zone control mission variant.
    Usage: Tests for zone control mechanics.
    """
    return CONTROL_TEMPLATE


@pytest.fixture
def integration_narrative_tracker() -> NarrativeGoalTracker:
    """Simple narrative goal tracker for combat integration testing.

    Purpose: Track goals across narrative/combat boundary.
    Usage: Tests for narrative-combat bridge.
    """
    goal = NarrativeGoal(
        id="test_goal_destroy_commander",
        description="Destroy the enemy commander",
        success_conditions=[],
    )
    goal_state = NarrativeGoalState(goal=goal, status="active")
    return NarrativeGoalTracker(goals=[goal_state])


@pytest.fixture
def integration_campaign(
    integration_character_ll0: Character,
    integration_mech_everest: tuple[MechFrameDefinition, MechBuild, SkillSet],
) -> Campaign:
    """Campaign with character and mech for persistence testing.

    Purpose: Full campaign state for save/load tests.
    Dependencies: Character and mech fixtures.
    """
    _, build, _ = integration_mech_everest

    character_dict = integration_character_ll0.model_dump(mode="json")

    mech_link = CharacterMechAssignment(
        character_id=integration_character_ll0.id,
        mech_id=integration_character_ll0.active_mech_id or "active_mech",
        mech_name="Everest",
        mech_build=build.model_dump(mode="json"),
        is_active=True,
    )

    return Campaign(
        id="test_campaign_alpha",
        name="Alpha Squad Campaign",
        description="Test campaign for integration testing",
        characters=[character_dict],
        character_mech_links=[mech_link],
        sessions=[
            Session(
                id="test_session_1",
                session_number=1,
                debrief="Initial briefing and first mission",
            )
        ],
    )


@pytest.fixture
def integration_active_session(
    integration_sitrep_template: SitrepTemplate,
    integration_character_ll0: Character,
) -> ActiveSessionMission:
    """Active mission session for testing.

    Purpose: In-progress mission state.
    Usage: Tests for mission progression.
    """
    return ActiveSessionMission(
        mission_state={
            "template_id": integration_sitrep_template.sitrep_type,
            "mission_name": "Test Escort Mission",
            "objectives": [
                {
                    "id": "obj_1",
                    "description": "Escort VIP to extraction point",
                    "status": "in_progress",
                }
            ],
            "zones": [
                {"id": "zone_start", "control": "players"},
                {"id": "zone_mid", "control": "neutral"},
                {"id": "zone_end", "control": "enemies"},
            ],
        },
        participating_character_ids=[integration_character_ll0.id],
    )


@pytest.fixture
def integration_pilot_with_talents() -> Pilot:
    """Pilot with varied talent ranks for progression testing.

    Purpose: Test talent rank advancement.
    Usage: Tests for talent progression mechanics.
    """
    return Pilot(
        id="test_pilot_talents",
        callsign="TALENT",
        name="",
        background=Background(
            id="background_mercenary",
            name="Mercenary",
            triggers=["read_a_situation", "spot", "take_someone_out", "survive"],
        ),
        level=6,
        skills=SkillSet(hull=4, agility=2, systems=0, engineering=0),
        triggers=[
            PilotTrigger(trigger_id="read_a_situation", rank=2),
            PilotTrigger(trigger_id="spot", rank=2),
            PilotTrigger(trigger_id="take_someone_out", rank=2),
            PilotTrigger(trigger_id="survive", rank=2),
        ],
        talents=[
            Talent(talent_id="crack_shot", rank=3),
            Talent(talent_id="infiltrator", rank=2),
            Talent(talent_id="leader", rank=2),
            Talent(talent_id="heavy_gunner", rank=1),
            Talent(talent_id="skirmisher", rank=1),
            Talent(talent_id="technophile", rank=1),
        ],
    )


@pytest.fixture
def integration_mech_minimal() -> tuple[MechFrameDefinition, MechBuild]:
    """Minimal mech build for testing basic functionality.

    Purpose: Simple mech with just essentials.
    Usage: Tests focused on core mech mechanics.
    """
    frame = get_frame_definition("gms_everest")
    if not frame:
        raise ValueError("GMS Everest frame definition not found")

    build = build_mech_from_compendium(
        frame_id=frame.id,
        weapon_mounts=[(0, "assault_rifle")],
        system_ids=[],
    )

    return frame, build
