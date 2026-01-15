"""Tests for campaign persistence layer."""

import json
import tempfile
import os
from datetime import date, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from core.pilot.pilot import Pilot, create_ll0_pilot
from core.pilot.background import Background
from core.pilot.skill import SkillSet, PilotTrigger
from core.pilot.talent import Talent
from core.pilot.license import License
from core.mech.build import MechBuild, MountedWeapon, InstalledSystem
from core.shared.scenario import Mission, MissionObjective, MissionState
from core.shared.campaign import (
    Campaign,
    Session,
    PilotMechAssignment,
    CampaignMissionRecord,
    ActiveSessionMission,
    MissionPrepPlan,
    CampaignIdentity,
    CampaignLobbyState,
    MissionOutcomeReport,
)
from core.shared.campaign.serialization import (
    save_campaign,
    load_campaign,
    save_campaign_to_string,
    load_campaign_from_string,
    campaign_to_dict,
    dict_to_campaign,
    validate_campaign,
    validate_campaign_synchronous,
    is_campaign_valid,
    get_campaign_summary,
    CampaignValidationError,
)


@pytest.fixture
def sample_pilot() -> dict:
    """Create a sample pilot dict for testing."""
    pilot = create_ll0_pilot(
        callsign="Viper",
        name="Alexei Viper",
        background=Background(
            id="background_mercenary",
            name="Mercenary",
            triggers=[
                "stay_cool",
                "read_a_situation",
                "invent_or_create",
                "investigate",
            ],
        ),
        skills=SkillSet(hull=1, systems=1),
        triggers=[
            PilotTrigger(trigger_id="on_hit", rank=2),
            PilotTrigger(trigger_id="on_overheat", rank=2),
            PilotTrigger(trigger_id="on_crit", rank=2),
            PilotTrigger(trigger_id="on_kill", rank=2),
        ],
        talents=[
            Talent(talent_id="cracked_hull", rank=1),
            Talent(talent_id="hunter", rank=1),
            Talent(talent_id="reckless", rank=1),
        ],
        id="pilot-viper-001",
    )
    return pilot.model_dump()


@pytest.fixture
def sample_pilot_2() -> dict:
    """Create a second sample pilot dict for testing."""
    pilot = create_ll0_pilot(
        callsign="Forge",
        name="Samira Okonkwo",
        background=Background(
            id="background_mercenary",
            name="Mercenary",
            triggers=[
                "threaten",
                "blow_something_up",
                "take_control",
                "apply_fists_to_faces",
            ],
        ),
        skills=SkillSet(hull=2),
        triggers=[
            PilotTrigger(trigger_id="on_crit", rank=2),
            PilotTrigger(trigger_id="on_reload", rank=2),
            PilotTrigger(trigger_id="on_hit", rank=2),
            PilotTrigger(trigger_id="on_overheat", rank=2),
        ],
        talents=[
            Talent(talent_id="heavy_weaponner", rank=1),
            Talent(talent_id="cracked_hull", rank=1),
            Talent(talent_id="tactical", rank=1),
        ],
        id="pilot-forge-002",
    )
    return pilot.model_dump()


@pytest.fixture
def sample_mech_build() -> dict:
    """Create a sample mech build dict for testing."""
    build = MechBuild(
        frame_id="everest",
        weapons=[
            MountedWeapon(mount_index=0, weapon_id="assault_rifle", weapon_size="aux"),
            MountedWeapon(
                mount_index=1, weapon_id="heavy_machine_gun", weapon_size="main"
            ),
        ],
        systems=[
            InstalledSystem(system_id="patch", sp_cost=1),
            InstalledSystem(system_id="mount", sp_cost=1),
        ],
    )
    return build.model_dump()


@pytest.fixture
def sample_campaign(
    sample_pilot: dict,
    sample_pilot_2: dict,
    sample_mech_build: dict,
) -> Campaign:
    """Create a sample campaign for testing."""
    pilot_mech_links = [
        PilotMechAssignment(
            pilot_id=sample_pilot["id"],
            mech_id=f"{sample_pilot['callsign']}-everest",
            mech_name="Viper's Everest",
            mech_build=sample_mech_build,
            is_active=True,
        ),
        PilotMechAssignment(
            pilot_id=sample_pilot_2["id"],
            mech_id=f"{sample_pilot_2['callsign']}-everest",
            mech_name="Forge's Everest",
            mech_build=sample_mech_build,
            is_active=True,
        ),
    ]

    return Campaign(
        id="campaign-test-001",
        name="Test Campaign",
        description="A test campaign for unit tests",
        pilots=[sample_pilot, sample_pilot_2],
        pilot_mech_links=pilot_mech_links,
    )


class TestPilotMechAssignment:
    """Tests for PilotMechAssignment model."""

    def test_create_assignment(self, sample_pilot: dict, sample_mech_build: dict):
        """Test creating a pilot-mech assignment."""
        link = PilotMechAssignment(
            pilot_id=sample_pilot["id"],
            mech_id="viper-mech-1",
            mech_name="Viper's Mech",
            mech_build=sample_mech_build,
            is_active=True,
        )
        assert link.pilot_id == sample_pilot["id"]
        assert link.mech_id == "viper-mech-1"
        assert link.is_active is True

    def test_inactive_assignment(self, sample_pilot: dict, sample_mech_build: dict):
        """Test creating an inactive pilot-mech assignment."""
        link = PilotMechAssignment(
            pilot_id=sample_pilot["id"],
            mech_id="viper-old-mech",
            mech_name="Viper's Old Mech",
            mech_build=sample_mech_build,
            is_active=False,
        )
        assert link.is_active is False


class TestSession:
    """Tests for Session model."""

    def test_create_session(self):
        """Test creating a session."""
        session = Session(
            id="session-001",
            session_number=1,
            session_date=date(2025, 1, 15),
            debrief="First session went well",
        )
        assert session.id == "session-001"
        assert session.session_number == 1
        assert session.debrief == "First session went well"

    def test_session_tracks_mission_outcome(self):
        """Sessions can store mission outcome summaries for exports."""
        outcome = MissionOutcomeReport(
            outcome="success",
            completion_score=0.9,
            debrief_notes="Pulled civilians out",
            reserves_spent=[{"id": "supply_drop", "notes": "Used on round 3"}],
            reserves_earned=[{"id": "intel_cache"}],
            rewards=["+1 reserve", "New contact"],
        )
        session = Session(
            id="session-002",
            session_number=2,
            mission_outcome=outcome,
        )
        assert session.mission_outcome is not None
        assert session.mission_outcome.outcome == "success"
        assert session.mission_outcome.rewards == ["+1 reserve", "New contact"]
        assert session.active_missions == []
        assert session.reserves_earned == []

    def test_session_has_default_lifecycle(self):
        """Session should auto-populate the lifecycle checkpoints."""
        session = Session(id="session-002", session_number=2)
        phases = [checkpoint.phase for checkpoint in session.lifecycle_checkpoints]
        assert phases == ["downtime", "brief", "prep", "mission", "debrief"]
        assert all(
            checkpoint.status == "pending"
            for checkpoint in session.lifecycle_checkpoints
        )

    def test_session_can_store_mission_plan(self):
        """Mission prep data should be stored alongside the session."""
        mission_plan = MissionPrepPlan(
            mission_name="Operation Dawn",
            objectives=[],
            support_assets=["Orbital overwatch"],
        )
        session = Session(
            id="session-003",
            session_number=3,
            mission_plan=mission_plan,
        )
        assert session.mission_plan is not None
        assert session.mission_plan.mission_name == "Operation Dawn"


class TestCampaign:
    """Tests for Campaign model."""

    def test_create_campaign(self):
        """Test creating a campaign."""
        campaign = Campaign(
            id="campaign-001",
            name="Test Campaign",
            description="A test campaign",
        )
        assert campaign.id == "campaign-001"
        assert campaign.name == "Test Campaign"
        assert campaign.sessions == []
        assert campaign.pilots == []
        assert campaign.pilot_mech_links == []
        assert campaign.mission_history == []

    def test_get_pilot(self, sample_campaign: Campaign, sample_pilot: dict):
        """Test getting a pilot by ID."""
        found = sample_campaign.get_pilot(sample_pilot["id"])
        assert found is not None
        assert found["callsign"] == "Viper"

    def test_get_pilot_not_found(self, sample_campaign: Campaign):
        """Test getting a non-existent pilot."""
        found = sample_campaign.get_pilot("non-existent-id")
        assert found is None

    def test_get_pilot_mech_assignment(
        self, sample_campaign: Campaign, sample_pilot: dict
    ):
        """Test getting mech assignments for a pilot."""
        links = sample_campaign.get_pilot_mech_assignment(sample_pilot["id"])
        assert len(links) == 1
        assert links[0].mech_name == "Viper's Everest"

    def test_get_active_mech_for_pilot(
        self, sample_campaign: Campaign, sample_pilot: dict
    ):
        """Test getting the active mech for a pilot."""
        active = sample_campaign.get_active_mech_for_pilot(sample_pilot["id"])
        assert active is not None
        assert active.mech_name == "Viper's Everest"

    def test_get_session(self, sample_campaign: Campaign):
        """Test getting a session by ID."""
        session = Session(id="session-001", session_number=1)
        sample_campaign.sessions.append(session)

        found = sample_campaign.get_session("session-001")
        assert found is not None
        assert found.session_number == 1

    def test_pilot_level(self, sample_campaign: Campaign, sample_pilot: dict):
        """Test getting a pilot's license level."""
        level = sample_campaign.pilot_level(sample_pilot["id"])
        assert level == 0

    def test_pilot_is_dead(self, sample_campaign: Campaign, sample_pilot: dict):
        """Test checking if a pilot is dead."""
        is_dead = sample_campaign.pilot_is_dead(sample_pilot["id"])
        assert is_dead is False

    def test_validate_pilot_mech_links_unknown_pilot(self):
        """Test validation fails with unknown pilot in mech links."""
        with pytest.raises(ValidationError, match="unknown pilot"):
            Campaign(
                id="campaign-001",
                name="Test Campaign",
                pilots=[],
                pilot_mech_links=[
                    PilotMechAssignment(
                        pilot_id="unknown-pilot",
                        mech_id="mech-1",
                        mech_name="Test Mech",
                        mech_build={},
                    )
                ],
            )

    def test_validate_unique_pilot_ids(self, sample_pilot: dict):
        """Test validation fails with duplicate pilot IDs."""
        campaign_dict = {
            "id": "campaign-001",
            "name": "Test Campaign",
            "pilots": [sample_pilot, sample_pilot],
            "pilot_mech_links": [],
        }
        with pytest.raises(ValueError, match="Duplicate pilot IDs"):
            Campaign.model_validate(campaign_dict)

    def test_validate_unique_mech_ids(
        self, sample_pilot: dict, sample_mech_build: dict
    ):
        """Test validation fails with duplicate mech IDs."""
        campaign_dict = {
            "id": "campaign-001",
            "name": "Test Campaign",
            "pilots": [sample_pilot],
            "pilot_mech_links": [
                {
                    "pilot_id": sample_pilot["id"],
                    "mech_id": "same-mech-id",
                    "mech_name": "Mech 1",
                    "mech_build": sample_mech_build,
                    "is_active": True,
                },
                {
                    "pilot_id": sample_pilot["id"],
                    "mech_id": "same-mech-id",
                    "mech_name": "Mech 2",
                    "mech_build": sample_mech_build,
                    "is_active": False,
                },
            ],
        }
        with pytest.raises(ValueError, match="Duplicate mech IDs"):
            Campaign.model_validate(campaign_dict)

    def test_campaign_identity_storage(self, sample_campaign: Campaign):
        """Campaign identity data should be persisted on the model."""
        identity = CampaignIdentity(
            squad_name="ThirdComm Auxilia",
            patron="Union Navy",
            who_we_are="Freelance rangers",
            relationships=["GM: Pax", "Handler: Ort"],
        )
        sample_campaign.identity = identity
        assert sample_campaign.identity is not None
        assert sample_campaign.identity.patron == "Union Navy"

    def test_campaign_lobby_state_defaults(self, sample_campaign: Campaign):
        """Campaign lobby should enforce seat defaults."""
        mission_plan = MissionPrepPlan(mission_name="Operation Emberfall")
        sample_campaign.lobby_state = CampaignLobbyState(mission_plan=mission_plan)
        assert sample_campaign.lobby_state.preferred_pilot_count == 4
        assert sample_campaign.lobby_state.min_pilot_count == 3


class TestCampaignSerialization:
    """Tests for campaign serialization."""

    def test_campaign_to_dict(self, sample_campaign: Campaign):
        """Test converting campaign to dictionary."""
        data = campaign_to_dict(sample_campaign)
        assert data["id"] == "campaign-test-001"
        assert data["name"] == "Test Campaign"
        assert len(data["pilots"]) == 2
        assert len(data["pilot_mech_links"]) == 2

    def test_dict_to_campaign(self, sample_campaign: Campaign):
        """Test converting dictionary to campaign."""
        data = campaign_to_dict(sample_campaign)
        restored = dict_to_campaign(data)
        assert restored.id == sample_campaign.id
        assert restored.name == sample_campaign.name
        assert len(restored.pilots) == len(sample_campaign.pilots)

    def test_save_and_load_campaign(self, sample_campaign: Campaign):
        """Test saving and loading a campaign from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            save_campaign(sample_campaign, temp_path)
            loaded = load_campaign(temp_path)

            assert loaded.id == sample_campaign.id
            assert loaded.name == sample_campaign.name
            assert len(loaded.pilots) == len(sample_campaign.pilots)
            assert len(loaded.pilot_mech_links) == len(sample_campaign.pilot_mech_links)
        finally:
            os.unlink(temp_path)

    def test_save_and_load_string(self, sample_campaign: Campaign):
        """Test saving and loading a campaign from string."""
        json_str = save_campaign_to_string(sample_campaign)
        loaded = load_campaign_from_string(json_str)

        assert loaded.id == sample_campaign.id
        assert loaded.name == sample_campaign.name

    def test_pilot_survives_serialization(self, sample_pilot: dict):
        """Test that pilot data survives serialization."""
        campaign = Campaign(
            id="campaign-001",
            name="Test Campaign",
            pilots=[sample_pilot],
        )

        json_str = save_campaign_to_string(campaign)
        loaded = load_campaign_from_string(json_str)

        loaded_pilot = loaded.pilots[0]
        assert loaded_pilot["id"] == sample_pilot["id"]
        assert loaded_pilot["callsign"] == sample_pilot["callsign"]
        assert loaded_pilot["name"] == sample_pilot["name"]
        assert loaded_pilot["level"] == sample_pilot["level"]

    def test_mech_build_survives_serialization(
        self, sample_pilot: dict, sample_mech_build: dict
    ):
        """Test that mech build survives serialization."""
        link = PilotMechAssignment(
            pilot_id=sample_pilot["id"],
            mech_id="test-mech",
            mech_name="Test Mech",
            mech_build=sample_mech_build,
        )
        campaign = Campaign(
            id="campaign-001",
            name="Test Campaign",
            pilots=[sample_pilot],
            pilot_mech_links=[link],
        )

        json_str = save_campaign_to_string(campaign)
        loaded = load_campaign_from_string(json_str)

        loaded_link = loaded.pilot_mech_links[0]
        assert loaded_link.mech_id == "test-mech"
        assert loaded_link.mech_build["frame_id"] == "everest"
        assert len(loaded_link.mech_build["weapons"]) == 2

    def test_lobby_state_survives_serialization(self, sample_campaign: Campaign):
        """Ensure lobby state and mission prep persist through serialization."""
        sample_campaign.identity = CampaignIdentity(
            squad_name="Drake Company",
            patron="Karr Nexus",
        )
        sample_campaign.lobby_state = CampaignLobbyState(
            mission_plan=MissionPrepPlan(
                mission_name="Operation Radiant",
                support_assets=["Desert recon drone"],
            ),
            assigned_member_ids=["member-1"],
        )
        json_str = save_campaign_to_string(sample_campaign)
        restored = load_campaign_from_string(json_str)
        assert restored.identity is not None
        assert restored.identity.squad_name == "Drake Company"
        assert restored.lobby_state is not None
        assert restored.lobby_state.mission_plan.mission_name == "Operation Radiant"
        assert restored.lobby_state.assigned_member_ids == ["member-1"]


class TestCampaignValidation:
    """Tests for campaign validation."""

    def test_validate_campaign_valid(self, sample_campaign: Campaign):
        """Test validating a valid campaign."""
        data = campaign_to_dict(sample_campaign)
        errors = validate_campaign(data)
        assert errors == []

    def test_validate_campaign_missing_id(self):
        """Test validation fails when id is missing."""
        data = {"name": "Test Campaign"}
        errors = validate_campaign(data)
        assert len(errors) > 0
        assert any("id" in e.lower() for e in errors)

    def test_validate_campaign_missing_name(self):
        """Test validation fails when name is missing."""
        data = {"id": "campaign-001"}
        errors = validate_campaign(data)
        assert len(errors) > 0
        assert any("name" in e.lower() for e in errors)

    def test_validate_campaign_invalid_pilots(self):
        """Test validation fails when pilots is not a list."""
        data = {"id": "campaign-001", "name": "Test", "pilots": "not a list"}
        errors = validate_campaign(data)
        assert len(errors) > 0
        assert any("pilots" in e for e in errors)

    def test_validate_campaign_unknown_pilot_in_link(self, sample_pilot: dict):
        """Test validation fails when mech link references unknown pilot."""
        data = {
            "id": "campaign-001",
            "name": "Test",
            "pilots": [sample_pilot],
            "pilot_mech_links": [
                {
                    "pilot_id": "unknown-pilot-id",
                    "mech_id": "mech-1",
                    "mech_name": "Test Mech",
                    "mech_build": {},
                    "is_active": True,
                }
            ],
        }
        errors = validate_campaign(data)
        assert len(errors) > 0
        assert any("unknown pilot" in e.lower() for e in errors)

    def test_validate_campaign_synchronous(self, sample_campaign: Campaign):
        """Test synchronous validation of campaign object."""
        errors = validate_campaign_synchronous(sample_campaign)
        assert errors == []

    def test_is_campaign_valid(self, sample_campaign: Campaign):
        """Test checking if campaign is valid."""
        assert is_campaign_valid(sample_campaign) is True


class TestCampaignSummary:
    """Tests for campaign summary generation."""

    def test_get_campaign_summary(self, sample_campaign: Campaign):
        """Test generating a campaign summary."""
        summary = get_campaign_summary(sample_campaign)
        assert summary["id"] == "campaign-test-001"
        assert summary["name"] == "Test Campaign"
        assert summary["session_count"] == 0
        assert summary["pilot_count"] == 2
        assert summary["living_pilots"] == 2
        assert summary["total_missions"] == 0

    def test_get_campaign_summary_with_missions(
        self, sample_campaign: Campaign, sample_pilot: dict
    ):
        """Test generating a summary with missions."""
        record = CampaignMissionRecord(
            mission_id="mission-001",
            session_id="session-001",
            mission_name="Test Mission",
            outcome="success",
            completion_score=1.0,
            participating_pilot_ids=[sample_pilot["id"]],
        )
        sample_campaign.mission_history.append(record)

        summary = get_campaign_summary(sample_campaign)
        assert summary["total_missions"] == 1
        assert summary["successful_missions"] == 1
        assert summary["average_completion"] == 1.0


class TestCampaignMissionRecord:
    """Tests for CampaignMissionRecord model."""

    def test_create_mission_record(self):
        """Test creating a mission record."""
        record = CampaignMissionRecord(
            mission_id="mission-001",
            session_id="session-001",
            mission_name="First Mission",
            outcome="success",
            completion_score=1.0,
            participating_pilot_ids=["pilot-1", "pilot-2"],
        )
        assert record.mission_id == "mission-001"
        assert record.outcome == "success"
        assert record.completion_score == 1.0
        assert len(record.participating_pilot_ids) == 2

    def test_mission_record_tracks_reserve_delta(self):
        """Mission records capture reserves spent/earned for GM summaries."""
        record = CampaignMissionRecord(
            mission_id="mission-002",
            session_id="session-010",
            mission_name="Second Mission",
            outcome="partial",
            completion_score=0.5,
            participating_pilot_ids=["pilot-1"],
            reserves_spent=[{"id": "airstrike"}],
            reserves_earned=[{"id": "intel_cache"}],
            rewards=["+Rep", "Prototype unlocked"],
        )
        assert record.reserves_spent[0]["id"] == "airstrike"
        assert record.reserves_earned[0]["id"] == "intel_cache"
        assert record.rewards[-1] == "Prototype unlocked"


class TestActiveSessionMission:
    """Tests for ActiveSessionMission model."""

    def test_create_active_mission(self):
        """Test creating an active session mission."""
        mission = Mission(
            id="mission-001",
            name="Test Mission",
            description="A test mission",
            objectives=[
                MissionObjective(
                    id="obj-1",
                    description="Test objective",
                    objective_type="destroy",
                )
            ],
        )
        mission_state = MissionState(mission=mission)

        active = ActiveSessionMission(
            mission_state=mission_state.model_dump(),
            participating_pilot_ids=["pilot-1"],
        )
        assert active.participating_pilot_ids[0] == "pilot-1"
        assert len(active.participating_pilot_ids) == 1
