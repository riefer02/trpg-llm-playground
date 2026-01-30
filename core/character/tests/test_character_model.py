"""Tests for the Character model."""

import pytest

from core.character import Character, MechConfiguration
from core.pilot.pilot import Pilot
from core.pilot.skill import SkillSet, PilotTrigger
from core.pilot.talent import Talent
from core.mech.build import MechBuild


def _make_ll0_pilot(callsign: str = "TEST") -> Pilot:
    """Create a minimal LL0 pilot for testing."""
    return Pilot(
        id="pilot_test",
        callsign=callsign,
        level=0,
        skills=SkillSet(hull=2),
        triggers=[
            PilotTrigger(trigger_id="assault", rank=2),
            PilotTrigger(trigger_id="survive", rank=2),
            PilotTrigger(trigger_id="spot", rank=2),
            PilotTrigger(trigger_id="take_someone_out", rank=2),
        ],
        talents=[
            Talent(talent_id="ace", rank=1),
            Talent(talent_id="combined_arms", rank=1),
            Talent(talent_id="crack_shot", rank=1),
        ],
    )


def _make_everest_mech(name: str = "TEST_MECH") -> MechConfiguration:
    """Create a minimal GMS Everest mech for testing."""
    return MechConfiguration(
        id="mech_test",
        name=name,
        frame_id="gms_everest",
        build=MechBuild(frame_id="gms_everest"),
    )


class TestCharacterModel:
    """Tests for the Character model basics."""

    def test_create_character_with_pilot_only(self) -> None:
        """Can create a character with just a pilot."""
        pilot = _make_ll0_pilot()
        character = Character(pilot=pilot)

        assert character.pilot.callsign == "TEST"
        assert character.mechs == []
        assert character.active_mech_id is None
        assert character.active_mech is None
        assert character.active_mech_stats is None

    def test_create_character_with_mech(self) -> None:
        """Can create a character with pilot and mech."""
        pilot = _make_ll0_pilot()
        mech = _make_everest_mech()

        character = Character(
            pilot=pilot,
            mechs=[mech],
            active_mech_id=mech.id,
        )

        assert len(character.mechs) == 1
        assert character.active_mech_id == mech.id
        assert character.active_mech is not None
        assert character.active_mech.name == "TEST_MECH"

    def test_active_mech_stats_computed(self) -> None:
        """Active mech stats are computed from pilot data."""
        pilot = _make_ll0_pilot()
        mech = _make_everest_mech()

        character = Character(
            pilot=pilot,
            mechs=[mech],
            active_mech_id=mech.id,
        )

        stats = character.active_mech_stats
        assert stats is not None

        # GMS Everest base HP is 10, pilot has Hull +2
        # HP = 10 (base) + 0 (grit at LL0) + 4 (2 * hull) = 14
        assert stats.hp == 14

        # Evasion = 8 (base) + 0 (agility) = 8
        assert stats.evasion == 8

    def test_active_mech_id_must_exist(self) -> None:
        """active_mech_id must reference an existing mech."""
        pilot = _make_ll0_pilot()

        with pytest.raises(ValueError, match="not found in mechs"):
            Character(
                pilot=pilot,
                mechs=[],
                active_mech_id="nonexistent_mech",
            )

    def test_core_bonus_effects_empty_at_ll0(self) -> None:
        """LL0 character has no core bonus effects."""
        pilot = _make_ll0_pilot()
        character = Character(pilot=pilot)

        assert character.core_bonus_effects == []


class TestCharacterMechManagement:
    """Tests for mech management methods."""

    def test_add_mech(self) -> None:
        """Can add a mech to a character."""
        pilot = _make_ll0_pilot()
        character = Character(pilot=pilot)
        mech = _make_everest_mech()

        new_character = character.add_mech(mech)

        # Original unchanged
        assert len(character.mechs) == 0

        # New character has mech
        assert len(new_character.mechs) == 1
        assert new_character.active_mech_id == mech.id  # First mech becomes active

    def test_add_mech_duplicate_id_fails(self) -> None:
        """Cannot add a mech with duplicate ID."""
        pilot = _make_ll0_pilot()
        mech = _make_everest_mech()
        character = Character(pilot=pilot, mechs=[mech], active_mech_id=mech.id)

        with pytest.raises(ValueError, match="already exists"):
            character.add_mech(mech)

    def test_remove_mech(self) -> None:
        """Can remove a mech from a character."""
        pilot = _make_ll0_pilot()
        mech = _make_everest_mech()
        character = Character(pilot=pilot, mechs=[mech], active_mech_id=mech.id)

        new_character = character.remove_mech(mech.id)

        assert len(new_character.mechs) == 0
        assert new_character.active_mech_id is None  # Active cleared

    def test_remove_nonexistent_mech_fails(self) -> None:
        """Cannot remove a mech that doesn't exist."""
        pilot = _make_ll0_pilot()
        character = Character(pilot=pilot)

        with pytest.raises(ValueError, match="not found"):
            character.remove_mech("nonexistent")

    def test_set_active_mech(self) -> None:
        """Can change the active mech."""
        pilot = _make_ll0_pilot()
        mech1 = MechConfiguration(
            id="mech_1", name="Alpha", frame_id="gms_everest", build=MechBuild(frame_id="gms_everest")
        )
        mech2 = MechConfiguration(
            id="mech_2", name="Beta", frame_id="gms_everest", build=MechBuild(frame_id="gms_everest")
        )
        character = Character(
            pilot=pilot,
            mechs=[mech1, mech2],
            active_mech_id=mech1.id,
        )

        new_character = character.set_active_mech(mech2.id)

        assert new_character.active_mech_id == mech2.id
        assert new_character.active_mech is not None
        assert new_character.active_mech.name == "Beta"

    def test_set_active_mech_none(self) -> None:
        """Can clear the active mech."""
        pilot = _make_ll0_pilot()
        mech = _make_everest_mech()
        character = Character(pilot=pilot, mechs=[mech], active_mech_id=mech.id)

        new_character = character.set_active_mech(None)

        assert new_character.active_mech_id is None
        assert new_character.active_mech is None

    def test_update_mech(self) -> None:
        """Can update a mech configuration."""
        pilot = _make_ll0_pilot()
        mech = _make_everest_mech()
        character = Character(pilot=pilot, mechs=[mech], active_mech_id=mech.id)

        new_character = character.update_mech(mech.id, name="NEW_NAME")

        assert new_character.mechs[0].name == "NEW_NAME"
        assert character.mechs[0].name == "TEST_MECH"  # Original unchanged


class TestCharacterPilotManagement:
    """Tests for pilot update methods."""

    def test_update_pilot(self) -> None:
        """Can update pilot fields."""
        pilot = _make_ll0_pilot("OLD_CALLSIGN")
        character = Character(pilot=pilot)

        new_character = character.update_pilot(callsign="NEW_CALLSIGN")

        assert new_character.pilot.callsign == "NEW_CALLSIGN"
        assert character.pilot.callsign == "OLD_CALLSIGN"  # Original unchanged

    def test_update_pilot_skills_affects_mech_stats(self) -> None:
        """Updating pilot skills changes computed mech stats."""
        pilot = _make_ll0_pilot()
        mech = _make_everest_mech()
        character = Character(pilot=pilot, mechs=[mech], active_mech_id=mech.id)

        # Original: Hull +2 → HP = 10 + 0 + 4 = 14
        assert character.active_mech_stats is not None
        original_hp = character.active_mech_stats.hp

        # Update to Hull +4 → HP = 10 + 0 + 8 = 18
        new_skills = SkillSet(hull=4)
        new_character = character.update_pilot(skills=new_skills)

        assert new_character.active_mech_stats is not None
        assert new_character.active_mech_stats.hp == original_hp + 4


class TestMechConfiguration:
    """Tests for the MechConfiguration model."""

    def test_create_mech_configuration(self) -> None:
        """Can create a basic mech configuration."""
        mech = MechConfiguration(
            id="test_mech",
            name="Raijin",
            frame_id="gms_everest",
        )

        assert mech.id == "test_mech"
        assert mech.name == "Raijin"
        assert mech.frame_id == "gms_everest"

    def test_mech_configuration_with_build(self) -> None:
        """Can create a mech configuration with loadout."""
        build = MechBuild(
            frame_id="gms_everest",
            weapons=[],
            systems=[],
        )
        mech = MechConfiguration(
            id="test_mech",
            name="Raijin",
            frame_id="gms_everest",
            build=build,
        )

        assert mech.build.frame_id == "gms_everest"

    def test_get_frame_returns_definition(self) -> None:
        """get_frame() returns the frame definition."""
        mech = MechConfiguration(
            id="test_mech",
            name="Test",
            frame_id="gms_everest",
        )

        frame = mech.get_frame()

        assert frame is not None
        assert frame.name == "GMS Everest"

    def test_get_frame_returns_none_for_unknown(self) -> None:
        """get_frame() returns None for unknown frame."""
        mech = MechConfiguration(
            id="test_mech",
            name="Test",
            frame_id="unknown_frame",
        )

        frame = mech.get_frame()

        assert frame is None
