"""Tests for LL0 character creation factory."""

import pytest

from core.character import (
    create_ll0_character,
    create_empty_character,
    validate_character,
)
from core.pilot.skill import SkillSet, PilotTrigger
from core.pilot.talent import Talent
from core.pilot.background import Background
from core.pilot.gear import PilotLoadout
from core.mech.build import MechBuild


class TestCreateLL0Character:
    """Tests for create_ll0_character() factory."""

    def test_create_ll0_character_defaults(self) -> None:
        """Can create LL0 character with defaults."""
        character = create_ll0_character(callsign="ALPHA")

        assert character.pilot.callsign == "ALPHA"
        assert character.pilot.level == 0
        assert len(character.mechs) == 1
        assert character.active_mech_id is not None
        assert character.active_mech is not None

    def test_ll0_character_validates(self) -> None:
        """Default LL0 character passes validation."""
        character = create_ll0_character(callsign="ALPHA")

        result = validate_character(character)

        assert result.valid, f"Issues: {[i.message for i in result.issues]}"

    def test_ll0_pilot_has_correct_skills(self) -> None:
        """LL0 pilot has exactly 2 skill points."""
        character = create_ll0_character(callsign="ALPHA")

        assert character.pilot.skills.total_points() == 2
        assert character.pilot.skills.hull == 2  # Default

    def test_ll0_pilot_has_correct_triggers(self) -> None:
        """LL0 pilot has 4 triggers at +2 each."""
        character = create_ll0_character(callsign="ALPHA")

        assert len(character.pilot.triggers) == 4
        assert all(t.rank == 2 for t in character.pilot.triggers)
        assert character.pilot.total_trigger_points() == 8

    def test_ll0_pilot_has_correct_talents(self) -> None:
        """LL0 pilot has 3 rank I talents."""
        character = create_ll0_character(callsign="ALPHA")

        assert len(character.pilot.talents) == 3
        assert all(t.rank == 1 for t in character.pilot.talents)
        assert character.pilot.total_talent_ranks() == 3

    def test_ll0_pilot_has_no_licenses(self) -> None:
        """LL0 pilot has no licenses."""
        character = create_ll0_character(callsign="ALPHA")

        assert character.pilot.licenses == []
        assert character.pilot.total_license_levels() == 0

    def test_ll0_pilot_has_no_core_bonuses(self) -> None:
        """LL0 pilot has no core bonuses."""
        character = create_ll0_character(callsign="ALPHA")

        assert character.pilot.core_bonuses == []
        assert character.core_bonus_effects == []

    def test_ll0_mech_is_gms_everest(self) -> None:
        """LL0 mech is GMS Everest."""
        character = create_ll0_character(callsign="ALPHA")

        mech = character.active_mech
        assert mech is not None
        assert mech.frame_id == "gms_everest"

    def test_ll0_mech_stats_correct(self) -> None:
        """LL0 mech has correct computed stats."""
        character = create_ll0_character(callsign="ALPHA")

        stats = character.active_mech_stats
        assert stats is not None

        # GMS Everest base stats + Hull +2 + Grit 0
        # HP = 10 + 0 + 4 = 14
        assert stats.hp == 14
        # Evasion = 8 + 0 = 8
        assert stats.evasion == 8
        # Speed = 4 + 0 = 4
        assert stats.speed == 4
        # Heat cap = 6 + 0 = 6
        assert stats.heat_cap == 6
        # System points = 6 + 0 + 0 = 6
        assert stats.system_points == 6


class TestCreateLL0CharacterCustomization:
    """Tests for customizing LL0 character creation."""

    def test_custom_name(self) -> None:
        """Can set pilot real name."""
        character = create_ll0_character(callsign="ALPHA", name="John Smith")

        assert character.pilot.name == "John Smith"

    def test_custom_background(self) -> None:
        """Can set pilot background."""
        background = Background(
            id="colonist",
            name="Colonist",
            triggers=["survive", "word_on_the_streets", "charm", "get_a_hold_of_something"],
        )
        character = create_ll0_character(callsign="ALPHA", background=background)

        assert character.pilot.background is not None
        assert character.pilot.background.name == "Colonist"

    def test_custom_skills(self) -> None:
        """Can set custom skill allocation."""
        skills = SkillSet(agility=1, systems=1)
        character = create_ll0_character(callsign="ALPHA", skills=skills)

        assert character.pilot.skills.agility == 1
        assert character.pilot.skills.systems == 1
        assert character.pilot.skills.hull == 0

    def test_custom_triggers(self) -> None:
        """Can set custom triggers."""
        triggers = [
            PilotTrigger(trigger_id="hack_or_fix", rank=2),
            PilotTrigger(trigger_id="charm", rank=2),
            PilotTrigger(trigger_id="read_a_situation", rank=2),
            PilotTrigger(trigger_id="investigate", rank=2),
        ]
        character = create_ll0_character(callsign="ALPHA", triggers=triggers)

        trigger_ids = [t.trigger_id for t in character.pilot.triggers]
        assert "hack_or_fix" in trigger_ids
        assert "charm" in trigger_ids

    def test_custom_talents(self) -> None:
        """Can set custom talents."""
        talents = [
            Talent(talent_id="technophile", rank=1),
            Talent(talent_id="leader", rank=1),
            Talent(talent_id="hacker", rank=1),
        ]
        character = create_ll0_character(callsign="ALPHA", talents=talents)

        talent_ids = [t.talent_id for t in character.pilot.talents]
        assert "technophile" in talent_ids
        assert "leader" in talent_ids

    def test_custom_mech_name(self) -> None:
        """Can set custom mech name."""
        character = create_ll0_character(callsign="ALPHA", mech_name="RAIJIN")

        assert character.active_mech is not None
        assert character.active_mech.name == "RAIJIN"

    def test_custom_pilot_gear(self) -> None:
        """Can set pilot gear loadout."""
        loadout = PilotLoadout(
            clothing="flight_suit",
            armor="light_hardsuit",
            weapons=["alloy_composite_light"],
            gear=["corrective"],
        )

        character = create_ll0_character(callsign="ALPHA", pilot_gear=loadout)

        assert character.pilot.pilot_gear is not None
        assert character.pilot.pilot_gear.clothing == "flight_suit"
        assert character.pilot.pilot_gear.weapons == ["alloy_composite_light"]

    def test_mech_name_defaults_to_callsign(self) -> None:
        """Mech name defaults to callsign if not provided."""
        character = create_ll0_character(callsign="OMEGA")

        assert character.active_mech is not None
        assert character.active_mech.name == "OMEGA"


class TestCreateLL0CharacterValidation:
    """Tests for validation during LL0 character creation."""

    def test_invalid_skill_points_rejected(self) -> None:
        """Too many skill points raises ValueError."""
        skills = SkillSet(hull=3)  # 3 points, should be 2

        with pytest.raises(ValueError, match="exactly 2 mech skill points"):
            create_ll0_character(callsign="ALPHA", skills=skills)

    def test_invalid_trigger_count_rejected(self) -> None:
        """Wrong number of triggers raises ValueError."""
        triggers = [
            PilotTrigger(trigger_id="assault", rank=2),
            PilotTrigger(trigger_id="survive", rank=2),
            # Only 2 triggers, should be 4
        ]

        with pytest.raises(ValueError, match="exactly 4 triggers"):
            create_ll0_character(callsign="ALPHA", triggers=triggers)

    def test_invalid_trigger_points_rejected(self) -> None:
        """Wrong trigger point total raises ValueError."""
        triggers = [
            PilotTrigger(trigger_id="assault", rank=4),  # +4 instead of +2
            PilotTrigger(trigger_id="survive", rank=2),
            PilotTrigger(trigger_id="spot", rank=2),
            PilotTrigger(trigger_id="take_someone_out", rank=2),
        ]

        with pytest.raises(ValueError, match="8 trigger points"):
            create_ll0_character(callsign="ALPHA", triggers=triggers)

    def test_invalid_talent_points_rejected(self) -> None:
        """Wrong talent point total raises ValueError."""
        talents = [
            Talent(talent_id="ace", rank=1),
            Talent(talent_id="combined_arms", rank=1),
            # Only 2 talents = 2 points, should be 3
        ]

        with pytest.raises(ValueError, match="exactly 3 talent points"):
            create_ll0_character(callsign="ALPHA", talents=talents)

    def test_rank_ii_talents_rejected(self) -> None:
        """Rank II talents at LL0 raises ValueError."""
        talents = [
            Talent(talent_id="ace", rank=2),  # Rank 2 not allowed at LL0
            Talent(talent_id="combined_arms", rank=1),
            # Total is 3 points, but rank 2 not allowed
        ]

        with pytest.raises(ValueError, match="rank I talents"):
            create_ll0_character(callsign="ALPHA", talents=talents)

    def test_non_everest_frame_rejected(self) -> None:
        """Non-GMS Everest frame at LL0 raises ValueError."""
        build = MechBuild(frame_id="ipsn_blackbeard")

        with pytest.raises(ValueError, match="GMS Everest frame"):
            create_ll0_character(callsign="ALPHA", mech_build=build)

    def test_invalid_pilot_gear_rejected(self) -> None:
        """Invalid pilot gear loadout raises ValueError."""
        loadout = PilotLoadout(weapons=["alloy_composite_light"])

        with pytest.raises(ValueError, match="Invalid pilot gear loadout"):
            create_ll0_character(callsign="ALPHA", pilot_gear=loadout)


class TestCreateEmptyCharacter:
    """Tests for create_empty_character() factory."""

    def test_create_empty_character(self) -> None:
        """Can create a minimal character."""
        character = create_empty_character(callsign="TEST")

        assert character.pilot.callsign == "TEST"
        assert character.mechs == []
        assert character.active_mech_id is None

    def test_empty_character_has_warnings(self) -> None:
        """Empty character has validation warnings."""
        character = create_empty_character(callsign="TEST")

        result = validate_character(character)

        # Should have warnings but not errors (empty is allowed)
        warning_codes = [i.code for i in result.issues if i.severity == "warning"]
        assert "no_mechs" in warning_codes
