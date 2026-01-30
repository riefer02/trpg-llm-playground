"""Tests for character validation."""


from core.character import (
    Character,
    MechConfiguration,
    validate_character,
)
from core.pilot.pilot import Pilot
from core.pilot.skill import SkillSet, PilotTrigger
from core.pilot.talent import Talent
from core.pilot.license import License
from core.pilot.core_bonus import CoreBonus
from core.mech.build import MechBuild, MountedWeapon, InstalledSystem


def _make_ll0_pilot() -> Pilot:
    """Create a valid LL0 pilot."""
    return Pilot(
        id="pilot_test",
        callsign="TEST",
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


def _make_everest_mech() -> MechConfiguration:
    """Create a GMS Everest mech."""
    return MechConfiguration(
        id="mech_test",
        name="Test Mech",
        frame_id="gms_everest",
        build=MechBuild(frame_id="gms_everest"),
    )


class TestCharacterValidation:
    """Tests for validate_character()."""

    def test_valid_ll0_character(self) -> None:
        """A properly configured LL0 character validates."""
        pilot = _make_ll0_pilot()
        mech = _make_everest_mech()
        character = Character(
            pilot=pilot,
            mechs=[mech],
            active_mech_id=mech.id,
        )

        result = validate_character(character)

        # Should be valid (warnings are OK)
        assert result.valid, f"Issues: {[i.message for i in result.issues]}"

    def test_character_without_mech_has_warning(self) -> None:
        """A character without mechs gets a warning."""
        pilot = _make_ll0_pilot()
        character = Character(pilot=pilot)

        result = validate_character(character)

        warning_codes = [i.code for i in result.issues if i.severity == "warning"]
        assert "no_mechs" in warning_codes

    def test_character_with_mechs_but_no_active_has_warning(self) -> None:
        """A character with mechs but no active mech gets a warning."""
        pilot = _make_ll0_pilot()
        mech = _make_everest_mech()
        character = Character(
            pilot=pilot,
            mechs=[mech],
            active_mech_id=None,
        )

        result = validate_character(character)

        warning_codes = [i.code for i in result.issues if i.severity == "warning"]
        assert "no_active_mech" in warning_codes


class TestLL0Validation:
    """Tests for LL0-specific validation rules."""

    def test_ll0_non_gms_frame_rejected(self) -> None:
        """LL0 character with non-GMS frame is invalid."""
        pilot = _make_ll0_pilot()
        # IPS-N Blackbeard requires license
        mech = MechConfiguration(
            id="mech_test",
            name="Test",
            frame_id="ipsn_blackbeard",
            build=MechBuild(frame_id="ipsn_blackbeard"),
        )
        character = Character(
            pilot=pilot,
            mechs=[mech],
            active_mech_id=mech.id,
        )

        result = validate_character(character)

        error_codes = [i.code for i in result.issues if i.severity == "error"]
        assert any("ll0_non_gms_frame" in code for code in error_codes)

    def test_ll0_non_gms_weapon_rejected(self) -> None:
        """LL0 character with non-GMS weapon is invalid."""
        pilot = _make_ll0_pilot()
        # IPS-N Hand Cannon requires Raleigh license
        build = MechBuild(
            frame_id="gms_everest",
            weapons=[
                MountedWeapon(mount_index=2, weapon_id="hand_cannon", weapon_size="aux"),
            ],
        )
        mech = MechConfiguration(
            id="mech_test",
            name="Test",
            frame_id="gms_everest",
            build=build,
        )
        character = Character(
            pilot=pilot,
            mechs=[mech],
            active_mech_id=mech.id,
        )

        result = validate_character(character)

        error_codes = [i.code for i in result.issues if i.severity == "error"]
        assert any("ll0_non_gms_weapon" in code or "license_requirement" in code for code in error_codes)

    def test_ll0_non_gms_system_rejected(self) -> None:
        """LL0 character with non-GMS system is invalid."""
        pilot = _make_ll0_pilot()
        # IPS-N system requires license
        build = MechBuild(
            frame_id="gms_everest",
            systems=[
                InstalledSystem(system_id="ipsn_breaching_charges", sp_cost=2),
            ],
        )
        mech = MechConfiguration(
            id="mech_test",
            name="Test",
            frame_id="gms_everest",
            build=build,
        )
        character = Character(
            pilot=pilot,
            mechs=[mech],
            active_mech_id=mech.id,
        )

        result = validate_character(character)

        error_codes = [i.code for i in result.issues if i.severity == "error"]
        assert any("ll0_non_gms_system" in code or "license_requirement" in code for code in error_codes)

    def test_ll0_with_gms_gear_valid(self) -> None:
        """LL0 character with GMS gear is valid."""
        pilot = _make_ll0_pilot()
        build = MechBuild(
            frame_id="gms_everest",
            weapons=[
                MountedWeapon(mount_index=0, weapon_id="anti_material_rifle", weapon_size="heavy"),
                MountedWeapon(mount_index=1, weapon_id="assault_rifle", weapon_size="main"),
            ],
            systems=[
                InstalledSystem(system_id="gms_manipulators", sp_cost=1),
            ],
        )
        mech = MechConfiguration(
            id="mech_test",
            name="Test",
            frame_id="gms_everest",
            build=build,
        )
        character = Character(
            pilot=pilot,
            mechs=[mech],
            active_mech_id=mech.id,
        )

        result = validate_character(character)

        # Should be valid
        assert result.valid, f"Issues: {[i.message for i in result.issues]}"


class TestPilotProgressionValidation:
    """Tests for pilot progression validation within character."""

    def test_too_many_skill_points_rejected(self) -> None:
        """Character with too many mech skill points is invalid."""
        pilot = Pilot(
            id="pilot_test",
            callsign="TEST",
            level=0,
            skills=SkillSet(hull=4),  # Too many points for LL0
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
        character = Character(pilot=pilot)

        result = validate_character(character)

        error_codes = [i.code for i in result.issues if i.severity == "error"]
        assert any("skill_points" in code for code in error_codes)

    def test_too_many_talent_points_rejected(self) -> None:
        """Character with too many talent points is invalid."""
        pilot = Pilot(
            id="pilot_test",
            callsign="TEST",
            level=0,
            skills=SkillSet(hull=2),
            triggers=[
                PilotTrigger(trigger_id="assault", rank=2),
                PilotTrigger(trigger_id="survive", rank=2),
                PilotTrigger(trigger_id="spot", rank=2),
                PilotTrigger(trigger_id="take_someone_out", rank=2),
            ],
            talents=[
                Talent(talent_id="ace", rank=2),  # Rank 2 not allowed at LL0
                Talent(talent_id="combined_arms", rank=1),
                Talent(talent_id="crack_shot", rank=1),
            ],
        )
        character = Character(pilot=pilot)

        result = validate_character(character)

        error_codes = [i.code for i in result.issues if i.severity == "error"]
        assert any("talent" in code for code in error_codes)


class TestHigherLevelValidation:
    """Tests for higher level character validation."""

    def test_ll3_with_license_and_licensed_gear(self) -> None:
        """LL3 character can use licensed gear if they have the license."""
        pilot = Pilot(
            id="pilot_test",
            callsign="TEST",
            level=3,
            skills=SkillSet(hull=5),  # 5 points at LL3
            triggers=[
                PilotTrigger(trigger_id="assault", rank=6),
                PilotTrigger(trigger_id="survive", rank=2),
                PilotTrigger(trigger_id="spot", rank=4),
                PilotTrigger(trigger_id="take_someone_out", rank=2),
            ],
            talents=[
                Talent(talent_id="ace", rank=2),
                Talent(talent_id="combined_arms", rank=2),
                Talent(talent_id="crack_shot", rank=2),
            ],
            licenses=[
                License(license_id="raleigh", rank=3),
            ],
            core_bonuses=[
                CoreBonus(core_bonus_id="ipsn_reinforced_frame"),
            ],
        )
        # IPS-N Raleigh requires Raleigh license rank 2
        mech = MechConfiguration(
            id="mech_test",
            name="Test",
            frame_id="ipsn_raleigh",
            build=MechBuild(frame_id="ipsn_raleigh"),
        )
        character = Character(
            pilot=pilot,
            mechs=[mech],
            active_mech_id=mech.id,
        )

        result = validate_character(character)

        # Should be valid (core bonus adds +5 HP)
        assert result.valid, f"Issues: {[i.message for i in result.issues]}"

        # Core bonus effects should be populated
        assert len(character.core_bonus_effects) == 1

        # Mech stats should include core bonus
        assert character.active_mech_stats is not None
        # Raleigh base HP is 10, grit +2, hull +5 = +10 HP, core bonus +5
        # HP = 10 + 2 + 10 + 5 = 27
        assert character.active_mech_stats.hp == 27
