"""Pilot gear models for Lancer TTRPG."""

from collections import Counter
from typing import Literal
from pydantic import Field, model_validator
from core.shared.models import FrozenModel

from core.shared.effects import BreakTriggerType, EffectDuration, MechanicalEffect, StatModifier
from core.shared.effects_validation import validate_mechanical_effects
from core.shared.enums import ActionType, StatusType
from core.shared.payloads import (
    PilotAreaEffect,
    PilotDamageSpec,
    PilotGrenadePayload,
    PilotWeaponProfile,
)


PilotGearCategory = Literal["clothing", "armor", "weapon", "gear"]
PilotGearTagType = Literal[
    "sidearm",
    "archaic",
    "loading",
    "ordnance",
    "inaccurate",
]


class PilotGearTag(FrozenModel):
    """Structured tag for pilot gear items."""

    tag: PilotGearTagType
    value: int | None = None


class PilotChargePayload(FrozenModel):
    """Planted explosive charge payload for pilot gear."""

    name: str
    plant_action: ActionType
    detonate_action: ActionType
    area: PilotAreaEffect



class PilotFlightEffect(FrozenModel):
    """Flight behavior granted by pilot gear."""

    mode: Literal["move", "boost", "move_or_boost"]
    must_end_on_surface: bool = False



class PilotMedicalEffect(FrozenModel):
    """Medical gear payload."""

    name: str
    action: ActionType
    heal_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    heal_round_up: bool = True
    can_heal_down_and_out: bool = False
    restores_consciousness: bool = False
    applies_to_adjacent: bool = True
    affects_mechs: bool = False



class PilotStimEffect(FrozenModel):
    """Stim gear payload."""

    name: str
    effect: Literal["awake_alert", "calm_emotional", "heightened_senses"]
    duration_hours: int | None = Field(default=None, ge=0)


PilotEnvironmentalHazard = Literal["vacuum", "radiation"]
PilotBreathingDuration = Literal["limited", "scene", "mission", "unlimited"]


class PilotStealthEffect(FrozenModel):
    """Stealth action granted by pilot gear."""

    action_type: ActionType = "quick"
    status: StatusType = "invisible"
    duration: EffectDuration = "until_cleared"
    break_triggers: list[BreakTriggerType] = Field(default_factory=lambda: ["take_damage"])


class PilotEnvironmentalSeal(FrozenModel):
    """Environmental sealing provided by pilot gear."""

    protects_from: list[PilotEnvironmentalHazard] = Field(default_factory=list)
    water_breathing_duration: PilotBreathingDuration | None = None


class PilotZeroGEffect(FrozenModel):
    """Zero-g maneuvering enhancement."""

    maneuverability_bonus: bool = True


class PilotSurfaceMarkEffect(FrozenModel):
    """Surface marking that can transmit simple data when scanned."""

    visibility: Literal["visible", "invisible"] = "invisible"
    data_capacity: Literal["simple", "limited"] = "simple"
    requires_scan: bool = True


class PilotSustenanceEffect(FrozenModel):
    """Sustenance and climate control support."""

    duration_days: int = Field(default=0, ge=0)
    recharge_days_min: int | None = Field(default=None, ge=0)
    recharge_days_max: int | None = Field(default=None, ge=0)
    provides_temperature_control: bool = True
    prevents_hunger: bool = False


class PilotHackInterfaceEffect(FrozenModel):
    """Hack interface without external gear or rigs."""

    requires_external_gear: bool = False
    requires_rig: bool = False
    provides_omninet_access: bool = True


class PilotDroneEffect(FrozenModel):
    """Non-combat drone with relay capability."""

    max_range_miles: float = Field(default=0.5, ge=0)
    relays_audio: bool = True
    relays_visual: bool = True
    combat_capable: bool = False
    noisy: bool = True


class PilotDisguiseEffect(FrozenModel):
    """Holographic disguise and voice modulation."""

    holographic_projection: bool = True
    voice_modulation: bool = True
    fools_electronics: bool = True
    fails_close_inspection: bool = True


class PilotShelterEffect(FrozenModel):
    """Portable shelter that trades mobility for protection."""

    enter_action: ActionType = "full"
    exit_action: ActionType = "full"
    immunities: list[str] = Field(default_factory=list)
    environmental_seal: list[PilotEnvironmentalHazard] = Field(default_factory=list)
    air_supply_hours: int | None = Field(default=None, ge=0)
    evasion_override: int | None = Field(default=None, ge=0)
    slowed: bool = True
    restrict_actions_except_exit: bool = True


class PilotExtraArmEffect(FrozenModel):
    """Powered auxiliary arm for tools or weapons."""

    powered: bool = True
    mount_options: list[Literal["manipulator", "weapon", "tool"]] = Field(default_factory=list)


class PilotGearItemDefinition(FrozenModel):
    """Definition for a pilot gear item."""

    id: str = Field(..., description="Unique gear identifier")
    name: str = Field(..., description="Display name")
    category: PilotGearCategory
    limited_uses: int | None = Field(default=None, ge=0)
    tags: list[PilotGearTag] = Field(default_factory=list)
    weapon_profile: PilotWeaponProfile | None = None
    grenades: list[PilotGrenadePayload] = Field(default_factory=list)
    charges: list[PilotChargePayload] = Field(default_factory=list)
    flight: PilotFlightEffect | None = None
    medical: PilotMedicalEffect | None = None
    stim: PilotStimEffect | None = None
    stealth: PilotStealthEffect | None = None
    environmental_seal: PilotEnvironmentalSeal | None = None
    zero_g: PilotZeroGEffect | None = None
    surface_marking: PilotSurfaceMarkEffect | None = None
    sustenance: PilotSustenanceEffect | None = None
    hack_interface: PilotHackInterfaceEffect | None = None
    drone: PilotDroneEffect | None = None
    disguise: PilotDisguiseEffect | None = None
    shelter: PilotShelterEffect | None = None
    extra_arm: PilotExtraArmEffect | None = None
    effects: MechanicalEffect = Field(default_factory=MechanicalEffect)

    @model_validator(mode="after")
    def _validate_weapon_profile(self) -> "PilotGearItemDefinition":
        if self.category == "weapon" and self.weapon_profile is None:
            raise ValueError("Weapon gear items must define a weapon_profile.")
        return self



class PilotGearRules(FrozenModel):
    """Loadout limits for pilot gear."""

    clothing_required: bool = True
    armor_optional: bool = True
    max_weapons: int = 2
    max_gear: int = 3



DEFAULT_PILOT_GEAR_RULES = PilotGearRules()


class PilotLoadout(FrozenModel):
    """Pilot gear selection for a mission."""

    clothing: str | None = Field(default=None, description="Clothing item ID")
    armor: str | None = Field(default=None, description="Armor item ID")
    weapons: list[str] = Field(default_factory=list, max_length=2, description="Weapon item IDs")
    gear: list[str] = Field(default_factory=list, max_length=3, description="Other gear item IDs")


    def total_items(self) -> int:
        """Total number of selected items."""
        count = len(self.weapons) + len(self.gear)
        if self.clothing:
            count += 1
        if self.armor:
            count += 1
        return count


PILOT_GEAR_DEFINITIONS: list[PilotGearItemDefinition] = [
    PilotGearItemDefinition(
        id="flight_suit",
        name="Flight Suit",
        category="clothing",
    ),
    PilotGearItemDefinition(
        id="light_hardsuit",
        name="Light Hardsuit",
        category="armor",
        effects=MechanicalEffect(
            stat_mods=[StatModifier(stat="hp", value=3)],
        ),
    ),
    PilotGearItemDefinition(
        id="assault_hardsuit",
        name="Assault Hardsuit",
        category="armor",
        effects=MechanicalEffect(
            stat_mods=[
                StatModifier(stat="hp", value=3),
                StatModifier(stat="armor", value=1),
                StatModifier(stat="evasion", value=-2),
                StatModifier(stat="e_defense", value=-2),
            ],
        ),
    ),
    PilotGearItemDefinition(
        id="heavy_hardsuit",
        name="Heavy Hardsuit",
        category="armor",
        effects=MechanicalEffect(
            stat_mods=[
                StatModifier(stat="hp", value=3),
                StatModifier(stat="armor", value=2),
                StatModifier(stat="evasion", value=-4),
                StatModifier(stat="e_defense", value=-2),
                StatModifier(stat="speed", value=-1),
            ],
        ),
    ),
    PilotGearItemDefinition(
        id="mobility_hardsuit",
        name="Mobility Hardsuit",
        category="armor",
        flight=PilotFlightEffect(
            mode="move_or_boost",
            must_end_on_surface=True,
        ),
        effects=MechanicalEffect(
            stat_mods=[StatModifier(stat="speed", value=1)],
        ),
    ),
    PilotGearItemDefinition(
        id="stealth_hardsuit",
        name="Stealth Hardsuit",
        category="armor",
        stealth=PilotStealthEffect(),
        effects=MechanicalEffect(
            stat_mods=[
                StatModifier(stat="evasion", value=-2),
                StatModifier(stat="e_defense", value=-2),
            ],
        ),
    ),
    PilotGearItemDefinition(
        id="archaic_melee",
        name="Archaic Melee Weapon",
        category="weapon",
        tags=[PilotGearTag(tag="archaic")],
        weapon_profile=PilotWeaponProfile(
            range_type="threat",
            range=1,
            damage=PilotDamageSpec(flat=1, damage_type="kinetic"),
        ),
    ),
    PilotGearItemDefinition(
        id="alloy_composite_light",
        name="Alloy/Composite Weapon (Light)",
        category="weapon",
        weapon_profile=PilotWeaponProfile(
            range_type="threat",
            range=1,
            damage=PilotDamageSpec(flat=1, damage_type="kinetic"),
        ),
    ),
    PilotGearItemDefinition(
        id="alloy_composite_combat",
        name="Alloy/Composite Weapon (Combat)",
        category="weapon",
        weapon_profile=PilotWeaponProfile(
            range_type="threat",
            range=1,
            damage=PilotDamageSpec(flat=2, damage_type="kinetic"),
        ),
    ),
    PilotGearItemDefinition(
        id="alloy_composite_heavy",
        name="Alloy/Composite Weapon (Heavy)",
        category="weapon",
        tags=[PilotGearTag(tag="inaccurate")],
        weapon_profile=PilotWeaponProfile(
            range_type="threat",
            range=1,
            damage=PilotDamageSpec(flat=3, damage_type="kinetic"),
        ),
    ),
    PilotGearItemDefinition(
        id="archaic_ranged",
        name="Archaic Ranged Weapon",
        category="weapon",
        tags=[PilotGearTag(tag="archaic")],
        weapon_profile=PilotWeaponProfile(
            range_type="range",
            range=5,
            damage=PilotDamageSpec(flat=1, damage_type="kinetic"),
        ),
    ),
    PilotGearItemDefinition(
        id="signature_weapon_sidearm",
        name="Signature Weapon (Sidearm)",
        category="weapon",
        tags=[PilotGearTag(tag="sidearm")],
        weapon_profile=PilotWeaponProfile(
            range_type="range",
            range=3,
            damage=PilotDamageSpec(
                flat=1,
                damage_type="kinetic",
                damage_type_options=["kinetic", "energy", "explosive"],
            ),
        ),
    ),
    PilotGearItemDefinition(
        id="signature_weapon_combat",
        name="Signature Weapon (Combat)",
        category="weapon",
        weapon_profile=PilotWeaponProfile(
            range_type="range",
            range=5,
            damage=PilotDamageSpec(
                flat=2,
                damage_type="kinetic",
                damage_type_options=["kinetic", "energy", "explosive"],
            ),
        ),
    ),
    PilotGearItemDefinition(
        id="signature_weapon_heavy",
        name="Signature Weapon (Heavy)",
        category="weapon",
        tags=[PilotGearTag(tag="loading"), PilotGearTag(tag="ordnance")],
        weapon_profile=PilotWeaponProfile(
            range_type="range",
            range=10,
            damage=PilotDamageSpec(
                flat=4,
                damage_type="kinetic",
                damage_type_options=["kinetic", "energy", "explosive"],
            ),
        ),
    ),
    PilotGearItemDefinition(
        id="corrective",
        name="Corrective",
        category="gear",
        limited_uses=1,
        medical=PilotMedicalEffect(
            name="Corrective",
            action="full",
            heal_fraction=0.5,
            heal_round_up=True,
            can_heal_down_and_out=True,
            restores_consciousness=True,
        ),
    ),
    PilotGearItemDefinition(
        id="fragmentation_grenade",
        name="Fragmentation Grenade",
        category="gear",
        limited_uses=2,
        grenades=[
            PilotGrenadePayload(
                name="Frag Grenade",
                range=5,
                area=PilotAreaEffect(
                    pattern="blast",
                    size=1,
                    attack_vs="evasion",
                    damage=PilotDamageSpec(
                        damage_type="explosive",
                        flat=2,
                    ),
                ),
            ),
        ],
    ),
    PilotGearItemDefinition(
        id="nanite_spray",
        name="Nanite Spray",
        category="gear",
        surface_marking=PilotSurfaceMarkEffect(
            visibility="invisible",
            data_capacity="limited",
            requires_scan=True,
        ),
    ),
    PilotGearItemDefinition(
        id="patch",
        name="Patch",
        category="gear",
        limited_uses=1,
        medical=PilotMedicalEffect(
            name="Patch",
            action="full",
            heal_fraction=0.5,
            heal_round_up=True,
            can_heal_down_and_out=True,
            restores_consciousness=False,
            applies_to_adjacent=True,
        ),
    ),
    PilotGearItemDefinition(
        id="stims_kick",
        name="Stims (Kick)",
        category="gear",
        limited_uses=3,
        stim=PilotStimEffect(
            name="Kick",
            effect="awake_alert",
            duration_hours=30,
        ),
    ),
    PilotGearItemDefinition(
        id="stims_freeze",
        name="Stims (Freeze)",
        category="gear",
        limited_uses=3,
        stim=PilotStimEffect(
            name="Freeze",
            effect="calm_emotional",
        ),
    ),
    PilotGearItemDefinition(
        id="stims_juice",
        name="Stims (Juice)",
        category="gear",
        limited_uses=3,
        stim=PilotStimEffect(
            name="Juice",
            effect="heightened_senses",
        ),
    ),
    PilotGearItemDefinition(
        id="thermal_charge",
        name="Thermal Charge",
        category="gear",
        limited_uses=1,
        charges=[
            PilotChargePayload(
                name="Thermal Charge",
                plant_action="full",
                detonate_action="quick",
                area=PilotAreaEffect(
                    pattern="blast",
                    size=1,
                    attack_vs="evasion",
                    damage=PilotDamageSpec(
                        damage_type="energy",
                        flat=3,
                        ap=True,
                    ),
                ),
            ),
        ],
    ),
    PilotGearItemDefinition(
        id="antiphoton_visor",
        name="Antiphoton Visor",
        category="gear",
    ),
    PilotGearItemDefinition(
        id="camo_cloth",
        name="Camo Cloth",
        category="gear",
    ),
    PilotGearItemDefinition(
        id="dataplating",
        name="Dataplating",
        category="gear",
    ),
    PilotGearItemDefinition(
        id="extra_rations",
        name="Extra Rations",
        category="gear",
    ),
    PilotGearItemDefinition(
        id="flexsuit",
        name="Flexsuit",
        category="gear",
        sustenance=PilotSustenanceEffect(
            duration_days=7,
            recharge_days_min=1,
            recharge_days_max=2,
            provides_temperature_control=True,
            prevents_hunger=False,
        ),
    ),
    PilotGearItemDefinition(
        id="handheld_printer",
        name="Handheld Printer",
        category="gear",
    ),
    PilotGearItemDefinition(
        id="horus_subjectivity_suite",
        name="HORUS Subjectivity Enhancement Suite",
        category="gear",
        hack_interface=PilotHackInterfaceEffect(
            requires_external_gear=False,
            requires_rig=False,
            provides_omninet_access=True,
        ),
    ),
    PilotGearItemDefinition(
        id="infoskin",
        name="Infoskin",
        category="gear",
    ),
    PilotGearItemDefinition(
        id="mag_clamps",
        name="Mag-Clamps",
        category="gear",
        zero_g=PilotZeroGEffect(),
    ),
    PilotGearItemDefinition(
        id="omnihook",
        name="Omnihook",
        category="gear",
    ),
    PilotGearItemDefinition(
        id="personal_drone",
        name="Personal Drone",
        category="gear",
        drone=PilotDroneEffect(
            max_range_miles=0.5,
            relays_audio=True,
            relays_visual=True,
            combat_capable=False,
            noisy=True,
        ),
    ),
    PilotGearItemDefinition(
        id="prosocollar",
        name="Prosocollar",
        category="gear",
        disguise=PilotDisguiseEffect(
            holographic_projection=True,
            voice_modulation=True,
            fools_electronics=True,
            fails_close_inspection=True,
        ),
    ),
    PilotGearItemDefinition(
        id="smart_scope",
        name="Smart Scope",
        category="gear",
    ),
    PilotGearItemDefinition(
        id="sleeping_bag",
        name="Sleeping Bag",
        category="gear",
        shelter=PilotShelterEffect(
            enter_action="full",
            exit_action="full",
            immunities=["burn"],
            environmental_seal=["vacuum"],
            air_supply_hours=1,
            evasion_override=5,
            slowed=True,
            restrict_actions_except_exit=True,
        ),
    ),
    PilotGearItemDefinition(
        id="ssc_sylph",
        name="SSC Sylph",
        category="gear",
        environmental_seal=PilotEnvironmentalSeal(
            protects_from=["vacuum", "radiation"],
            water_breathing_duration="limited",
        ),
    ),
    PilotGearItemDefinition(
        id="sound_system",
        name="Sound System",
        category="gear",
    ),
    PilotGearItemDefinition(
        id="tertiary_arm",
        name="Tertiary Arm",
        category="gear",
        extra_arm=PilotExtraArmEffect(
            powered=True,
            mount_options=["manipulator", "weapon", "tool"],
        ),
    ),
    PilotGearItemDefinition(
        id="wilderness_survival_kit",
        name="Wilderness Survival Kit",
        category="gear",
    ),
    PilotGearItemDefinition(
        id="cooking_gear",
        name="Cooking Gear",
        category="gear",
    ),
]


def get_pilot_gear_definition(gear_id: str) -> PilotGearItemDefinition | None:
    """Look up a pilot gear definition by ID."""
    return PILOT_GEAR_DEFINITIONS_BY_ID.get(gear_id)


PILOT_GEAR_DEFINITIONS_BY_ID = {item.id: item for item in PILOT_GEAR_DEFINITIONS}


class PilotGearIssue(FrozenModel):
    """A pilot gear validation issue."""

    code: str
    message: str
    severity: Literal["error", "warning"] = "error"



class PilotGearValidation(FrozenModel):
    """Validation result for pilot gear loadouts."""

    valid: bool
    issues: list[PilotGearIssue] = Field(default_factory=list)



def _iter_loadout_ids(loadout: PilotLoadout) -> list[str]:
    ids: list[str] = []
    if loadout.clothing:
        ids.append(loadout.clothing)
    if loadout.armor:
        ids.append(loadout.armor)
    ids.extend(loadout.weapons)
    ids.extend(loadout.gear)
    return ids


def validate_pilot_loadout(
    loadout: PilotLoadout,
    rules: PilotGearRules = DEFAULT_PILOT_GEAR_RULES,
    gear_definitions: dict[str, PilotGearItemDefinition] | None = None,
) -> PilotGearValidation:
    """Validate a pilot gear loadout against selection rules."""
    issues: list[PilotGearIssue] = []
    definitions = gear_definitions or PILOT_GEAR_DEFINITIONS_BY_ID

    if rules.clothing_required and not loadout.clothing:
        issues.append(
            PilotGearIssue(
                code="missing_clothing",
                message="Pilot loadout must include clothing.",
            )
        )

    if not rules.armor_optional and not loadout.armor:
        issues.append(
            PilotGearIssue(
                code="missing_armor",
                message="Pilot loadout must include armor.",
            )
        )

    if len(loadout.weapons) > rules.max_weapons:
        issues.append(
            PilotGearIssue(
                code="too_many_weapons",
                message=f"Pilot loadout exceeds max weapons ({rules.max_weapons}).",
            )
        )

    if len(loadout.gear) > rules.max_gear:
        issues.append(
            PilotGearIssue(
                code="too_many_gear_items",
                message=f"Pilot loadout exceeds max gear items ({rules.max_gear}).",
            )
        )

    for gear_id in _iter_loadout_ids(loadout):
        definition = definitions.get(gear_id)
        if not definition:
            issues.append(
                PilotGearIssue(
                    code="unknown_gear_id",
                    message=f"Unknown pilot gear ID: {gear_id}.",
                )
            )

    if loadout.clothing:
        definition = definitions.get(loadout.clothing)
        if definition and definition.category != "clothing":
            issues.append(
                PilotGearIssue(
                    code="invalid_clothing_category",
                    message=f"Clothing item '{loadout.clothing}' is not clothing.",
                )
            )

    if loadout.armor:
        definition = definitions.get(loadout.armor)
        if definition and definition.category != "armor":
            issues.append(
                PilotGearIssue(
                    code="invalid_armor_category",
                    message=f"Armor item '{loadout.armor}' is not armor.",
                )
            )

    for weapon_id in loadout.weapons:
        definition = definitions.get(weapon_id)
        if definition and definition.category != "weapon":
            issues.append(
                PilotGearIssue(
                    code="invalid_weapon_category",
                    message=f"Weapon item '{weapon_id}' is not a weapon.",
                )
            )

    for gear_id in loadout.gear:
        definition = definitions.get(gear_id)
        if definition and definition.category != "gear":
            issues.append(
                PilotGearIssue(
                    code="invalid_gear_category",
                    message=f"Gear item '{gear_id}' is not gear.",
                )
            )

    id_counts = Counter(_iter_loadout_ids(loadout))
    duplicates = [gear_id for gear_id, count in id_counts.items() if count > 1]
    if duplicates:
        issues.append(
            PilotGearIssue(
                code="duplicate_gear_items",
                message=f"Duplicate gear items are not allowed: {', '.join(duplicates)}.",
            )
        )

    effects: list[MechanicalEffect] = []
    for gear_id in _iter_loadout_ids(loadout):
        definition = definitions.get(gear_id)
        if definition:
            effects.append(definition.effects)
    for issue in validate_mechanical_effects(effects):
        issues.append(
            PilotGearIssue(
                code=f"effect_validation_{issue.severity}",
                message=f"Effect validation: {issue.message}",
                severity=issue.severity,
            )
        )

    return PilotGearValidation(valid=not any(i.severity == "error" for i in issues), issues=issues)


def get_pilot_gear_stat_mods(
    loadout: PilotLoadout,
    gear_definitions: dict[str, PilotGearItemDefinition] | None = None,
) -> dict[str, int]:
    """Aggregate stat modifiers provided by a pilot gear loadout."""
    totals: dict[str, int] = {}
    definitions = gear_definitions or PILOT_GEAR_DEFINITIONS_BY_ID
    for gear_id in _iter_loadout_ids(loadout):
        definition = definitions.get(gear_id)
        if not definition:
            continue
        for mod in definition.effects.stat_mods:
            totals[mod.stat] = totals.get(mod.stat, 0) + mod.value
    return totals
