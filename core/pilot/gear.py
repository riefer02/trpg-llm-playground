"""Pilot gear models for Lancer TTRPG."""

from collections import Counter
from typing import Literal
from pydantic import BaseModel, Field

from core.shared.effects import MechanicalEffect, StatModifier
from core.shared.enums import ActionType, DamageType, RangeType


PilotGearCategory = Literal["clothing", "armor", "weapon", "gear"]
PilotGearTagType = Literal[
    "sidearm",
    "archaic",
    "loading",
    "ordnance",
    "inaccurate",
]


class PilotGearTag(BaseModel):
    """Structured tag for pilot gear items."""

    tag: PilotGearTagType
    value: int | None = None

    model_config = {"frozen": True}


class PilotDamageSpec(BaseModel):
    """Damage specification for pilot gear."""

    damage_type: DamageType
    flat: int = 0
    ap: bool = False

    model_config = {"frozen": True}


class PilotAreaEffect(BaseModel):
    """Area effect payload for pilot gear."""

    pattern: RangeType
    size: int = Field(..., ge=0)
    damage: PilotDamageSpec | None = None
    attack_vs: Literal["evasion", "e_defense"] | None = None

    model_config = {"frozen": True}


class PilotGrenadePayload(BaseModel):
    """Grenade option for pilot gear."""

    name: str
    range: int = Field(..., ge=0)
    area: PilotAreaEffect

    model_config = {"frozen": True}


class PilotChargePayload(BaseModel):
    """Planted explosive charge payload for pilot gear."""

    name: str
    plant_action: ActionType
    detonate_action: ActionType
    area: PilotAreaEffect

    model_config = {"frozen": True}


class PilotFlightEffect(BaseModel):
    """Flight behavior granted by pilot gear."""

    mode: Literal["move", "boost", "move_or_boost"]
    must_end_on_surface: bool = False

    model_config = {"frozen": True}


class PilotMedicalEffect(BaseModel):
    """Medical gear payload."""

    name: str
    action: ActionType
    heal_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    heal_round_up: bool = True
    can_heal_down_and_out: bool = False
    restores_consciousness: bool = False
    applies_to_adjacent: bool = True
    affects_mechs: bool = False

    model_config = {"frozen": True}


class PilotStimEffect(BaseModel):
    """Stim gear payload."""

    name: str
    effect: Literal["awake_alert", "calm_emotional", "heightened_senses"]
    duration_hours: int | None = Field(default=None, ge=0)

    model_config = {"frozen": True}


class PilotGearItemDefinition(BaseModel):
    """Definition for a pilot gear item."""

    id: str = Field(..., description="Unique gear identifier")
    name: str = Field(..., description="Display name")
    category: PilotGearCategory
    limited_uses: int | None = Field(default=None, ge=0)
    tags: list[PilotGearTag] = Field(default_factory=list)
    grenades: list[PilotGrenadePayload] = Field(default_factory=list)
    charges: list[PilotChargePayload] = Field(default_factory=list)
    flight: PilotFlightEffect | None = None
    medical: PilotMedicalEffect | None = None
    stim: PilotStimEffect | None = None
    effects: MechanicalEffect = Field(default_factory=MechanicalEffect)

    model_config = {"frozen": True}


class PilotGearRules(BaseModel):
    """Loadout limits for pilot gear."""

    clothing_required: bool = True
    armor_optional: bool = True
    max_weapons: int = 2
    max_gear: int = 3

    model_config = {"frozen": True}


DEFAULT_PILOT_GEAR_RULES = PilotGearRules()


class PilotLoadout(BaseModel):
    """Pilot gear selection for a mission."""

    clothing: str | None = Field(default=None, description="Clothing item ID")
    armor: str | None = Field(default=None, description="Armor item ID")
    weapons: list[str] = Field(default_factory=list, max_length=2, description="Weapon item IDs")
    gear: list[str] = Field(default_factory=list, max_length=3, description="Other gear item IDs")

    model_config = {"frozen": True}

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
        effects=MechanicalEffect(
            stat_mods=[
                StatModifier(stat="evasion", value=-2),
                StatModifier(stat="e_defense", value=-2),
            ],
            special="quick_action_invisible_breaks_on_damage",
        ),
    ),
    PilotGearItemDefinition(
        id="archaic_melee",
        name="Archaic Melee Weapon",
        category="weapon",
        tags=[PilotGearTag(tag="archaic")],
        effects=MechanicalEffect(
            special="pilot_melee_threat1_damage1_kinetic",
        ),
    ),
    PilotGearItemDefinition(
        id="alloy_composite_light",
        name="Alloy/Composite Weapon (Light)",
        category="weapon",
        effects=MechanicalEffect(
            special="pilot_melee_threat1_damage1_kinetic",
        ),
    ),
    PilotGearItemDefinition(
        id="alloy_composite_combat",
        name="Alloy/Composite Weapon (Combat)",
        category="weapon",
        effects=MechanicalEffect(
            special="pilot_melee_threat1_damage2_kinetic",
        ),
    ),
    PilotGearItemDefinition(
        id="alloy_composite_heavy",
        name="Alloy/Composite Weapon (Heavy)",
        category="weapon",
        tags=[PilotGearTag(tag="inaccurate")],
        effects=MechanicalEffect(
            special="pilot_melee_threat1_damage3_kinetic",
        ),
    ),
    PilotGearItemDefinition(
        id="archaic_ranged",
        name="Archaic Ranged Weapon",
        category="weapon",
        tags=[PilotGearTag(tag="archaic")],
        effects=MechanicalEffect(
            special="pilot_ranged_range5_damage1_kinetic",
        ),
    ),
    PilotGearItemDefinition(
        id="signature_weapon_sidearm",
        name="Signature Weapon (Sidearm)",
        category="weapon",
        tags=[PilotGearTag(tag="sidearm")],
        effects=MechanicalEffect(
            special="pilot_ranged_range3_damage1_choose_type",
        ),
    ),
    PilotGearItemDefinition(
        id="signature_weapon_combat",
        name="Signature Weapon (Combat)",
        category="weapon",
        effects=MechanicalEffect(
            special="pilot_ranged_range5_damage2_choose_type",
        ),
    ),
    PilotGearItemDefinition(
        id="signature_weapon_heavy",
        name="Signature Weapon (Heavy)",
        category="weapon",
        tags=[PilotGearTag(tag="loading"), PilotGearTag(tag="ordnance")],
        effects=MechanicalEffect(
            special="pilot_ranged_range10_damage4_choose_type",
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
        effects=MechanicalEffect(
            special="mark_surface_transmit_simple_data_unlimited",
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
        effects=MechanicalEffect(
            special="no_food_or_water_week_after_use",
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
        effects=MechanicalEffect(
            special="hack_without_external_gear",
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
        effects=MechanicalEffect(
            special="zero_g_maneuvering_bonus",
        ),
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
        effects=MechanicalEffect(
            special="non_combat_drone_relay_audio_visual",
        ),
    ),
    PilotGearItemDefinition(
        id="prosocollar",
        name="Prosocollar",
        category="gear",
        effects=MechanicalEffect(
            special="holographic_face_projection_voice_masking",
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
        effects=MechanicalEffect(
            special="full_action_enter_immunity_burn_vacuum_air1hr_evasion5_slowed_no_actions_except_exit",
        ),
    ),
    PilotGearItemDefinition(
        id="ssc_sylph",
        name="SSC Sylph",
        category="gear",
        effects=MechanicalEffect(
            special="undersuit_environmental_seal_breathe_water_limited_time",
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
        effects=MechanicalEffect(
            special="powered_third_arm_tool_or_weapon_mount",
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


class PilotGearIssue(BaseModel):
    """A pilot gear validation issue."""

    code: str
    message: str
    severity: Literal["error", "warning"] = "error"

    model_config = {"frozen": True}


class PilotGearValidation(BaseModel):
    """Validation result for pilot gear loadouts."""

    valid: bool
    issues: list[PilotGearIssue] = Field(default_factory=list)

    model_config = {"frozen": True}


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
