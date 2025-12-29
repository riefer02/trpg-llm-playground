"""Combat state models for mech combat."""

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel

from core.shared.enums import StatusType, SizeClass, ActionType, AttackType
from core.shared.rolls import ContestedCheck
from core.mech.grid import HexPosition, HexCoord
from core.mech.terrain import TerrainMap
from core.mech.weapon import WeaponTagType
from core.mech.mounts import MountSlotType
from core.mech.combat_rules import AttackPatternDefinition


CombatSide = Literal["players", "hostiles", "neutral"]
CombatantKind = Literal["mech", "pilot", "npc", "object"]
CombatEnvironment = Literal["standard", "zero_g", "underwater"]


class CombatStats(FrozenModel):
    """Combat-relevant stats for a combatant."""

    size: SizeClass
    hp_max: int = Field(..., ge=0)
    evasion: int = Field(..., ge=0)
    e_defense: int = Field(..., ge=0)
    armor: int = Field(default=0, ge=0)
    speed: int = Field(default=0, ge=0)
    sensor_range: int = Field(default=0, ge=0)
    tech_attack: int = Field(default=0)



class CombatResources(FrozenModel):
    """Resource tracks for a combatant."""

    hp_current: int = Field(..., ge=0)
    heat_current: int = Field(default=0, ge=0)
    heat_cap: int = Field(default=0, ge=0)
    structure_current: int = Field(default=0, ge=0)
    stress_current: int = Field(default=0, ge=0)
    repairs_remaining: int = Field(default=0, ge=0)



class WeaponState(FrozenModel):
    """Weapon state for a mounted weapon."""

    weapon_id: str
    tags: list[WeaponTagType] = Field(default_factory=list)
    destroyed: bool = False
    limited_charges_remaining: int | None = Field(default=None, ge=0)



class WeaponMountState(FrozenModel):
    """Mount slot state and installed weapons."""

    mount_index: int = Field(..., ge=0)
    slot_type: MountSlotType | None = None
    weapons: list[WeaponState] = Field(default_factory=list)
    destroyed: bool = False



class MechSystemState(FrozenModel):
    """System state for a mech."""

    system_id: str
    destroyed: bool = False
    limited_charges_remaining: int | None = Field(default=None, ge=0)



class MechInventory(FrozenModel):
    """Inventory state for mounts and systems."""

    mounts: list[WeaponMountState] = Field(default_factory=list)
    systems: list[MechSystemState] = Field(default_factory=list)



class CombatantState(FrozenModel):
    """State for a combatant in mech combat."""

    id: str
    name: str
    side: CombatSide
    kind: CombatantKind
    stats: CombatStats
    resources: CombatResources
    position: HexPosition | None = None
    statuses: list[StatusType] = Field(default_factory=list)
    conditions: list[StatusType] = Field(default_factory=list)
    inventory: MechInventory | None = None
    ai_controlled: bool = False



class GrappleLink(FrozenModel):
    """Link between grappling combatants."""

    grappler_id: str
    target_id: str
    grappler_total_size: int = Field(default=1, ge=0)
    target_total_size: int = Field(default=1, ge=0)



class ActionUse(FrozenModel):
    """An action taken during a combat turn."""

    action_id: str
    action_type: ActionType
    target_id: str | None = None
    target_position: HexPosition | None = None
    target_ids: list[str] = Field(default_factory=list)
    target_positions: list[HexPosition] = Field(default_factory=list)
    range_spaces: int | None = Field(default=None, ge=0)
    attack_type_override: AttackType | None = None
    weapon_tags: list[WeaponTagType] = Field(default_factory=list)
    area_pattern: AttackPatternDefinition | None = None
    area_origin: HexPosition | None = None
    area_direction: HexCoord | None = None
    area_affected: list[HexCoord] = Field(default_factory=list)
    weapon_count: int | None = Field(default=None, ge=0)
    uses_superheavy: bool | None = None
    uses_aux_bonus_attack: bool | None = None
    stabilize_primary: Literal["cool_heat", "spend_repair_full_hp"] | None = None
    stabilize_secondary: Literal["reload_loading", "clear_burn", "clear_condition"] | None = None
    ignores_line_of_sight: bool = False
    ignores_cover: bool = False
    used_as_free_action: bool = False
    used_as_reaction: bool = False
    granted_by_overcharge: bool = False
    contested_check: ContestedCheck | None = None
    consumes_lock_on: bool = False



class CombatTurn(FrozenModel):
    """A single combat turn."""

    actor_id: str
    move_used: bool = False
    movement_mode: Literal["ground", "flight", "hover", "teleport"] = "ground"
    movement_path: list[HexPosition] = Field(default_factory=list)
    actions: list[ActionUse] = Field(default_factory=list)



class CombatRound(FrozenModel):
    """A combat round."""

    round_index: int = Field(..., ge=1)
    turns: list[CombatTurn] = Field(default_factory=list)



class MechCombatScenario(FrozenModel):
    """Full combat scenario for evaluation."""

    combatants: list[CombatantState] = Field(default_factory=list)
    grapples: list[GrappleLink] = Field(default_factory=list)
    rounds: list[CombatRound] = Field(default_factory=list)
    terrain: TerrainMap | None = None
    environment: CombatEnvironment = "standard"

