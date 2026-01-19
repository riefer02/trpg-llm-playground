"""Combat execution helper functions.

Private implementation helpers for combat resolution.
These are internal functions used by combat_execution.py.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from core.shared.enums import StatusType
from core.shared.dice import roll_dice
from core.shared.state_helpers import add_statuses
from core.mech.statuses import (
    StatusInstance,
    StatusClearTrigger,
    get_status_default_duration,
    get_status_clear_triggers,
)
from core.shared.full_tech import (
    FullTechFirstOption,
    FullTechSecondOption,
    FullTechOptionSelection,
    ScanTechParams,
    BolsterTechParams,
    LockOnTechParams,
    InvadeTechParams,
)
from core.mech.grid import HexCoord, HexPosition, hex_add, hex_scale
from core.mech.compendium import get_weapon_definition
from core.mech.combat_rules import AttackPatternDefinition
from core.mech.weapon import WeaponProfile, WeaponTag, resolve_weapon_profile
from core.mech.tech_actions import (
    ScanResult,
    BolsterResult,
    LockOnResult,
    InvadeResult,
)
from core.mech.combat_state import (
    MechCombatScenario,
    CombatantState,
    GrappleLink,
    WeaponState,
    MechSystemState,
)
from core.shared.involuntary_movement import resolve_knockback
from core.shared.los import LOSCheckRequest, check_line_of_sight
from core.shared.movement import check_engagement_stop, _is_hostile

if TYPE_CHECKING:
    from core.mech.combat_models import ResourceChange, StabilizePrimary, StabilizeSecondary, BurnTickResult


# =============================================================================
# Weapon Resolution Helpers
# =============================================================================


def _resolve_weapon_profile(
    weapon_id: str | None,
    profile_id: str | None = None,
) -> WeaponProfile | None:
    """Resolve weapon profile from weapon ID.

    Args:
        weapon_id: Weapon ID to look up
        profile_id: Optional profile ID for weapons with multiple profiles

    Returns:
        WeaponProfile or None if weapon not found
    """
    if weapon_id is None:
        return None
    weapon_def = get_weapon_definition(weapon_id)
    if weapon_def is None:
        return None
    return resolve_weapon_profile(weapon_def, profile_id)


def _extract_tag_value(tags: list[WeaponTag], tag_name: str) -> int | None:
    """Extract maximum value for a tag from weapon tags."""
    values = [tag.value for tag in tags if tag.tag == tag_name and tag.value is not None]
    if not values:
        return None
    return max(values)


def _extract_area_pattern(profile: WeaponProfile) -> AttackPatternDefinition | None:
    """Extract area pattern from weapon profile."""
    for range_entry in profile.ranges:
        if range_entry.range_type in ("line", "cone", "blast", "burst"):
            return AttackPatternDefinition(
                pattern=range_entry.range_type,
                size=range_entry.value,
            )
    for tag in profile.tags:
        if tag.tag in {"line", "cone", "blast", "burst"} and tag.value is not None:
            return AttackPatternDefinition(pattern=tag.tag, size=tag.value)
    return None


def _get_weapon_range(
    weapon_id: str | None,
    is_melee: bool = False,
    profile_id: str | None = None,
) -> int:
    """Get effective range for a weapon.

    Returns the weapon's range (for ranged) or threat (for melee).
    Defaults: ranged=10, threat=1 if weapon not found.

    Args:
        weapon_id: Weapon ID to look up, or None for default
        is_melee: Whether to look for threat (melee) or range (ranged)
        profile_id: Optional profile ID for weapons with multiple profiles

    Returns:
        The effective range/threat value
    """
    default_range = 1 if is_melee else 10

    if weapon_id is None:
        return default_range

    weapon_def = get_weapon_definition(weapon_id)
    if weapon_def is None:
        return default_range

    profile = resolve_weapon_profile(weapon_def, profile_id)

    # Look for the appropriate range type in ranges
    for range_entry in profile.ranges:
        if is_melee and range_entry.range_type == "threat":
            return range_entry.value
        elif not is_melee and range_entry.range_type == "range":
            return range_entry.value

    return default_range


def _get_thrown_range(
    weapon_id: str | None,
    profile_id: str | None = None,
) -> int | None:
    """Get thrown range for a weapon (if any).

    Thrown ranges may be represented as:
    - Range entries with range_type="thrown"
    - Weapon tags with tag="thrown" and a numeric value

    Args:
        weapon_id: Weapon ID to look up
        profile_id: Optional profile ID for weapons with multiple profiles
    """
    if weapon_id is None:
        return None

    weapon_def = get_weapon_definition(weapon_id)
    if weapon_def is None:
        return None

    profile = resolve_weapon_profile(weapon_def, profile_id)

    thrown_values = [
        range_entry.value
        for range_entry in profile.ranges
        if range_entry.range_type == "thrown"
    ]
    thrown_values.extend(
        tag.value
        for tag in profile.tags
        if tag.tag == "thrown" and tag.value is not None
    )

    return max(thrown_values) if thrown_values else None


def _has_weapon_tag(
    weapon_id: str | None,
    tag_name: str,
    profile_id: str | None = None,
) -> bool:
    """Check if a weapon has a specific tag.

    Args:
        weapon_id: Weapon ID to look up
        tag_name: Tag name to search for
        profile_id: Optional profile ID for weapons with multiple profiles

    Returns:
        True if weapon has the tag, False otherwise
    """
    if weapon_id is None:
        return False

    weapon_def = get_weapon_definition(weapon_id)
    if weapon_def is None:
        return False

    profile = resolve_weapon_profile(weapon_def, profile_id)
    return any(tag.tag == tag_name for tag in profile.tags)


def _is_melee_weapon(
    weapon_id: str | None,
    profile_id: str | None = None,
) -> bool:
    """Check if a weapon is melee (has threat range).

    Args:
        weapon_id: Weapon ID to look up
        profile_id: Optional profile ID for weapons with multiple profiles

    Returns:
        True if weapon is melee, False otherwise
    """
    if weapon_id is None:
        return False

    weapon_def = get_weapon_definition(weapon_id)
    if weapon_def is None:
        return False

    profile = resolve_weapon_profile(weapon_def, profile_id)

    for range_entry in profile.ranges:
        if range_entry.range_type == "threat":
            return True

    return False


def _get_weapon_state(
    actor: CombatantState,
    weapon_id: str,
) -> WeaponState | None:
    """Find weapon state in actor's inventory by weapon ID.

    Args:
        actor: The combatant to search
        weapon_id: Weapon ID to find

    Returns:
        WeaponState if found, None otherwise
    """
    if not actor.inventory:
        return None
    for mount in actor.inventory.mounts:
        for weapon in mount.weapons:
            if weapon.weapon_id == weapon_id:
                return weapon
    return None


def _validate_weapon_usable(
    weapon_state: WeaponState | None,
    weapon_id: str | None,
    actor: CombatantState,
    has_moved_or_acted: bool,
) -> tuple[bool, str | None]:
    """Check if a weapon can be used (loading, limited, ordnance restrictions).

    Per PR2 rules:
    - Loading (5029-5030): Must reload before using again
    - Limited (5080-5081): Finite charges per full repair
    - Ordnance (5035-5037): Must fire before moving/acting, not while engaged
    - Thrown (5047-5049): Thrown weapons are disarmed until retrieved

    Args:
        weapon_state: The weapon's current state from inventory
        weapon_id: Weapon ID for tag lookup
        actor: The attacking combatant
        has_moved_or_acted: Whether actor has moved or taken non-protocol actions

    Returns:
        Tuple of (valid, error_message). If valid=True, error_message is None.
    """
    # If weapon_id provided but not found in inventory, fail
    # Only check if actor actually has inventory set up (allows compendium-only tests)
    if weapon_id is not None and weapon_state is None:
        if actor.inventory is not None:
            return (False, "Weapon not found in inventory")
        # If no inventory, allow compendium lookup (backward compatible for testing)

    # If no weapon specified at all, pass (action doesn't require weapon)
    if weapon_state is None and weapon_id is None:
        return (True, None)

    # Check destroyed
    if weapon_state is not None and weapon_state.destroyed:
        return (False, "Weapon is destroyed")

    # Check thrown/disarmed state
    if weapon_state is not None and weapon_state.thrown_coord is not None:
        return (False, "Weapon was thrown and must be retrieved before use")

    # Check loading - needs reload
    if weapon_state is not None and weapon_state.needs_reload:
        return (False, "Weapon needs reload (Stabilize action)")

    # Check limited - no charges remaining
    if weapon_state is not None and weapon_state.limited_charges_remaining is not None:
        if weapon_state.limited_charges_remaining <= 0:
            return (False, "Weapon has no charges remaining")

    # Check ordnance - must fire before moving/acting
    # Check both compendium tags and weapon state tags
    has_ordnance = _has_weapon_tag(weapon_id, "ordnance")
    if weapon_state is not None and "ordnance" in weapon_state.tags:
        has_ordnance = True

    if has_ordnance:
        if has_moved_or_acted:
            return (False, "Ordnance weapons must fire before other actions/movement")
        # Check engaged status - ordnance cannot target enemies while engaged
        if "engaged" in (actor.statuses or []):
            return (False, "Ordnance weapons cannot be used while engaged")

    return (True, None)


def _update_weapon_after_attack(
    actor: CombatantState,
    weapon_id: str,
    thrown_coord: HexCoord | None = None,
) -> CombatantState:
    """Update weapon state after attack (reload, limited, thrown disarm).

    Per PR2 rules:
    - Loading weapons need reload after firing
    - Limited weapons consume one charge per attack
    - Thrown attacks disarm the weapon until retrieved

    Args:
        actor: The attacking combatant
        weapon_id: ID of the weapon that was fired

    Returns:
        Updated CombatantState with modified weapon state
    """
    if not actor.inventory or not weapon_id:
        return actor

    new_mounts = []
    for mount in actor.inventory.mounts:
        new_weapons = []
        for weapon in mount.weapons:
            if weapon.weapon_id == weapon_id:
                updates: dict = {}
                # Set needs_reload for loading weapons
                if "loading" in weapon.tags:
                    updates["needs_reload"] = True
                # Decrement limited charges
                if weapon.limited_charges_remaining is not None:
                    updates["limited_charges_remaining"] = max(
                        0, weapon.limited_charges_remaining - 1
                    )
                if thrown_coord is not None:
                    updates["thrown_coord"] = thrown_coord
                if updates:
                    weapon = weapon.model_copy(update=updates)
            new_weapons.append(weapon)
        new_mounts.append(mount.model_copy(update={"weapons": new_weapons}))

    new_inventory = actor.inventory.model_copy(update={"mounts": new_mounts})
    return actor.model_copy(update={"inventory": new_inventory})


def _reload_all_loading_weapons(
    actor: CombatantState,
) -> tuple[CombatantState, list[str]]:
    """Reset needs_reload for all loading weapons.

    Called when Stabilize action is used with reload_loading option.

    Args:
        actor: The combatant performing stabilize

    Returns:
        Tuple of (updated CombatantState, list of reloaded weapon IDs)
    """
    if not actor.inventory:
        return actor, []

    reloaded_weapons: list[str] = []
    new_mounts = []
    for mount in actor.inventory.mounts:
        new_weapons = []
        for weapon in mount.weapons:
            if "loading" in weapon.tags and weapon.needs_reload:
                weapon = weapon.model_copy(update={"needs_reload": False})
                reloaded_weapons.append(weapon.weapon_id)
            new_weapons.append(weapon)
        new_mounts.append(mount.model_copy(update={"weapons": new_weapons}))

    new_inventory = actor.inventory.model_copy(update={"mounts": new_mounts})
    return actor.model_copy(update={"inventory": new_inventory}), reloaded_weapons


def _get_system_state(
    actor: CombatantState,
    system_id: str,
) -> MechSystemState | None:
    """Find system state in actor's inventory by system ID.

    Args:
        actor: The combatant to search
        system_id: System ID to find

    Returns:
        MechSystemState if found, None otherwise
    """
    if not actor.inventory:
        return None
    for system in actor.inventory.systems:
        if system.system_id == system_id:
            return system
    return None


def _validate_system_usable(
    system_state: MechSystemState | None,
    system_id: str | None,
) -> tuple[bool, str | None]:
    """Validate system can be activated.

    Args:
        system_state: The system's current state from inventory
        system_id: System ID for reference

    Returns:
        Tuple of (valid, error_message). If valid=True, error_message is None.
    """
    # If system_id provided but not found in inventory, fail
    if system_id is not None and system_state is None:
        return (False, "System not found in inventory")

    # If no system specified at all, pass (action doesn't require system)
    if system_state is None and system_id is None:
        return (True, None)

    # Check destroyed
    if system_state is not None and system_state.destroyed:
        return (False, "System is destroyed")

    # Check limited charges
    if system_state is not None and system_state.limited_charges_remaining is not None:
        if system_state.limited_charges_remaining <= 0:
            return (False, "System has no charges remaining")

    return (True, None)


def _validate_attack_range_and_los(
    scenario: MechCombatScenario,
    attacker: CombatantState,
    target: CombatantState,
    weapon_id: str | None,
    is_tech_attack: bool = False,
    use_thrown: bool = False,
) -> tuple[bool, str | None]:
    """Validate range and LOS for an attack.

    Returns (valid, error_message). If valid=True, error_message is None.

    Per PR2 pp 99-100:
    - Ranged weapons: target must be within weapon range
    - Melee weapons: target must be within threat range
    - Thrown melee weapons: if use_thrown=True, target must be within thrown range
    - Tech attacks: target must be within sensor range
    - LOS must not be blocked (except: seeking ignores LOS, arcing ignores LOS but not cover)

    Args:
        scenario: Current combat scenario
        attacker: The attacking combatant
        target: The target combatant
        weapon_id: Weapon ID being used (None for tech attacks)
        is_tech_attack: Whether this is a tech attack (uses sensor range)

    Returns:
        Tuple of (valid, error_message)
    """
    # Check positions exist
    if attacker.position is None:
        return (False, "Attacker has no position")
    if target.position is None:
        return (False, "Target has no position")

    # Calculate hex distance
    distance = attacker.position.coord.distance_to(target.position.coord)

    # Determine required range
    if is_tech_attack:
        # Tech attacks use sensor range
        required_range = attacker.stats.sensor_range if attacker.stats else 10
        range_type = "sensor"
    else:
        # Check if melee or ranged
        is_melee = _is_melee_weapon(weapon_id)
        if use_thrown and not is_melee:
            return (False, "Only melee weapons can be thrown")

        if is_melee:
            threat_range = _get_weapon_range(weapon_id, is_melee=True)
            if use_thrown:
                thrown_range = _get_thrown_range(weapon_id)
                if thrown_range is None:
                    return (False, "Weapon has no thrown range")
                required_range = thrown_range
                range_type = "thrown"
            else:
                required_range = threat_range
                range_type = "threat"
        else:
            required_range = _get_weapon_range(weapon_id, is_melee=False)
            range_type = "range"

    # Validate range
    if distance > required_range:
        return (False, f"Target out of range ({distance} > {required_range} {range_type})")

    # Check for seeking/arcing tags that bypass LOS
    has_seeking = _has_weapon_tag(weapon_id, "seeking")
    has_arcing = _has_weapon_tag(weapon_id, "arcing")

    # Check LOS
    los_request = LOSCheckRequest(
        attacker_pos=attacker.position,
        target_pos=target.position,
        terrain=scenario.terrain,
    )
    los_result = check_line_of_sight(los_request)

    if los_result.los_type == "blocked":
        # Seeking weapons ignore LOS entirely
        if has_seeking:
            pass  # Valid despite blocked LOS
        # Arcing weapons ignore LOS (but cover still applies, handled elsewhere)
        elif has_arcing:
            pass  # Valid despite blocked LOS
        else:
            return (False, "No line of sight to target")

    return (True, None)


def _roll_damage_with_overkill(
    damage_expr,
    apply_overkill: bool,
) -> tuple[list[int], int]:
    """Roll damage dice with overkill reroll mechanic."""
    rolls: list[int] = []
    overkill_heat = 0
    for _ in range(damage_expr.count):
        roll = random.randint(1, damage_expr.size)
        if apply_overkill:
            while roll == 1:
                overkill_heat += 1
                roll = random.randint(1, damage_expr.size)
        rolls.append(roll)
    return rolls, overkill_heat


def _roll_weapon_damage(
    profile: WeaponProfile | None,
    apply_overkill: bool,
) -> tuple[int, int]:
    """Roll weapon damage and return (total_damage, overkill_heat)."""
    if profile is None:
        return 6, 0

    total_damage = 0
    overkill_heat = 0
    for damage_component in profile.damage:
        component_total = 0
        if damage_component.dice is not None:
            rolls, component_overkill = _roll_damage_with_overkill(
                damage_component.dice,
                apply_overkill,
            )
            component_total += sum(rolls) + damage_component.dice.modifier
            overkill_heat += component_overkill
        component_total += damage_component.flat
        total_damage += component_total

    if total_damage == 0:
        total_damage = 6

    return total_damage, overkill_heat


# =============================================================================
# Tech Action Helpers
# =============================================================================


def _build_full_tech_option(
    selection: FullTechOptionSelection,
    actor: CombatantState,
    target: CombatantState,
    option_cls: type[FullTechFirstOption] | type[FullTechSecondOption],
) -> FullTechFirstOption | FullTechSecondOption:
    """Build a Full Tech option payload using actor/target stats."""
    attacker_systems = actor.stats.tech_attack if actor.stats else 0
    target_e_defense = target.stats.e_defense if target.stats else 10

    if selection.option == "scan":
        if selection.scan_options is None:
            scan_params = ScanTechParams(target_id=selection.target_id)
        else:
            scan_params = ScanTechParams(
                target_id=selection.target_id,
                scan_options=selection.scan_options,
            )
        return option_cls(option="scan", scan_params=scan_params)
    if selection.option == "bolster":
        bolster_params = BolsterTechParams(
            target_id=selection.target_id,
            attacker_systems=attacker_systems,
        )
        return option_cls(option="bolster", bolster_params=bolster_params)
    if selection.option == "lock_on":
        lock_on_params = LockOnTechParams(target_id=selection.target_id)
        return option_cls(option="lock_on", lock_on_params=lock_on_params)
    if selection.option == "invade":
        invade_params = InvadeTechParams(
            target_id=selection.target_id,
            attacker_systems=attacker_systems,
            target_e_defense=target_e_defense,
        )
        return option_cls(option="invade", invade_params=invade_params)

    return option_cls(option=selection.option)


def _apply_tech_result(
    scenario: MechCombatScenario,
    result: ScanResult | BolsterResult | LockOnResult | InvadeResult,
    effects_applied: list[dict],
    resource_changes: list["ResourceChange"],
    statuses_applied: dict[str, list[StatusType]],
    overheat_checks: list[dict],
    apply_heat_func,
) -> tuple[MechCombatScenario, int]:
    """Apply a tech action result to the scenario and collect effects."""
    heat_generated = 0

    if isinstance(result, ScanResult):
        effects_applied.append({
            "type": "scan",
            "target_id": result.target_id,
            "revealed": result.revealed_info,
        })
        return scenario, heat_generated

    if isinstance(result, BolsterResult):
        effects_applied.append({
            "type": "bolster",
            "target_id": result.target_id,
            "rolls": result.systems_roll.rolls if result.systems_roll else [],
            "total": result.check_total,
            "accuracy_bonus": result.accuracy_bonus,
            "duration": result.duration,
        })
        return scenario, heat_generated

    if isinstance(result, LockOnResult):
        scenario, added_statuses = _apply_statuses_to_target(
            scenario, result.target_id, [result.status_granted]
        )
        _record_statuses_applied(statuses_applied, result.target_id, added_statuses)
        effects_applied.append({
            "type": "lock_on",
            "target_id": result.target_id,
            "status": result.status_granted,
            "accuracy_bonus": result.accuracy_bonus,
            "duration": result.duration,
        })
        return scenario, heat_generated

    if isinstance(result, InvadeResult):
        effects_applied.append({
            "type": "invade",
            "target_id": result.target_id,
            "rolls": result.systems_roll.rolls if result.systems_roll else [],
            "total": result.check_total,
            "target_e_defense": result.target_e_defense,
            "hit": result.hit,
        })

        if result.hit and result.heat_applied:
            scenario, change, overheat_result = apply_heat_func(
                scenario, result.target_id, result.heat_applied
            )
            resource_changes.append(change)
            heat_generated += result.heat_applied

            if overheat_result:
                overheat_checks.append({
                    "type": "overheat_check",
                    "target_id": result.target_id,
                    "outcome": overheat_result.outcome,
                    "statuses": [str(s) for s in overheat_result.statuses_to_apply],
                    "dice_rolls": overheat_result.dice_rolls,
                    "lowest_roll": overheat_result.lowest_roll,
                    "meltdown_state": overheat_result.meltdown_state is not None,
                })

            scenario, added_statuses = _apply_statuses_to_target(
                scenario, result.target_id, result.conditions_applied
            )
            _record_statuses_applied(statuses_applied, result.target_id, added_statuses)

        return scenario, heat_generated

    return scenario, heat_generated


# =============================================================================
# Status Helpers
# =============================================================================


def _record_statuses_applied(
    statuses_applied: dict[str, list[StatusType]],
    target_id: str,
    statuses: list[StatusType],
) -> None:
    """Record applied statuses for response output."""
    if not statuses:
        return
    existing = statuses_applied.setdefault(target_id, [])
    for status in statuses:
        if status not in existing:
            existing.append(status)


def _apply_statuses_to_target(
    scenario: MechCombatScenario,
    target_id: str,
    statuses: list[StatusType],
) -> tuple[MechCombatScenario, list[StatusType]]:
    """Apply statuses to a target combatant and return added statuses."""
    if not statuses:
        return scenario, []

    target: CombatantState | None = None
    target_idx: int = -1
    for i, c in enumerate(scenario.combatants):
        if c.id == target_id:
            target = c
            target_idx = i
            break

    if target is None:
        return scenario, []

    existing = set(target.statuses)
    new_statuses = [status for status in statuses if status not in existing]
    if not new_statuses:
        return scenario, []

    updated_target = add_statuses(target, new_statuses)
    updated_combatants = list(scenario.combatants)
    updated_combatants[target_idx] = updated_target

    updated_scenario = MechCombatScenario(
        combatants=updated_combatants,
        grapples=list(scenario.grapples),
        rounds=list(scenario.rounds),
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables=dict(scenario.deployables),
    )

    return updated_scenario, new_statuses


def _remove_status_from_target(
    scenario: MechCombatScenario,
    target_id: str,
    status: StatusType,
) -> MechCombatScenario:
    """Remove a status from a target combatant."""
    target_idx = next((i for i, c in enumerate(scenario.combatants) if c.id == target_id), -1)
    if target_idx < 0:
        return scenario

    target = scenario.combatants[target_idx]
    if status not in target.statuses:
        return scenario

    new_statuses = [s for s in target.statuses if s != status]
    updated_target = target.model_copy(update={"statuses": new_statuses})

    updated_combatants = list(scenario.combatants)
    updated_combatants[target_idx] = updated_target

    return MechCombatScenario(
        combatants=updated_combatants,
        grapples=list(scenario.grapples),
        rounds=list(scenario.rounds),
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables=dict(scenario.deployables),
    )


def _get_basic_available_actions(actor: CombatantState) -> list[str]:
    """Get basic list of available action IDs for an actor.

    This is a simplified version - a full implementation would check
    weapons, systems, statuses, cooldowns, etc.
    """
    actions = [
        # Full actions
        "barrage",
        "full_tech",
        "improvised_attack",
        "stabilize",
        "disengage",
        # Quick actions
        "skirmish",
        "boost",
        "ram",
        "grapple",
        "scan",
        "bolster",
        "lock_on",
        "invade",
        "hide",
        "search",
        "activate",
        # Free actions
        "overcharge",
        "mount_dismount",
        # Reactions
        "brace",
        "overwatch",
    ]

    # Filter based on status conditions
    if "stunned" in actor.statuses:
        # Stunned: cannot take actions except mount/dismount/eject
        actions = ["mount_dismount"]
    elif "jammed" in actor.statuses:
        # Jammed: only improvised attacks and grapples
        actions = [a for a in actions if a in ["improvised_attack", "grapple", "mount_dismount", "brace"]]

    return actions


# =============================================================================
# Status Duration & Clearing Helpers
# =============================================================================


def _apply_status_with_duration(
    scenario: MechCombatScenario,
    target_id: str,
    status: StatusType,
    current_round: int,
    applied_by: str | None = None,
) -> tuple[MechCombatScenario, bool]:
    """Apply a status to a target with duration tracking.

    Creates a StatusInstance for the status with appropriate duration metadata.
    Also updates the legacy statuses list for backwards compatibility.

    Args:
        scenario: Current combat scenario
        target_id: ID of target combatant
        status: Status type to apply
        current_round: Current round number for tracking duration
        applied_by: ID of combatant who applied this status (optional)

    Returns:
        Tuple of (updated scenario, whether status was newly applied)
    """
    target_idx = next(
        (i for i, c in enumerate(scenario.combatants) if c.id == target_id), -1
    )
    if target_idx < 0:
        return scenario, False

    target = scenario.combatants[target_idx]

    # Check if status already exists
    if status in target.statuses:
        return scenario, False

    # Determine duration type from status definition
    duration_type = get_status_default_duration(status)

    # Create status instance
    new_instance = StatusInstance(
        status=status,
        applied_on_round=current_round,
        applied_by=applied_by,
        duration_type=duration_type,
    )

    # Update both status_instances and statuses lists
    new_instances = list(target.status_instances) + [new_instance]
    new_statuses = list(target.statuses) + [status]

    updated_target = target.model_copy(
        update={"status_instances": new_instances, "statuses": new_statuses}
    )

    updated_combatants = list(scenario.combatants)
    updated_combatants[target_idx] = updated_target

    updated_scenario = MechCombatScenario(
        combatants=updated_combatants,
        grapples=list(scenario.grapples),
        rounds=list(scenario.rounds),
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables=dict(scenario.deployables),
    )

    return updated_scenario, True


def _clear_statuses_by_trigger(
    scenario: MechCombatScenario,
    actor_id: str,
    trigger: StatusClearTrigger,
) -> tuple[MechCombatScenario, list[StatusType]]:
    """Clear all statuses on actor matching the given trigger.

    Per PR2 rules, certain actions trigger automatic status clearing:
    - attack: Clears hidden
    - boost: Clears hidden
    - reaction: Clears hidden
    - boot_up: Clears shutdown
    - stand_up: Clears prone

    Args:
        scenario: Current combat scenario
        actor_id: ID of the combatant whose statuses to check
        trigger: The trigger event that occurred

    Returns:
        Tuple of (updated scenario, list of statuses that were cleared)
    """
    actor_idx = next(
        (i for i, c in enumerate(scenario.combatants) if c.id == actor_id), -1
    )
    if actor_idx < 0:
        return scenario, []

    actor = scenario.combatants[actor_idx]
    cleared_statuses: list[StatusType] = []

    # Find statuses to clear based on trigger
    for status in actor.statuses:
        triggers = get_status_clear_triggers(status)
        if trigger in triggers:
            cleared_statuses.append(status)

    if not cleared_statuses:
        return scenario, []

    # Remove cleared statuses from both lists
    new_statuses = [s for s in actor.statuses if s not in cleared_statuses]
    new_instances = [
        inst for inst in actor.status_instances if inst.status not in cleared_statuses
    ]

    updated_actor = actor.model_copy(
        update={"statuses": new_statuses, "status_instances": new_instances}
    )

    updated_combatants = list(scenario.combatants)
    updated_combatants[actor_idx] = updated_actor

    updated_scenario = MechCombatScenario(
        combatants=updated_combatants,
        grapples=list(scenario.grapples),
        rounds=list(scenario.rounds),
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables=dict(scenario.deployables),
    )

    return updated_scenario, cleared_statuses


def _expire_turn_duration_statuses(
    scenario: MechCombatScenario,
    actor_id: str,
    current_round: int,
) -> tuple[MechCombatScenario, list[dict]]:
    """Expire duration-based statuses at end of turn.

    Handles:
    - end_of_turn: Expires immediately at end of current turn
    - end_of_next_turn: Expires at end of the round AFTER they were applied

    Args:
        scenario: Current combat scenario
        actor_id: ID of the combatant whose turn is ending
        current_round: Current round number

    Returns:
        Tuple of (updated scenario, list of expiration effect dicts)
    """
    actor_idx = next(
        (i for i, c in enumerate(scenario.combatants) if c.id == actor_id), -1
    )
    if actor_idx < 0:
        return scenario, []

    actor = scenario.combatants[actor_idx]
    expired_effects: list[dict] = []
    expired_statuses: list[StatusType] = []

    for instance in actor.status_instances:
        should_expire = False

        if instance.duration_type == "end_of_turn":
            # Always expires at end of turn
            should_expire = True
        elif instance.duration_type == "end_of_next_turn":
            # Expires at end of the round AFTER it was applied
            # e.g., applied in round 1 → expires at end of round 2
            if current_round > instance.applied_on_round:
                should_expire = True

        if should_expire:
            expired_statuses.append(instance.status)
            expired_effects.append({
                "type": "status_expired",
                "status": instance.status,
                "duration_type": instance.duration_type,
                "applied_on_round": instance.applied_on_round,
                "expired_on_round": current_round,
            })

    if not expired_statuses:
        return scenario, []

    # Remove expired statuses from both lists
    new_statuses = [s for s in actor.statuses if s not in expired_statuses]
    new_instances = [
        inst for inst in actor.status_instances if inst.status not in expired_statuses
    ]

    updated_actor = actor.model_copy(
        update={"statuses": new_statuses, "status_instances": new_instances}
    )

    updated_combatants = list(scenario.combatants)
    updated_combatants[actor_idx] = updated_actor

    updated_scenario = MechCombatScenario(
        combatants=updated_combatants,
        grapples=list(scenario.grapples),
        rounds=list(scenario.rounds),
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables=dict(scenario.deployables),
    )

    return updated_scenario, expired_effects


def _sync_statuses_from_instances(combatant: CombatantState) -> CombatantState:
    """Rebuild the statuses list from status_instances.

    Ensures backwards compatibility by keeping the statuses list
    in sync with the status_instances list.

    Args:
        combatant: Combatant whose statuses to sync

    Returns:
        Updated combatant with synced statuses list
    """
    instance_statuses = [inst.status for inst in combatant.status_instances]

    # Keep any statuses that are in the legacy list but not tracked
    # This handles statuses applied before the instance tracking system
    legacy_only = [s for s in combatant.statuses if s not in instance_statuses]

    synced_statuses = instance_statuses + legacy_only

    return combatant.model_copy(update={"statuses": synced_statuses})


# =============================================================================
# Attack Modifiers
# =============================================================================


def _get_attacker_status_modifiers(actor: CombatantState) -> tuple[int, int]:
    """Get accuracy and difficulty modifiers from attacker's statuses.

    Per PR2 Status Effects:
    - Impaired: +1 difficulty to all attacks, saves, and skill checks

    Returns:
        Tuple of (accuracy_mod, difficulty_mod)
    """
    accuracy_mod = 0
    difficulty_mod = 0

    if "impaired" in actor.statuses:
        difficulty_mod += 1  # All attacks harder

    return accuracy_mod, difficulty_mod


def _get_target_status_modifiers(
    target: CombatantState,
    is_ranged: bool,
) -> tuple[int, int, bool]:
    """Get accuracy and difficulty modifiers from target's statuses.

    Per PR2 Status Effects:
    - Prone: +1 accuracy (easier to hit)
    - Braced: -1 accuracy (harder to hit, attacker's perspective)
    - Lock On: +1 consumable accuracy (consumed on successful hit)
    - Engaged: +1 difficulty for ranged attacks

    Returns:
        Tuple of (accuracy_mod, difficulty_mod, has_lock_on)
    """
    accuracy_mod = 0
    difficulty_mod = 0
    has_lock_on = False

    if "prone" in target.statuses:
        accuracy_mod += 1  # Easier to hit

    if "braced" in target.statuses:
        accuracy_mod -= 1  # Harder to hit

    if "lock_on" in target.statuses:
        accuracy_mod += 1
        has_lock_on = True

    if "engaged" in target.statuses and is_ranged:
        difficulty_mod += 1  # Ranged attacks harder when target engaged

    return accuracy_mod, difficulty_mod, has_lock_on


def _check_invisibility_miss(target: CombatantState) -> bool:
    """Check if invisible target causes attack to miss.

    Per PR2: Invisible targets have 50% miss chance.

    Returns:
        True if invisibility causes miss, False otherwise
    """
    if "invisible" not in target.statuses:
        return False

    return random.random() < 0.5


def _get_cover_modifier(
    scenario: MechCombatScenario,
    attacker: CombatantState,
    target: CombatantState,
) -> tuple[int, dict | None]:
    """Get cover difficulty modifier for ranged attack.

    Per PR2:
    - Soft Cover: +1 Difficulty
    - Hard Cover: +2 Difficulty (requires adjacency)
    - Flanking negates hard cover

    Returns:
        Tuple of (difficulty_modifier, effect_dict_or_none)
    """
    if scenario.terrain is None:
        return 0, None
    if attacker.position is None or target.position is None:
        return 0, None

    from core.shared.terrain import get_cover_difficulty
    from core.shared.enums import SizeClass

    target_size: SizeClass = target.stats.size if target.stats else "size_1"
    result = get_cover_difficulty(
        terrain=scenario.terrain,
        attacker_coord=attacker.position.coord,
        target_coord=target.position.coord,
        target_size=target_size,
    )

    if result.difficulty_modifier == 0:
        return 0, None

    return result.difficulty_modifier, {
        "type": "cover_modifier",
        "cover_type": result.cover_type,
        "difficulty_added": result.difficulty_modifier,
        "reason": result.reason,
    }


# =============================================================================
# Action Resolution Helpers (PR2 Combat Actions)
# =============================================================================


def _resolve_stabilize(
    scenario: MechCombatScenario,
    actor: CombatantState,
    primary_option: "StabilizePrimary | None",
    secondary_option: "StabilizeSecondary | None",
    condition_target_id: str | None,
) -> tuple[MechCombatScenario, list[dict], list["ResourceChange"]]:
    """Resolve Stabilize action (PR2 4275-4286).

    Stabilize is a full action with two choices:
    - Primary: Cool heat to 0 OR Spend 1 repair to refill HP to max
    - Secondary: Reload all Loading weapons OR End all Burn OR Clear one condition

    Returns:
        Tuple of (updated scenario, effects list, resource changes list)
    """
    from core.mech.combat_models import ResourceChange

    effects: list[dict] = []
    resource_changes: list[ResourceChange] = []

    # Find actor in scenario for updates
    actor_idx = next((i for i, c in enumerate(scenario.combatants) if c.id == actor.id), -1)
    if actor_idx < 0:
        return scenario, effects, resource_changes

    updated_actor = actor

    # Primary option
    if primary_option == "cool_heat":
        # Reset heat to 0, end exposed condition
        heat_cleared = updated_actor.resources.heat_current
        new_resources = updated_actor.resources.model_copy(update={"heat_current": 0})
        new_statuses = [s for s in updated_actor.statuses if s != "exposed"]
        updated_actor = updated_actor.model_copy(update={
            "resources": new_resources,
            "statuses": new_statuses,
        })
        resource_changes.append(ResourceChange(
            combatant_id=actor.id,
            heat_change=-heat_cleared,
        ))
        effects.append({
            "type": "stabilize_primary",
            "option": "cool_heat",
            "heat_cleared": heat_cleared,
            "exposed_ended": "exposed" in actor.statuses,
        })

    elif primary_option == "spend_repair_full_hp":
        # Spend 1 repair to refill HP to max
        if updated_actor.resources.repairs_remaining > 0:
            hp_restored = updated_actor.stats.hp_max - updated_actor.resources.hp_current
            new_resources = updated_actor.resources.model_copy(update={
                "hp_current": updated_actor.stats.hp_max,
                "repairs_remaining": updated_actor.resources.repairs_remaining - 1,
            })
            updated_actor = updated_actor.model_copy(update={"resources": new_resources})
            resource_changes.append(ResourceChange(
                combatant_id=actor.id,
                hp_change=hp_restored,
                repairs_change=-1,
            ))
            effects.append({
                "type": "stabilize_primary",
                "option": "spend_repair_full_hp",
                "hp_restored": hp_restored,
                "repairs_remaining": updated_actor.resources.repairs_remaining,
            })
        else:
            effects.append({
                "type": "stabilize_primary",
                "option": "spend_repair_full_hp",
                "failed": True,
                "reason": "No repairs remaining",
            })

    elif primary_option == "cancel_meltdown":
        # Attempt to cancel meltdown via engineering check (PR2 4700-4706)
        # Requires meltdown_state to be present
        if updated_actor.meltdown_state is None:
            effects.append({
                "type": "stabilize_primary",
                "option": "cancel_meltdown",
                "failed": True,
                "reason": "No active meltdown countdown",
            })
        else:
            # Engineering check: 1d20 + engineering vs DC 10
            import random
            engineering_bonus = updated_actor.stats.engineering_skill if updated_actor.stats else 0
            roll = random.randint(1, 20)
            total = roll + engineering_bonus
            dc = 10
            success = total >= dc

            if success:
                # Clear meltdown state
                updated_actor = updated_actor.model_copy(update={"meltdown_state": None})
                effects.append({
                    "type": "stabilize_primary",
                    "option": "cancel_meltdown",
                    "success": True,
                    "roll": roll,
                    "engineering_bonus": engineering_bonus,
                    "total": total,
                    "dc": dc,
                    "meltdown_cancelled": True,
                })
            else:
                effects.append({
                    "type": "stabilize_primary",
                    "option": "cancel_meltdown",
                    "success": False,
                    "roll": roll,
                    "engineering_bonus": engineering_bonus,
                    "total": total,
                    "dc": dc,
                    "meltdown_cancelled": False,
                    "reason": f"Engineering check failed ({total} vs DC {dc})",
                })

    # Secondary option
    if secondary_option == "reload_loading":
        # Reload all Loading weapons - clear needs_reload flag
        updated_actor, reloaded_weapon_ids = _reload_all_loading_weapons(updated_actor)
        effects.append({
            "type": "stabilize_secondary",
            "option": "reload_loading",
            "weapons_reloaded": len(reloaded_weapon_ids) > 0,
            "reloaded_weapon_ids": reloaded_weapon_ids,
        })

    elif secondary_option == "clear_burn":
        # End all Burn on self and reset burn_marked
        new_statuses = [s for s in updated_actor.statuses if s != "burn"]
        new_resources = updated_actor.resources.model_copy(update={"burn_marked": 0})
        updated_actor = updated_actor.model_copy(update={
            "statuses": new_statuses,
            "resources": new_resources,
        })
        effects.append({
            "type": "stabilize_secondary",
            "option": "clear_burn",
            "burn_cleared": "burn" in actor.statuses,
            "burn_marked_cleared": actor.resources.burn_marked,
        })

    elif secondary_option == "clear_condition":
        # Clear one condition on self or adjacent ally
        target_id = condition_target_id or actor.id
        clearable_conditions: list[StatusType] = [
            "impaired", "shredded", "jammed", "slowed", "immobilized", "stunned", "lock_on"
        ]

        target_idx = next((i for i, c in enumerate(scenario.combatants) if c.id == target_id), -1)
        if target_idx >= 0:
            target = scenario.combatants[target_idx]
            # Find first clearable condition
            condition_to_clear = next(
                (s for s in target.statuses if s in clearable_conditions), None
            )
            if condition_to_clear:
                new_target_statuses = [s for s in target.statuses if s != condition_to_clear]
                updated_target = target.model_copy(update={"statuses": new_target_statuses})

                # Update combatants list
                updated_combatants = list(scenario.combatants)
                updated_combatants[target_idx] = updated_target
                if target_id == actor.id:
                    updated_actor = updated_target

                scenario = MechCombatScenario(
                    combatants=updated_combatants,
                    grapples=list(scenario.grapples),
                    rounds=list(scenario.rounds),
                    terrain=scenario.terrain,
                    environment=scenario.environment,
                    deployables=dict(scenario.deployables),
                )

                effects.append({
                    "type": "stabilize_secondary",
                    "option": "clear_condition",
                    "target_id": target_id,
                    "condition_cleared": condition_to_clear,
                })
            else:
                effects.append({
                    "type": "stabilize_secondary",
                    "option": "clear_condition",
                    "target_id": target_id,
                    "failed": True,
                    "reason": "No clearable conditions",
                })

    # Update scenario with actor changes
    updated_combatants = list(scenario.combatants)
    updated_combatants[actor_idx] = updated_actor
    scenario = MechCombatScenario(
        combatants=updated_combatants,
        grapples=list(scenario.grapples),
        rounds=list(scenario.rounds),
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables=dict(scenario.deployables),
    )

    return scenario, effects, resource_changes


def _resolve_hide(
    scenario: MechCombatScenario,
    actor: CombatantState,
) -> tuple[MechCombatScenario, bool, str]:
    """Resolve Hide action (PR2 4221-4237).

    Hide is a quick action that always succeeds if:
    - Actor has hard cover, OR
    - Actor is in soft cover (smoke, etc.), OR
    - Actor is invisible

    Returns:
        Tuple of (updated scenario, success, reason)
    """
    # Check if actor is invisible (can always hide)
    if "invisible" in actor.statuses:
        return scenario, True, "Invisible mechs can always hide"

    # Check if actor is engaged (cannot hide while engaged)
    if "engaged" in actor.statuses:
        return scenario, False, "Cannot hide while engaged"

    # Check for cover at actor's position
    # For now, assume hiding is allowed (terrain/cover checks would go here)
    # A full implementation would check scenario.terrain for cover
    has_cover = True  # Simplified - assume cover available

    if has_cover:
        return scenario, True, "Hidden in cover"

    return scenario, False, "No cover available for hiding"


def _resolve_ram(
    scenario: MechCombatScenario,
    actor: CombatantState,
    target: CombatantState,
    apply_knockback: bool = True,
) -> tuple[MechCombatScenario, list[dict]]:
    """Resolve Ram action (PR2 4152-4155).

    Ram is a quick action melee attack:
    - On hit: target becomes Prone
    - May knock target back up to 1 space directly away

    Returns:
        Tuple of (updated scenario, effects list)
    """
    from core.shared.grapple import RamAttempt, attempt_ram
    from core.shared.rolls import resolve_attack

    effects: list[dict] = []

    # Roll melee attack (grit vs evasion)
    attack_bonus = actor.stats.grit if actor.stats else 0
    target_evasion = target.stats.evasion if target.stats else 10

    attack_result = resolve_attack(
        attack_bonus=attack_bonus,
        target_defense=target_evasion,
    )

    effects.append({
        "type": "ram_attack",
        "roll": attack_result.roll,
        "total": attack_result.total_accuracy,
        "hit": attack_result.hit,
        "target_evasion": target_evasion,
    })

    if not attack_result.hit:
        return scenario, effects

    # Resolve ram effects
    ram_attempt = RamAttempt(
        attacker_size=actor.stats.size,
        target_size=target.stats.size,
        hit=True,
        knockback_bonus=0,
    )

    # Get terrain occupancies for knockback blocking
    terrain_occupancies: dict[HexCoord, bool] = {}
    for c in scenario.combatants:
        if c.position and c.id != actor.id and c.id != target.id:
            terrain_occupancies[c.position.coord] = True

    ram_result = attempt_ram(
        ram_attempt,
        terrain_occupancies=terrain_occupancies if apply_knockback else None,
        attacker_position=actor.position.coord if actor.position else None,
        target_position=target.position.coord if target.position else None,
    )

    effects.append({
        "type": "ram_result",
        "target_becomes_prone": ram_result.target_becomes_prone,
        "knockback_spaces": ram_result.knockback_spaces if apply_knockback else 0,
        "knockback_blocked": ram_result.knockback_blocked,
        "reason": ram_result.reason,
    })

    # Apply knockback position change if applicable
    if apply_knockback and ram_result.knockback_spaces > 0 and target.position and actor.position:
        from core.shared.grapple import get_knockback_direction
        direction = get_knockback_direction(actor.position.coord, target.position.coord)
        new_coord = HexCoord(
            q=target.position.coord.q + (direction.q * ram_result.knockback_spaces),
            r=target.position.coord.r + (direction.r * ram_result.knockback_spaces),
        )
        new_position = target.position.model_copy(update={"coord": new_coord})

        target_idx = next((i for i, c in enumerate(scenario.combatants) if c.id == target.id), -1)
        if target_idx >= 0:
            updated_target = target.model_copy(update={"position": new_position})
            updated_combatants = list(scenario.combatants)
            updated_combatants[target_idx] = updated_target
            scenario = MechCombatScenario(
                combatants=updated_combatants,
                grapples=list(scenario.grapples),
                rounds=list(scenario.rounds),
                terrain=scenario.terrain,
                environment=scenario.environment,
                deployables=dict(scenario.deployables),
            )

    return scenario, effects


def _apply_knockback_on_hit(
    scenario: MechCombatScenario,
    attacker: CombatantState,
    target: CombatantState,
    knockback_spaces: int,
) -> tuple[MechCombatScenario, dict | None]:
    """Apply knockback to target after a successful attack hit.

    Per PR2 5027-5028:
    "On hit, you may knock back a target X spaces in a straight line
    directly away from the point of origin"

    Args:
        scenario: Current combat scenario
        attacker: The attacking combatant (knockback origin)
        target: The target combatant to knock back
        knockback_spaces: Number of spaces to knock back

    Returns:
        Tuple of (updated scenario, knockback effect dict or None)
    """
    if knockback_spaces <= 0:
        return scenario, None

    if attacker.position is None or target.position is None:
        return scenario, None

    # Build occupancy set for blocking check (all combatants except target)
    occupied_hexes: dict[HexCoord, bool] = {}
    for c in scenario.combatants:
        if c.id != target.id and c.position is not None:
            occupied_hexes[c.position.coord] = True

    # Resolve knockback using involuntary movement helper
    result = resolve_knockback(
        source=attacker.position.coord,
        target=target.position.coord,
        spaces=knockback_spaces,
        terrain=scenario.terrain,
        occupied_hexes=occupied_hexes,
    )

    # Calculate final position from knockback result
    final_position = result.end_position if result.end_position else target.position.coord

    # Update target position in scenario if it changed
    if final_position != target.position.coord:
        new_position = target.position.model_copy(update={"coord": final_position})
        updated_target = target.model_copy(update={"position": new_position})

        updated_combatants = [
            updated_target if c.id == target.id else c
            for c in scenario.combatants
        ]
        scenario = MechCombatScenario(
            combatants=updated_combatants,
            grapples=list(scenario.grapples),
            rounds=list(scenario.rounds),
            terrain=scenario.terrain,
            environment=scenario.environment,
            deployables=dict(scenario.deployables),
        )

    effect: dict = {
        "type": "knockback",
        "target_id": str(target.id),
        "spaces_requested": knockback_spaces,
        "spaces_moved": result.spaces_knocked,
        "final_position": {"q": final_position.q, "r": final_position.r},
        "blocked": result.obstructed,
    }

    return scenario, effect


def _resolve_grapple(
    scenario: MechCombatScenario,
    actor: CombatantState,
    target: CombatantState,
) -> tuple[MechCombatScenario, list[dict]]:
    """Resolve Grapple action (PR2 4157-4177).

    Grapple is a quick action melee attack:
    - On hit: both parties become engaged, neither can boost/react
    - Smaller party is immobilized, moves when larger moves
    - Same size: contested HULL check at start of turn

    Returns:
        Tuple of (updated scenario, effects list)
    """
    from core.shared.grapple import GrappleAttempt, attempt_grapple
    from core.shared.rolls import resolve_attack

    effects: list[dict] = []

    # Roll melee attack (grit vs evasion)
    attack_bonus = actor.stats.grit if actor.stats else 0
    target_evasion = target.stats.evasion if target.stats else 10

    attack_result = resolve_attack(
        attack_bonus=attack_bonus,
        target_defense=target_evasion,
    )

    effects.append({
        "type": "grapple_attack",
        "roll": attack_result.roll,
        "total": attack_result.total_accuracy,
        "hit": attack_result.hit,
        "target_evasion": target_evasion,
    })

    if not attack_result.hit:
        return scenario, effects

    # Resolve grapple effects
    grapple_attempt = GrappleAttempt(
        attacker_size=actor.stats.size,
        target_size=target.stats.size,
        hit=True,
        attacker_hull_bonus=0,  # Would come from pilot skills
        target_hull_bonus=0,
    )

    grapple_result = attempt_grapple(grapple_attempt)

    effects.append({
        "type": "grapple_result",
        "grapple_initiated": grapple_result.grapple_initiated,
        "attacker_engaged": grapple_result.attacker_engaged,
        "target_engaged": grapple_result.target_engaged,
        "smaller_party": grapple_result.smaller_party,
        "target_becomes_immobilized": grapple_result.target_becomes_immobilized,
        "attacker_becomes_immobilized": grapple_result.attacker_becomes_immobilized,
        "contested_check_required": grapple_result.contested_check_required,
        "contested_check_winner": grapple_result.contested_check_winner,
        "reason": grapple_result.reason,
    })

    if grapple_result.grapple_initiated:
        # Add engaged status to both parties
        actor_idx = next((i for i, c in enumerate(scenario.combatants) if c.id == actor.id), -1)
        target_idx = next((i for i, c in enumerate(scenario.combatants) if c.id == target.id), -1)

        updated_combatants = list(scenario.combatants)

        if actor_idx >= 0:
            actor_statuses = list(actor.statuses)
            if "engaged" not in actor_statuses:
                actor_statuses.append("engaged")
            if grapple_result.attacker_becomes_immobilized and "immobilized" not in actor_statuses:
                actor_statuses.append("immobilized")
            updated_combatants[actor_idx] = actor.model_copy(update={"statuses": actor_statuses})

        if target_idx >= 0:
            target_statuses = list(target.statuses)
            if "engaged" not in target_statuses:
                target_statuses.append("engaged")
            if grapple_result.target_becomes_immobilized and "immobilized" not in target_statuses:
                target_statuses.append("immobilized")
            updated_combatants[target_idx] = target.model_copy(update={"statuses": target_statuses})

        # Add grapple link
        new_grapple = GrappleLink(
            grappler_id=actor.id,
            target_id=target.id,
        )
        updated_grapples = list(scenario.grapples) + [new_grapple]

        scenario = MechCombatScenario(
            combatants=updated_combatants,
            grapples=updated_grapples,
            rounds=list(scenario.rounds),
            terrain=scenario.terrain,
            environment=scenario.environment,
            deployables=dict(scenario.deployables),
        )

    return scenario, effects


def _resolve_search(
    scenario: MechCombatScenario,
    actor: CombatantState,
    target: CombatantState,
) -> tuple[MechCombatScenario, list[dict]]:
    """Resolve Search action (PR2 4241-4249).

    Search is a quick action contested check:
    - Searching party: Systems check
    - Hidden mech target: Agility check
    - On success: target loses Hidden status

    Returns:
        Tuple of (updated scenario, effects list)
    """
    effects: list[dict] = []

    # Check if target is actually hidden
    if "hidden" not in target.statuses:
        effects.append({
            "type": "search",
            "search_success": False,
            "reason": "Target is not hidden",
        })
        return scenario, effects

    # Contested check: Systems (searcher) vs Agility (hider)
    # Using tech_attack as systems bonus, speed as rough agility proxy
    searcher_systems = actor.stats.tech_attack if actor.stats else 0
    target_agility = (target.stats.speed // 2) if target.stats else 0  # Rough approximation

    searcher_roll = roll_dice("1d20")
    target_roll = roll_dice("1d20")

    searcher_total = searcher_roll + searcher_systems
    target_total = target_roll + target_agility

    search_success = searcher_total > target_total

    effects.append({
        "type": "search",
        "searcher_roll": searcher_roll,
        "searcher_systems": searcher_systems,
        "searcher_total": searcher_total,
        "target_roll": target_roll,
        "target_agility": target_agility,
        "target_total": target_total,
        "search_success": search_success,
        "reason": f"Contested check: {searcher_total} vs {target_total}",
    })

    return scenario, effects


# =============================================================================
# Burn Tick Resolution
# =============================================================================


def _resolve_burn_tick(
    scenario: MechCombatScenario,
    actor: CombatantState,
    force_roll: int | None = None,
) -> tuple[MechCombatScenario, "BurnTickResult | None"]:
    """Resolve burn damage tick at end of turn.

    Per PR2 5017-5021: Engineering check (1d20 + ENG vs DC 10).
    Success clears all burn. Failure deals burn damage (ignores armor).

    Args:
        scenario: Current combat scenario
        actor: The combatant whose turn is ending
        force_roll: Optional forced roll value for testing

    Returns:
        Tuple of (updated scenario, BurnTickResult or None if no burn)
    """
    from core.mech.combat_models import BurnTickResult

    # Check if actor has burn status AND burn_marked > 0
    if "burn" not in actor.statuses or actor.resources.burn_marked <= 0:
        return scenario, None

    burn_amount = actor.resources.burn_marked
    engineering_bonus = actor.stats.engineering_skill if actor.stats else 0
    dc = 10

    # Roll engineering check
    if force_roll is not None:
        engineering_roll = force_roll
    else:
        engineering_roll = roll_dice("1d20")

    total = engineering_roll + engineering_bonus
    success = total >= dc

    # Find actor index
    actor_idx = next((i for i, c in enumerate(scenario.combatants) if c.id == actor.id), -1)
    if actor_idx < 0:
        return scenario, None

    damage_taken = 0
    burn_cleared = False
    updated_actor = actor

    if success:
        # Success: Clear all burn
        new_statuses = [s for s in actor.statuses if s != "burn"]
        new_resources = actor.resources.model_copy(update={"burn_marked": 0})
        updated_actor = actor.model_copy(update={
            "statuses": new_statuses,
            "resources": new_resources,
        })
        burn_cleared = True
    else:
        # Failure: Take burn damage (ignores armor - full AP bypass)
        # We need to apply damage directly to HP, bypassing armor
        damage_taken = burn_amount
        new_hp = max(0, actor.resources.hp_current - damage_taken)
        new_resources = actor.resources.model_copy(update={"hp_current": new_hp})
        updated_actor = actor.model_copy(update={"resources": new_resources})

    # Update scenario
    updated_combatants = list(scenario.combatants)
    updated_combatants[actor_idx] = updated_actor

    scenario = MechCombatScenario(
        combatants=updated_combatants,
        grapples=list(scenario.grapples),
        rounds=list(scenario.rounds),
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables=dict(scenario.deployables),
    )

    return scenario, BurnTickResult(
        target_id=actor.id,
        burn_amount=burn_amount,
        engineering_roll=engineering_roll,
        engineering_bonus=engineering_bonus,
        total=total,
        dc=dc,
        success=success,
        damage_taken=damage_taken,
        burn_cleared=burn_cleared,
    )


# =============================================================================
# Movement Resolution
# =============================================================================


def _apply_engagement_status(
    scenario: MechCombatScenario,
    mover: CombatantState,
    final_position: HexPosition,
    ignore_engagement: bool = False,
) -> tuple[MechCombatScenario, list[dict]]:
    """Apply engaged status to mover and all adjacent hostiles after movement.

    Per PR2 3817-3819:
    "If you move adjacent to a hostile character, you become engaged."

    Both parties become engaged (mutual engagement, matching grapple behavior).

    Args:
        scenario: Current combat scenario
        mover: The combatant that just moved
        final_position: The final position of the mover
        ignore_engagement: Whether to skip status application (e.g., Disengage)

    Returns:
        Tuple of (updated scenario, list of effects)
    """
    effects: list[dict] = []

    # Skip if ignoring engagement (Disengage action)
    if ignore_engagement:
        return scenario, effects

    # Find all adjacent hostile combatants
    updated_combatants = list(scenario.combatants)
    mover_idx = next((i for i, c in enumerate(updated_combatants) if c.id == mover.id), -1)

    adjacent_hostile_ids: list[tuple[int, CombatantState]] = []
    for i, combatant in enumerate(updated_combatants):
        if combatant.id == mover.id:
            continue
        if combatant.position is None:
            continue
        if not _is_hostile(mover.id, combatant.id, scenario):
            continue
        if final_position.coord.distance_to(combatant.position.coord) == 1:
            adjacent_hostile_ids.append((i, combatant))

    if not adjacent_hostile_ids:
        return scenario, effects

    # Apply engaged to mover if not already engaged
    if mover_idx >= 0:
        mover_statuses = list(updated_combatants[mover_idx].statuses)
        if "engaged" not in mover_statuses:
            mover_statuses.append("engaged")
            updated_combatants[mover_idx] = updated_combatants[mover_idx].model_copy(
                update={"statuses": mover_statuses}
            )
            effects.append({
                "type": "status_applied",
                "target_id": str(mover.id),
                "status": "engaged",
                "reason": "moved_adjacent_to_hostile",
            })

    # Apply engaged to each adjacent hostile if not already engaged
    for idx, hostile in adjacent_hostile_ids:
        hostile_statuses = list(hostile.statuses)
        if "engaged" not in hostile_statuses:
            hostile_statuses.append("engaged")
            updated_combatants[idx] = hostile.model_copy(
                update={"statuses": hostile_statuses}
            )
            effects.append({
                "type": "status_applied",
                "target_id": str(hostile.id),
                "status": "engaged",
                "reason": "hostile_moved_adjacent",
            })

    # Build updated scenario
    updated_scenario = MechCombatScenario(
        combatants=updated_combatants,
        grapples=list(scenario.grapples),
        rounds=list(scenario.rounds),
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables=dict(scenario.deployables),
    )

    return updated_scenario, effects


def _resolve_movement(
    scenario: MechCombatScenario,
    actor: CombatantState,
    path: list[HexPosition],
    is_boost: bool = False,
    apply_damage_func=None,
    ignore_engagement: bool = False,
) -> tuple[MechCombatScenario, list[dict]]:
    """Resolve movement action - validate path, apply terrain effects, update position.

    Per PR2:
    - Move: Free action, move up to Speed in spaces
    - Boost: Quick action, move up to 2x Speed in spaces
    - Difficult terrain costs 2 spaces per 1 space moved
    - Dangerous terrain triggers engineering check (DC 10), 5 damage on failure
    - Engagement: If moving adjacent to same-size-or-larger hostile, must stop

    Args:
        scenario: Current combat scenario
        actor: The moving combatant
        path: List of hex positions for the movement path
        is_boost: Whether this is a Boost action (2x speed)
        apply_damage_func: Function to apply damage (for dangerous terrain)
        ignore_engagement: Whether to ignore engagement rules (e.g., Disengage action)

    Returns:
        Tuple of (updated scenario, effects list)
    """
    from core.shared.terrain import (
        terrain_index,
        calculate_movement_cost,
        resolve_dangerous_terrain,
    )

    effects: list[dict] = []

    if not path:
        effects.append({
            "type": "movement",
            "success": False,
            "reason": "Empty movement path",
        })
        return scenario, effects

    # Check engagement stop (unless ignoring via Disengage)
    # Per PR2 3817-3819: Must stop when moving adjacent to same-size-or-larger hostile
    if not ignore_engagement and actor.position is not None:
        # Include starting position in path for engagement check
        # check_engagement_stop expects path[0] to be the starting position
        path_coords = [actor.position.coord] + [p.coord for p in path]
        entity_size = actor.stats.size if actor.stats else "size_1"
        should_stop, stop_coord = check_engagement_stop(
            entity_id=actor.id,
            entity_size=entity_size,
            path=path_coords,
            scenario=scenario,
            ignore_engagement=False,
        )

        if should_stop and stop_coord:
            # Truncate path to stop position
            stop_idx = next(
                (i for i, p in enumerate(path) if p.coord == stop_coord),
                len(path) - 1
            )
            path = path[:stop_idx + 1]
            effects.append({
                "type": "engagement_stop",
                "stop_coord": {"q": stop_coord.q, "r": stop_coord.r},
                "reason": "same_size_or_larger_hostile",
            })

    # Calculate total movement cost with terrain
    total_cost = 0
    terrain_idx = terrain_index(scenario.terrain)

    for hex_pos in path:
        cost = calculate_movement_cost(1, scenario.terrain, hex_pos.coord)
        total_cost += cost

    # Check speed budget (boost = 2x speed)
    base_speed = actor.stats.speed if actor.stats else 4
    speed = base_speed * (2 if is_boost else 1)

    if total_cost > speed:
        effects.append({
            "type": "movement",
            "success": False,
            "reason": "exceeds_speed",
            "speed": speed,
            "cost": total_cost,
        })
        return scenario, effects

    # Check dangerous terrain and resolve checks
    # Get current round for tracking checks
    current_round = len(scenario.rounds) if scenario.rounds else 1

    for hex_pos in path:
        terrain_hex = terrain_idx.get(hex_pos.coord)
        if terrain_hex and terrain_hex.dangerous:
            # Engineering skill bonus (use tech_attack as proxy)
            skill_bonus = actor.stats.tech_attack if actor.stats else 0

            danger_result = resolve_dangerous_terrain(
                terrain=scenario.terrain,
                coord=hex_pos.coord,
                skill_bonus=skill_bonus,
                round_checked=current_round,
            )

            effects.append({
                "type": "dangerous_terrain",
                "coord": {"q": hex_pos.coord.q, "r": hex_pos.coord.r},
                "check_passed": danger_result.check_passed,
                "roll": danger_result.roll_result,
                "damage": danger_result.damage_dealt,
            })

            # Apply damage if check failed
            if danger_result.check_passed is False and danger_result.damage_dealt > 0:
                if apply_damage_func is not None:
                    scenario, change, structure_result = apply_damage_func(
                        scenario, actor.id, danger_result.damage_dealt, armor_piercing=0
                    )
                    effects.append({
                        "type": "damage",
                        "target_id": actor.id,
                        "amount": danger_result.damage_dealt,
                        "source": "dangerous_terrain",
                    })

    # Update actor position to final hex
    final_position = path[-1]
    actor_idx = next((i for i, c in enumerate(scenario.combatants) if c.id == actor.id), -1)

    if actor_idx >= 0:
        updated_actor = actor.model_copy(update={"position": final_position})

        # Auto-retrieve thrown weapons when moving adjacent to their location.
        if actor.inventory is not None:
            updated_mounts = []
            for mount in actor.inventory.mounts:
                new_weapons = []
                for weapon in mount.weapons:
                    if weapon.thrown_coord is not None:
                        for step in path:
                            if step.coord.distance_to(weapon.thrown_coord) <= 1:
                                effects.append({
                                    "type": "retrieve_thrown_weapon",
                                    "weapon_id": weapon.weapon_id,
                                    "coord": {
                                        "q": weapon.thrown_coord.q,
                                        "r": weapon.thrown_coord.r,
                                    },
                                })
                                weapon = weapon.model_copy(update={"thrown_coord": None})
                                break
                    new_weapons.append(weapon)
                updated_mounts.append(mount.model_copy(update={"weapons": new_weapons}))

            updated_inventory = actor.inventory.model_copy(update={"mounts": updated_mounts})
            updated_actor = updated_actor.model_copy(update={"inventory": updated_inventory})

        updated_combatants = list(scenario.combatants)
        updated_combatants[actor_idx] = updated_actor

        scenario = MechCombatScenario(
            combatants=updated_combatants,
            grapples=list(scenario.grapples),
            rounds=list(scenario.rounds),
            terrain=scenario.terrain,
            environment=scenario.environment,
            deployables=dict(scenario.deployables),
        )

    # Apply engaged status to mover and adjacent hostiles after position update
    # Per PR2 3817: Moving adjacent to a hostile triggers engagement
    # Re-fetch updated actor from scenario
    updated_actor_for_engagement = next(
        (c for c in scenario.combatants if c.id == actor.id), actor
    )
    scenario, engagement_effects = _apply_engagement_status(
        scenario,
        updated_actor_for_engagement,
        final_position,
        ignore_engagement=ignore_engagement,
    )
    effects.extend(engagement_effects)

    effects.append({
        "type": "movement",
        "success": True,
        "spaces": len(path),
        "cost": total_cost,
        "final_position": {
            "q": final_position.coord.q,
            "r": final_position.coord.r,
            "elevation": final_position.elevation,
        },
    })

    return scenario, effects


# =============================================================================
# Mount/Dismount/Eject Resolution
# =============================================================================


def _resolve_mount(
    scenario: MechCombatScenario,
    actor: CombatantState,
    target_mech_id: str,
) -> tuple[MechCombatScenario, list[dict]]:
    """Resolve Mount action - pilot enters a mech.

    Per PR2 4318-4327:
    - Mount is a Full action
    - Pilot must be adjacent to mech to mount
    - Pilot enters mech, no longer on the battlefield separately

    Returns:
        Tuple of (updated scenario, effects list)
    """
    effects: list[dict] = []

    # Validate actor is a pilot
    if actor.kind != "pilot":
        effects.append({
            "type": "mount",
            "success": False,
            "reason": "Only pilots can mount mechs",
        })
        return scenario, effects

    # Find target mech
    target_mech = next(
        (c for c in scenario.combatants if c.id == target_mech_id),
        None
    )
    if target_mech is None:
        effects.append({
            "type": "mount",
            "success": False,
            "reason": f"Mech {target_mech_id} not found",
        })
        return scenario, effects

    # Validate target is a mech
    if target_mech.kind != "mech":
        effects.append({
            "type": "mount",
            "success": False,
            "reason": "Can only mount mechs",
        })
        return scenario, effects

    # Check adjacency (pilot must be adjacent to mech)
    if actor.position is not None and target_mech.position is not None:
        distance = actor.position.coord.distance_to(target_mech.position.coord)
        if distance > 1:
            effects.append({
                "type": "mount",
                "success": False,
                "reason": "Must be adjacent to mount mech",
            })
            return scenario, effects

    # Check if mech already has a pilot
    if target_mech.mounted_pilot_id is not None:
        effects.append({
            "type": "mount",
            "success": False,
            "reason": "Mech already has a pilot mounted",
        })
        return scenario, effects

    # Update pilot and mech state
    actor_idx = next((i for i, c in enumerate(scenario.combatants) if c.id == actor.id), -1)
    mech_idx = next((i for i, c in enumerate(scenario.combatants) if c.id == target_mech_id), -1)

    if actor_idx >= 0 and mech_idx >= 0:
        updated_combatants = list(scenario.combatants)

        # Pilot now piloting the mech
        updated_pilot = actor.model_copy(update={
            "piloting_mech_id": target_mech_id,
            "position": None,  # Pilot is no longer on the battlefield separately
        })
        updated_combatants[actor_idx] = updated_pilot

        # Mech now has pilot mounted
        updated_mech = target_mech.model_copy(update={
            "mounted_pilot_id": actor.id,
        })
        updated_combatants[mech_idx] = updated_mech

        scenario = MechCombatScenario(
            combatants=updated_combatants,
            grapples=list(scenario.grapples),
            rounds=list(scenario.rounds),
            terrain=scenario.terrain,
            environment=scenario.environment,
            deployables=dict(scenario.deployables),
        )

    effects.append({
        "type": "mount",
        "success": True,
        "pilot_id": actor.id,
        "mech_id": target_mech_id,
    })

    return scenario, effects


def _resolve_dismount(
    scenario: MechCombatScenario,
    actor: CombatantState,
) -> tuple[MechCombatScenario, list[dict]]:
    """Resolve Dismount action - pilot exits a mech.

    Per PR2 4318-4327:
    - Dismount is a Full action
    - Pilot is placed in an adjacent space
    - Pilot becomes a separate combatant on the battlefield

    Returns:
        Tuple of (updated scenario, effects list)
    """
    effects: list[dict] = []

    # Validate actor is a mech with a mounted pilot
    if actor.kind != "mech":
        effects.append({
            "type": "dismount",
            "success": False,
            "reason": "Dismount can only be done from a mech",
        })
        return scenario, effects

    if actor.mounted_pilot_id is None:
        effects.append({
            "type": "dismount",
            "success": False,
            "reason": "No pilot mounted in this mech",
        })
        return scenario, effects

    # Find the mounted pilot
    pilot = next(
        (c for c in scenario.combatants if c.id == actor.mounted_pilot_id),
        None
    )
    if pilot is None:
        effects.append({
            "type": "dismount",
            "success": False,
            "reason": f"Pilot {actor.mounted_pilot_id} not found",
        })
        return scenario, effects

    # Calculate adjacent position for pilot
    pilot_position = None
    if actor.position is not None:
        # Find first free adjacent hex
        for neighbor in actor.position.coord.neighbors():
            # Check if any combatant is in this position
            occupied = any(
                c.position is not None and c.position.coord == neighbor
                for c in scenario.combatants if c.id != pilot.id
            )
            if not occupied:
                pilot_position = HexPosition(coord=neighbor, elevation=actor.position.elevation)
                break

    if pilot_position is None and actor.position is not None:
        effects.append({
            "type": "dismount",
            "success": False,
            "reason": "No free adjacent space for pilot",
        })
        return scenario, effects

    # Update pilot and mech state
    pilot_idx = next((i for i, c in enumerate(scenario.combatants) if c.id == pilot.id), -1)
    mech_idx = next((i for i, c in enumerate(scenario.combatants) if c.id == actor.id), -1)

    if pilot_idx >= 0 and mech_idx >= 0:
        updated_combatants = list(scenario.combatants)

        # Pilot exits mech
        updated_pilot = pilot.model_copy(update={
            "piloting_mech_id": None,
            "position": pilot_position,
        })
        updated_combatants[pilot_idx] = updated_pilot

        # Mech no longer has pilot
        updated_mech = actor.model_copy(update={
            "mounted_pilot_id": None,
        })
        updated_combatants[mech_idx] = updated_mech

        scenario = MechCombatScenario(
            combatants=updated_combatants,
            grapples=list(scenario.grapples),
            rounds=list(scenario.rounds),
            terrain=scenario.terrain,
            environment=scenario.environment,
            deployables=dict(scenario.deployables),
        )

    effects.append({
        "type": "dismount",
        "success": True,
        "pilot_id": pilot.id,
        "mech_id": actor.id,
        "pilot_position": {
            "q": pilot_position.coord.q,
            "r": pilot_position.coord.r,
            "elevation": pilot_position.elevation,
        } if pilot_position else None,
    })

    return scenario, effects


def _resolve_eject(
    scenario: MechCombatScenario,
    actor: CombatantState,
    eject_direction: HexCoord | None,
) -> tuple[MechCombatScenario, list[dict]]:
    """Resolve Eject action - pilot emergency ejects from mech.

    Per PR2 4318-4327:
    - Eject is a Quick action
    - Pilot flies 6 spaces in chosen direction
    - One-way system: cannot eject again until full repair
    - Pilot is PERMANENTLY impaired until full repair

    Returns:
        Tuple of (updated scenario, effects list)
    """
    effects: list[dict] = []

    # Validate actor is a mech
    if actor.kind != "mech":
        effects.append({
            "type": "eject",
            "success": False,
            "reason": "Eject can only be done from a mech",
        })
        return scenario, effects

    # Check if eject was already used
    if actor.eject_used:
        effects.append({
            "type": "eject",
            "success": False,
            "reason": "Eject system already used this combat",
        })
        return scenario, effects

    # Check if there's a mounted pilot
    if actor.mounted_pilot_id is None:
        effects.append({
            "type": "eject",
            "success": False,
            "reason": "No pilot mounted in this mech",
        })
        return scenario, effects

    # Find the mounted pilot
    pilot = next(
        (c for c in scenario.combatants if c.id == actor.mounted_pilot_id),
        None
    )
    if pilot is None:
        effects.append({
            "type": "eject",
            "success": False,
            "reason": f"Pilot {actor.mounted_pilot_id} not found",
        })
        return scenario, effects

    # Calculate final position (6 spaces in direction)
    pilot_position = None
    if actor.position is not None:
        if eject_direction is not None:
            # Scale direction by 6 spaces
            final_coord = hex_add(actor.position.coord, hex_scale(eject_direction, 6))
            pilot_position = HexPosition(coord=final_coord, elevation=actor.position.elevation)
        else:
            # Default: adjacent to mech
            for neighbor in actor.position.coord.neighbors():
                occupied = any(
                    c.position is not None and c.position.coord == neighbor
                    for c in scenario.combatants if c.id != pilot.id
                )
                if not occupied:
                    pilot_position = HexPosition(coord=neighbor, elevation=actor.position.elevation)
                    break

    # Update pilot and mech state
    pilot_idx = next((i for i, c in enumerate(scenario.combatants) if c.id == pilot.id), -1)
    mech_idx = next((i for i, c in enumerate(scenario.combatants) if c.id == actor.id), -1)

    if pilot_idx >= 0 and mech_idx >= 0:
        updated_combatants = list(scenario.combatants)

        # Pilot ejects with impaired status
        pilot_statuses = list(pilot.statuses)
        if "impaired" not in pilot_statuses:
            pilot_statuses.append("impaired")

        updated_pilot = pilot.model_copy(update={
            "piloting_mech_id": None,
            "position": pilot_position,
            "statuses": pilot_statuses,
        })
        updated_combatants[pilot_idx] = updated_pilot

        # Mech: pilot ejected, mark eject as used
        updated_mech = actor.model_copy(update={
            "mounted_pilot_id": None,
            "eject_used": True,
        })
        updated_combatants[mech_idx] = updated_mech

        scenario = MechCombatScenario(
            combatants=updated_combatants,
            grapples=list(scenario.grapples),
            rounds=list(scenario.rounds),
            terrain=scenario.terrain,
            environment=scenario.environment,
            deployables=dict(scenario.deployables),
        )

    effects.append({
        "type": "eject",
        "success": True,
        "pilot_id": pilot.id,
        "mech_id": actor.id,
        "impaired_applied": True,
        "eject_used_set": True,
        "pilot_position": {
            "q": pilot_position.coord.q,
            "r": pilot_position.coord.r,
            "elevation": pilot_position.elevation,
        } if pilot_position else None,
    })

    return scenario, effects


# =============================================================================
# Deployable Resolution Helpers (PR2 5070-5088)
# =============================================================================


def _resolve_deploy(
    scenario: MechCombatScenario,
    actor: CombatantState,
    target_position: HexPosition,
    deploy_kind: str,
    deploy_name: str | None,
    system_id: str | None,
    mine_type: str | None = None,
    current_round: int = 1,
) -> tuple[MechCombatScenario, list[dict]]:
    """Resolve Deploy quick action - place a deployable on the field.

    Per PR2 rules:
    - Deploy is a quick action
    - Deployables are placed at target position
    - Mines arm at the start of the deployer's next turn
    - Drones act on owner's turn

    Args:
        scenario: Current combat scenario
        actor: The combatant deploying
        target_position: Position to deploy to
        deploy_kind: Kind of deployable (drone, mine, deployable)
        deploy_name: Optional name for the deployable
        system_id: Optional system ID that provides the deployable
        mine_type: Type of mine if deploying a mine
        current_round: Current round number for arming calculation

    Returns:
        Tuple of (updated scenario, effects list)
    """
    from core.shared.deployables import create_drone, create_mine, create_deployable

    effects: list[dict] = []

    # Generate unique deployable ID
    existing_ids = set(scenario.deployables.keys())
    base_id = f"{actor.id}_{deploy_kind}"
    deploy_id = base_id
    counter = 1
    while deploy_id in existing_ids:
        deploy_id = f"{base_id}_{counter}"
        counter += 1

    # Generate name if not provided
    if deploy_name is None:
        deploy_name = f"{actor.name}'s {deploy_kind.title()}"

    # Create the deployable based on kind
    if deploy_kind == "mine":
        # Default mine type if not specified
        actual_mine_type = mine_type or "explosive"
        deployable = create_mine(
            id=deploy_id,
            name=deploy_name,
            owner_id=actor.id,
            position=target_position,
            mine_type=actual_mine_type,  # type: ignore
            tier=1,
        )
        # Set arming turn to next turn
        arming_turn = current_round + 1
        deployable = deployable.model_copy(update={"arming_turn": arming_turn})

        effects.append({
            "type": "deploy",
            "deploy_kind": "mine",
            "deploy_id": deploy_id,
            "deploy_name": deploy_name,
            "mine_type": actual_mine_type,
            "position": {
                "q": target_position.coord.q,
                "r": target_position.coord.r,
                "elevation": target_position.elevation,
            },
            "arming_turn": arming_turn,
            "owner_id": actor.id,
        })

    elif deploy_kind == "drone":
        deployable = create_drone(
            id=deploy_id,
            name=deploy_name,
            owner_id=actor.id,
            position=target_position,
            can_act=False,  # Default to no actions unless system specifies
            can_move=True,
            speed=4,
        )

        effects.append({
            "type": "deploy",
            "deploy_kind": "drone",
            "deploy_id": deploy_id,
            "deploy_name": deploy_name,
            "position": {
                "q": target_position.coord.q,
                "r": target_position.coord.r,
                "elevation": target_position.elevation,
            },
            "owner_id": actor.id,
            "acts_on_owner_turn": True,
        })

    else:  # generic deployable
        deployable = create_deployable(
            id=deploy_id,
            name=deploy_name,
            owner_id=actor.id,
            position=target_position,
            size=1,
            cover=None,
            armor=0,
        )

        effects.append({
            "type": "deploy",
            "deploy_kind": "deployable",
            "deploy_id": deploy_id,
            "deploy_name": deploy_name,
            "position": {
                "q": target_position.coord.q,
                "r": target_position.coord.r,
                "elevation": target_position.elevation,
            },
            "owner_id": actor.id,
        })

    # Add deployable to scenario
    updated_deployables = dict(scenario.deployables)
    updated_deployables[deploy_id] = deployable

    scenario = MechCombatScenario(
        combatants=list(scenario.combatants),
        grapples=list(scenario.grapples),
        rounds=list(scenario.rounds),
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables=updated_deployables,
    )

    return scenario, effects


def _check_mine_triggers(
    scenario: MechCombatScenario,
    mover_id: str,
    path: list[HexPosition],
) -> tuple[MechCombatScenario, list[dict]]:
    """Check if movement path triggers armed mines.

    Per PR2 5085-5086:
    "Mines activate as soon as any character enters an adjacent space...
    creating a burst attack starting from the space in which they were placed."

    Args:
        scenario: Current combat scenario
        mover_id: ID of the combatant moving
        path: Movement path being taken

    Returns:
        Tuple of (updated scenario with detonated mines removed, detonation effects)
    """
    from core.shared.deployables import MineDetonationInput, resolve_mine_detonation, get_mine_effect_profile

    effects: list[dict] = []
    mines_to_remove: list[str] = []

    # Get the mover for faction check
    mover = next((c for c in scenario.combatants if c.id == mover_id), None)
    if mover is None:
        return scenario, effects

    # Check each hex in the path
    path_coords = {(pos.coord.q, pos.coord.r) for pos in path}

    for mine_id, mine in scenario.deployables.items():
        # Only armed mines with trigger_on_adjacent_entry
        if mine.kind != "mine" or not mine.is_armed or not mine.trigger_on_adjacent_entry:
            continue

        # Don't trigger on owner's movement
        if mine.owner_id == mover_id:
            continue

        # Check if any path hex is adjacent to mine
        mine_coord = mine.position.coord
        mine_neighbors = {(n.q, n.r) for n in mine_coord.neighbors()}

        # Include the mine's own hex as a trigger
        mine_neighbors.add((mine_coord.q, mine_coord.r))

        if path_coords & mine_neighbors:
            # Mine triggered!
            # Get mine effect profile for detonation
            mine_type = "explosive"  # Default
            profile = get_mine_effect_profile(mine_type, tier=1)

            detonation_input = MineDetonationInput(
                mine_id=mine_id,
                triggerer_id=mover_id,
                scenario=scenario,
                effect_profile=profile,
                tier=1,
            )
            detonation_result = resolve_mine_detonation(detonation_input)

            if detonation_result.detonated:
                mines_to_remove.append(mine_id)
                effects.append({
                    "type": "mine_detonation",
                    "mine_id": mine_id,
                    "mine_name": mine.name,
                    "triggerer_id": mover_id,
                    "affected_combatant_ids": detonation_result.affected_combatant_ids,
                    "burst_radius": profile.burst_radius,
                    "reason": detonation_result.reason,
                })

    # Remove detonated mines from scenario
    if mines_to_remove:
        updated_deployables = {
            k: v for k, v in scenario.deployables.items()
            if k not in mines_to_remove
        }
        scenario = MechCombatScenario(
            combatants=list(scenario.combatants),
            grapples=list(scenario.grapples),
            rounds=list(scenario.rounds),
            terrain=scenario.terrain,
            environment=scenario.environment,
            deployables=updated_deployables,
        )

    return scenario, effects


# =============================================================================
# Single Attack Resolution (for overwatch and extracted attack logic)
# =============================================================================


def resolve_single_attack(
    scenario: MechCombatScenario,
    attacker: CombatantState,
    target: CombatantState,
    weapon_id: str,
    apply_damage_func=None,
    profile_id: str | None = None,
) -> tuple[MechCombatScenario, "AttackOutcome"]:
    """Resolve a single attack from attacker to target.

    This is a reusable helper for attack resolution that can be called from
    both execute_action() (skirmish/barrage) and execute_reaction() (overwatch).

    Per PR2 rules:
    - Roll 1d20 + grit vs target evasion (or e-defense for smart weapons)
    - Apply accuracy/difficulty modifiers from statuses
    - Apply cover difficulty for ranged attacks
    - Check invisibility 50% miss chance
    - Roll damage on hit (critical = 2x damage)
    - Apply armor and AP

    Args:
        scenario: Current combat scenario
        attacker: The attacking combatant
        target: The target combatant
        weapon_id: Weapon ID being used
        apply_damage_func: Optional damage application function (for dependency injection)
        profile_id: Optional weapon profile ID for weapons with multiple profiles

    Returns:
        Tuple of (updated scenario, AttackOutcome with hit/miss/damage info)
    """
    from core.shared.rolls import resolve_attack
    from core.mech.combat_models import AttackOutcome, ResourceChange

    # Lazy import to avoid circular dependency
    if apply_damage_func is None:
        from core.mech.combat_execution import apply_damage
        apply_damage_func = apply_damage

    effects: list[dict] = []

    # Get weapon profile and tags
    weapon_profile = _resolve_weapon_profile(weapon_id, profile_id)
    weapon_tags = list(weapon_profile.tags) if weapon_profile else []

    # Extract relevant weapon properties
    accuracy_bonus = sum(1 for tag in weapon_tags if tag.tag == "accurate")
    difficulty_bonus = sum(1 for tag in weapon_tags if tag.tag == "inaccurate")
    armor_piercing = _extract_tag_value(weapon_tags, "ap") or 0
    reliable_value = _extract_tag_value(weapon_tags, "reliable")
    has_overkill = any(tag.tag == "overkill" for tag in weapon_tags)
    smart_attack = any(tag.tag == "smart" for tag in weapon_tags)

    # Get attack bonus from attacker's grit
    attack_bonus = attacker.stats.grit if attacker.stats else 0

    # Determine if attack is ranged
    is_ranged_attack = True
    if weapon_profile is not None:
        for range_entry in weapon_profile.ranges:
            if range_entry.range_type == "threat":
                is_ranged_attack = False
                break

    # Get target defense
    target_defense = target.stats.e_defense if smart_attack else target.stats.evasion
    if target.stats is None:
        target_defense = 8 if smart_attack else 10

    # Get attacker status modifiers
    attacker_acc_mod, attacker_diff_mod = _get_attacker_status_modifiers(attacker)

    # Get target status modifiers
    target_acc_mod, target_diff_mod, has_lock_on = _get_target_status_modifiers(
        target, is_ranged_attack
    )

    # Get cover modifier for ranged attacks
    cover_difficulty = 0
    if is_ranged_attack:
        cover_difficulty, cover_info = _get_cover_modifier(scenario, attacker, target)
        if cover_info is not None:
            effects.append(cover_info)

    # Combine all accuracy/difficulty modifiers
    final_accuracy_bonus = accuracy_bonus + attacker_acc_mod + target_acc_mod
    final_difficulty_bonus = (
        difficulty_bonus + attacker_diff_mod + target_diff_mod + cover_difficulty
    )

    # Resolve attack roll
    attack_result = resolve_attack(
        attack_bonus=attack_bonus,
        target_defense=target_defense,
        accuracy_bonus=final_accuracy_bonus,
        difficulty_bonus=final_difficulty_bonus,
    )

    # Check for invisibility miss (50% miss chance)
    if attack_result.hit and _check_invisibility_miss(target):
        attack_result = attack_result.model_copy(update={"hit": False})
        effects.append({
            "type": "invisibility_miss",
            "target_id": target.id,
            "reason": "50% miss chance from invisible status",
        })

    # Record attack effect
    effects.append({
        "type": "attack",
        "target_id": target.id,
        "roll": attack_result.roll,
        "total": attack_result.total_accuracy,
        "hit": attack_result.hit,
        "critical": attack_result.is_critical,
        "accuracy_bonus": final_accuracy_bonus,
        "difficulty_bonus": final_difficulty_bonus,
        "status_modifiers": {
            "attacker_acc": attacker_acc_mod,
            "attacker_diff": attacker_diff_mod,
            "target_acc": target_acc_mod,
            "target_diff": target_diff_mod,
            "cover_diff": cover_difficulty,
        },
    })

    # Initialize outcome tracking
    damage_dealt = 0
    resource_change: ResourceChange | None = None
    structure_check: dict | None = None

    # Apply damage if hit
    if attack_result.hit:
        # Roll damage (overkill heat not tracked for reactions)
        base_damage, _overkill_heat = _roll_weapon_damage(
            weapon_profile, apply_overkill=has_overkill
        )

        # Critical = 2x damage
        final_damage = base_damage * 2 if attack_result.is_critical else base_damage

        # Apply exposed damage multiplier
        if "exposed" in target.statuses:
            final_damage = final_damage * 2
            effects.append({
                "type": "exposed_multiplier",
                "target_id": target.id,
                "multiplier": 2,
            })

        # Reliable minimum
        if reliable_value is not None and final_damage < reliable_value:
            final_damage = reliable_value

        # Shredded ignores armor
        effective_ap = armor_piercing
        if "shredded" in target.statuses:
            effective_ap = target.stats.armor if target.stats else 0
            effects.append({
                "type": "shredded_armor_bypass",
                "target_id": target.id,
                "armor_bypassed": effective_ap,
            })

        # Apply damage
        scenario, change, structure_result = apply_damage_func(
            scenario, target.id, final_damage, effective_ap
        )
        resource_change = change
        damage_dealt = abs(change.hp_change or 0)

        if structure_result:
            structure_check = {
                "type": "structure_check",
                "target_id": target.id,
                "outcome": structure_result.outcome,
                "mech_destroyed": structure_result.mech_destroyed,
                "statuses": [str(s) for s in structure_result.statuses_to_apply],
                "dice_rolls": structure_result.dice_rolls,
                "lowest_roll": structure_result.lowest_roll,
            }

        # Consume lock-on if it was used
        if has_lock_on:
            scenario = _remove_status_from_target(scenario, target.id, "lock_on")
            effects.append({
                "type": "lock_on_consumed",
                "target_id": target.id,
            })

    return scenario, AttackOutcome(
        hit=attack_result.hit,
        critical=attack_result.is_critical,
        damage_dealt=damage_dealt,
        roll=attack_result.roll,
        total=attack_result.total_accuracy,
        target_defense=target_defense,
        accuracy_bonus=final_accuracy_bonus,
        difficulty_bonus=final_difficulty_bonus,
        effects=effects,
        resource_change=resource_change,
        structure_check=structure_check,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Weapon Resolution
    "_resolve_weapon_profile",
    "_extract_tag_value",
    "_extract_area_pattern",
    "_roll_damage_with_overkill",
    "_roll_weapon_damage",
    # Range and LOS Validation
    "_get_weapon_range",
    "_get_thrown_range",
    "_has_weapon_tag",
    "_is_melee_weapon",
    "_validate_attack_range_and_los",
    # Weapon Tag Enforcement
    "_get_weapon_state",
    "_validate_weapon_usable",
    "_update_weapon_after_attack",
    "_reload_all_loading_weapons",
    # System Validation
    "_get_system_state",
    "_validate_system_usable",
    # Tech Actions
    "_build_full_tech_option",
    "_apply_tech_result",
    # Status Helpers
    "_record_statuses_applied",
    "_apply_statuses_to_target",
    "_remove_status_from_target",
    "_get_basic_available_actions",
    # Status Duration & Clearing
    "_apply_status_with_duration",
    "_clear_statuses_by_trigger",
    "_expire_turn_duration_statuses",
    "_sync_statuses_from_instances",
    # Attack Modifiers
    "_get_attacker_status_modifiers",
    "_get_target_status_modifiers",
    "_check_invisibility_miss",
    "_get_cover_modifier",
    # Action Resolution
    "_resolve_stabilize",
    "_resolve_hide",
    "_resolve_ram",
    "_apply_knockback_on_hit",
    "_resolve_grapple",
    "_resolve_search",
    "_resolve_burn_tick",
    # Movement Resolution
    "_apply_engagement_status",
    "_resolve_movement",
    # Mount/Dismount/Eject
    "_resolve_mount",
    "_resolve_dismount",
    "_resolve_eject",
    # Single Attack Resolution
    "resolve_single_attack",
]
