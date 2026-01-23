"""Combat execution helper functions.

Private implementation helpers for combat resolution.
These are internal functions used by combat_execution.py.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from core.shared.enums import StatusType, DamageType, AttackType
from core.shared.dice import roll_dice, round_up
from core.shared.damage import DamageBreakdown
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
from core.mech.grid import (
    HexCoord,
    HexPosition,
    hex_add,
    hex_scale,
    adjacency_distance,
    is_adjacent_by_size,
    hexes_in_radius,
)
from core.mech.compendium import get_weapon_definition
from core.mech.combat_rules import AttackPatternDefinition
from core.mech.weapon import WeaponProfile, WeaponTag, WeaponDamageType, resolve_weapon_profile
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
from core.shared.effects import (
    MechanicalEffect,
    AccuracyModifier,
    CheckModifierEffect,
    EffectCondition,
    SpatialCondition,
    AttackContextCondition,
    CheckContextCondition,
    SizeCondition,
    ReactionCondition,
    ConditionGroup,
)
from core.shared.involuntary_movement import resolve_knockback
from core.shared.los import LOSCheckRequest, check_line_of_sight
from core.shared.movement import check_engagement_stop, _is_hostile
from core.shared.hide_search import get_cover_for_hiding

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
    profile_id: str | None = None,
    force_threat: bool = False,
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
        # Overwatch can force threat range even for ranged weapons
        if force_threat:
            required_range = _get_weapon_range(
                weapon_id, is_melee=True, profile_id=profile_id
            )
            range_type = "threat"
        else:
            # Check if melee or ranged
            is_melee = _is_melee_weapon(weapon_id, profile_id)
            if use_thrown and not is_melee:
                return (False, "Only melee weapons can be thrown")

            if is_melee:
                threat_range = _get_weapon_range(
                    weapon_id, is_melee=True, profile_id=profile_id
                )
                if use_thrown:
                    thrown_range = _get_thrown_range(weapon_id, profile_id)
                    if thrown_range is None:
                        return (False, "Weapon has no thrown range")
                    required_range = thrown_range
                    range_type = "thrown"
                else:
                    required_range = threat_range
                    range_type = "threat"
            else:
                required_range = _get_weapon_range(
                    weapon_id, is_melee=False, profile_id=profile_id
                )
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


def _validate_blast_origin(
    scenario: MechCombatScenario,
    attacker: CombatantState,
    blast_origin: HexPosition,
    weapon_id: str | None,
    profile_id: str | None = None,
) -> tuple[bool, str | None]:
    """Validate blast origin is within weapon range and line of sight.

    Per PR2 3993-3994: Blast is drawn from a point in range AND line of sight.

    Args:
        scenario: Current combat scenario
        attacker: The attacking combatant
        blast_origin: The intended blast origin position
        weapon_id: Weapon ID being used
        profile_id: Optional weapon profile ID

    Returns:
        Tuple of (valid, error_message)
    """
    if attacker.position is None:
        return (False, "Attacker has no position")

    # Calculate distance to blast origin
    distance = attacker.position.coord.distance_to(blast_origin.coord)

    # Get weapon range
    weapon_range = _get_weapon_range(weapon_id, is_melee=False, profile_id=profile_id)

    # Check range
    if distance > weapon_range:
        return (False, f"Blast origin out of range ({distance} > {weapon_range})")

    # Check for seeking/arcing tags that bypass LOS
    has_seeking = _has_weapon_tag(weapon_id, "seeking", profile_id)
    has_arcing = _has_weapon_tag(weapon_id, "arcing", profile_id)

    # Check LOS to blast origin
    los_request = LOSCheckRequest(
        attacker_pos=attacker.position,
        target_pos=blast_origin,
        terrain=scenario.terrain,
    )
    los_result = check_line_of_sight(los_request)

    if los_result.los_type == "blocked":
        if has_seeking:
            pass  # Seeking bypasses LOS
        elif has_arcing:
            pass  # Arcing bypasses LOS
        else:
            return (False, "No line of sight to blast origin")

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


def _roll_damage_dice_critical(
    dice_expr,
    is_critical: bool,
    apply_overkill: bool,
) -> tuple[list[int], int]:
    """Roll damage dice with critical hit 'roll twice, pick highest' mechanic.

    Per PR2 3965-3969: On critical, roll all dice twice and pick the highest N.

    Args:
        dice_expr: The dice expression (e.g., 2d6)
        is_critical: Whether this is a critical hit
        apply_overkill: Whether to apply overkill rerolls (reroll 1s, gain heat)

    Returns:
        Tuple of (selected_rolls, overkill_heat)
    """
    if not is_critical:
        return _roll_damage_with_overkill(dice_expr, apply_overkill)

    # Roll each die twice (total 2N dice)
    first_rolls, first_heat = _roll_damage_with_overkill(dice_expr, apply_overkill)
    second_rolls, second_heat = _roll_damage_with_overkill(dice_expr, apply_overkill)

    # Combine and sort descending
    all_rolls = first_rolls + second_rolls
    all_rolls.sort(reverse=True)

    # Pick highest N dice (where N = original dice count)
    selected_rolls = all_rolls[: dice_expr.count]
    total_heat = first_heat + second_heat

    return selected_rolls, total_heat


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


def _roll_weapon_damage_components(
    profile: WeaponProfile | None,
    apply_overkill: bool,
) -> tuple[list[tuple[WeaponDamageType, int]], int]:
    """Roll weapon damage per component and return (components, overkill_heat)."""
    if profile is None:
        return [("kinetic", 6)], 0

    components: list[tuple[WeaponDamageType, int]] = []
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
        components.append((damage_component.damage_type, component_total))

    if not components:
        default_type = profile.damage_type or "kinetic"
        components.append((default_type, 6))

    return components, overkill_heat


def _roll_weapon_damage_components_critical(
    profile: WeaponProfile | None,
    is_critical: bool,
    apply_overkill: bool,
) -> tuple[list[tuple[WeaponDamageType, int]], int]:
    """Roll weapon damage per component with critical hit handling.

    For crits: rolls each die twice, picks highest N from 2N rolls.
    Flat bonuses and modifiers are NOT doubled (per PR2 3965-3969).

    Args:
        profile: The weapon profile to roll damage for
        is_critical: Whether this is a critical hit
        apply_overkill: Whether to apply overkill rerolls

    Returns:
        Tuple of (damage components, overkill_heat)
    """
    if profile is None:
        return [("kinetic", 6)], 0

    components: list[tuple[WeaponDamageType, int]] = []
    overkill_heat = 0

    for damage_component in profile.damage:
        component_total = 0
        if damage_component.dice is not None:
            rolls, component_overkill = _roll_damage_dice_critical(
                damage_component.dice,
                is_critical,
                apply_overkill,
            )
            # Sum selected dice + modifier (modifier NOT doubled)
            component_total += sum(rolls) + damage_component.dice.modifier
            overkill_heat += component_overkill
        # Flat damage is NOT doubled on crit
        component_total += damage_component.flat
        components.append((damage_component.damage_type, component_total))

    if not components:
        default_type = profile.damage_type or "kinetic"
        components.append((default_type, 6))

    return components, overkill_heat


def _get_primary_damage_type(profile: WeaponProfile | None) -> DamageType:
    """Pick a default damage type for reliable damage and bonuses."""
    if profile is None:
        return "kinetic"

    if profile.damage_type in ("kinetic", "explosive", "energy", "burn"):
        return profile.damage_type

    for component in profile.damage:
        if component.damage_type != "heat":
            return component.damage_type

    return "kinetic"


def _collect_active_effects(combatant: CombatantState) -> list[MechanicalEffect]:
    """Collect all active mechanical effects from a combatant."""
    effects: list[MechanicalEffect] = []
    effects.extend(combatant.talent_effects)
    effects.extend(combatant.frame_trait_effects)
    for mode in combatant.active_mode_effects:
        effects.append(mode.effects)
    if combatant.core_power_active and combatant.core_power_effects:
        effects.append(combatant.core_power_effects)
    return effects


def _build_damage_context(
    attacker: CombatantState | None,
    target: CombatantState,
    is_melee: bool,
    is_ranged: bool,
    is_tech: bool,
    extra_context: dict | None = None,
) -> dict:
    """Build an EffectCondition context for incoming damage resolution."""
    ctx: dict = {
        "is_melee": is_melee,
        "is_ranged": is_ranged,
        "is_tech": is_tech,
        "attack_type": "tech" if is_tech else ("ranged" if is_ranged else "melee"),
        "is_incoming": True,
        "is_outgoing": False,
        "is_engaged": "engaged" in target.statuses,
        "structure_remaining": target.resources.structure_current,
        "structure_1_or_less": target.resources.structure_current <= 1,
    }

    if attacker and attacker.position and target.position:
        distance = attacker.position.coord.distance_to(target.position.coord)
        ctx["attack_range"] = distance
        # Use size-aware adjacency if stats available
        if attacker.stats and target.stats:
            ctx["is_adjacent"] = is_adjacent_by_size(
                attacker.position.coord,
                target.position.coord,
                attacker.stats.size,
                target.stats.size,
            )
        else:
            ctx["is_adjacent"] = distance == 1

    if attacker and attacker.stats and target.stats:
        size_map = {
            "size_half": 0.5,
            "size_1": 1,
            "size_2": 2,
            "size_3": 3,
            "size_4": 4,
            "size_5": 5,
        }
        attacker_size = size_map.get(attacker.stats.size, 1)
        target_size = size_map.get(target.stats.size, 1)
        ctx["actor_size"] = attacker_size
        ctx["target_size"] = target_size
        ctx["target_larger"] = target_size > attacker_size
        ctx["target_smaller"] = target_size < attacker_size

    if extra_context:
        ctx.update(extra_context)

    return ctx


def _collect_damage_resistances(
    target: CombatantState,
    context: dict,
) -> list[DamageType]:
    """Collect active damage resistances for a target."""
    resistances: set[DamageType] = set()
    for effect in _collect_active_effects(target):
        for resistance in effect.resistances:
            if resistance.target not in ("self", "all"):
                continue
            if not _evaluate_condition(resistance.condition, context):
                continue
            if resistance.damage_type == "all":
                resistances.update(["kinetic", "explosive", "energy", "burn"])
            else:
                resistances.add(resistance.damage_type)
    return list(resistances)


def _collect_heat_resistance_multiplier(
    target: CombatantState,
    context: dict,
) -> float:
    """Collect the strongest heat resistance multiplier for a target."""
    multiplier = 1.0
    for effect in _collect_active_effects(target):
        for heat_resistance in effect.heat_resistances:
            if heat_resistance.target not in ("self", "all"):
                continue
            if not _evaluate_condition(heat_resistance.condition, context):
                continue
            multiplier = min(multiplier, heat_resistance.multiplier)
    return multiplier


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
            tech_attack_bonus=attacker_systems,
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
            "attack_roll": result.attack_roll,
            "attack_bonus": result.attack_bonus,
            "total": result.total,
            "target_e_defense": result.target_e_defense,
            "hit": result.hit,
            "is_critical": result.is_critical,
        })

        if result.hit and result.heat_applied:
            target = next(
                (c for c in scenario.combatants if c.id == result.target_id), None
            )
            actor = next(
                (c for c in scenario.combatants if c.id == result.actor_id), None
            )
            heat_multiplier = 1.0
            if target is not None:
                heat_context = _build_damage_context(
                    actor,
                    target,
                    is_melee=False,
                    is_ranged=False,
                    is_tech=True,
                )
                heat_multiplier = _collect_heat_resistance_multiplier(
                    target, heat_context
                )
                if "shredded" in target.statuses:
                    heat_multiplier = 1.0
            adjusted_heat = round_up(result.heat_applied * heat_multiplier)

            scenario, change, overheat_result = apply_heat_func(
                scenario,
                result.target_id,
                result.heat_applied,
                heat_resistance_multiplier=heat_multiplier,
            )
            resource_changes.append(change)
            heat_generated += adjusted_heat

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
        sitrep_resolution=scenario.sitrep_resolution,
        pending_decisions=list(scenario.pending_decisions),
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
        sitrep_resolution=scenario.sitrep_resolution,
        pending_decisions=list(scenario.pending_decisions),
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
        sitrep_resolution=scenario.sitrep_resolution,
        pending_decisions=list(scenario.pending_decisions),
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
        sitrep_resolution=scenario.sitrep_resolution,
        pending_decisions=list(scenario.pending_decisions),
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
        sitrep_resolution=scenario.sitrep_resolution,
        pending_decisions=list(scenario.pending_decisions),
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
# Talent Effect Helpers (Phase 32)
# =============================================================================


def _evaluate_condition(condition: EffectCondition | None, context: dict) -> bool:
    """Evaluate a condition against the current combat context.

    Handles all EffectCondition types:
    - None: Always true
    - str: Simple condition string lookup
    - SpatialCondition: Check spatial relationships
    - AttackContextCondition: Check attack context
    - CheckContextCondition: Check skill/save context
    - SizeCondition: Check size comparisons
    - ReactionCondition: Check reaction triggers
    - ConditionGroup: Combine conditions with AND/OR

    Args:
        condition: The condition to evaluate
        context: Dictionary with context like {"is_melee": True, "is_first_attack": True}

    Returns:
        True if condition is met or None, False otherwise
    """
    if condition is None:
        return True

    # Handle string conditions
    if isinstance(condition, str):
        # Map known condition strings to context checks
        condition_checks = {
            "engaged": lambda c: c.get("is_engaged", False),
            "while_flying": lambda c: c.get("is_flying", False),
            "first_melee": lambda c: c.get("is_first_melee", False),
            "first_ranged": lambda c: c.get("is_first_ranged", False),
            "melee_attack": lambda c: c.get("is_melee", False),
            "ranged_attack": lambda c: c.get("is_ranged", False),
            "tech_attack": lambda c: c.get("is_tech", False),
            "natural_20": lambda c: c.get("is_nat_20", False),
            "in_danger_zone": lambda c: c.get("in_danger_zone", False),
            "bondmate_adjacent": lambda c: c.get("bondmate_adjacent", False),
            "target_larger": lambda c: c.get("target_larger", False),
            "target_smaller": lambda c: c.get("target_smaller", False),
            "attacks_within_range_3": lambda c: (
                c.get("attack_range") is not None and c.get("attack_range") <= 3
            ),
            "benefiting_from_trail_cover": lambda c: c.get(
                "benefiting_from_trail_cover", False
            ),
            "from_triggering_attack": lambda c: c.get("from_triggering_attack", False),
            "structure_1_or_less": lambda c: c.get("structure_remaining", 99) <= 1,
            "can_see_bondmate": lambda c: c.get("can_see_bondmate", False),
        }

        if condition in condition_checks:
            return condition_checks[condition](context)

        # Unknown string condition - conservatively return False
        return False

    # Handle SpatialCondition
    if isinstance(condition, SpatialCondition):
        # Check spatial relationship against context
        rel = condition.relation
        if rel == "adjacent":
            return context.get("is_adjacent", False)
        if rel == "within_range":
            return context.get("within_range", False)
        if rel == "engaged":
            return context.get("is_engaged", False)
        return False

    # Handle AttackContextCondition
    if isinstance(condition, AttackContextCondition):
        # Check attack context (attack_types is a list)
        if condition.attack_types:
            ctx_attack_type = context.get("attack_type")
            if ctx_attack_type not in condition.attack_types:
                return False
        # Check applies_to direction
        applies_to = condition.applies_to
        if applies_to == "outgoing" and not context.get("is_outgoing", True):
            return False
        if applies_to == "incoming" and not context.get("is_incoming", False):
            return False
        return True

    # Handle CheckContextCondition
    if isinstance(condition, CheckContextCondition):
        # Check check kinds (check_kinds is a list)
        if condition.check_kinds:
            ctx_check_kind = context.get("check_kind")
            if ctx_check_kind not in condition.check_kinds:
                return False
        # Check save types (saves is a list)
        if condition.saves:
            ctx_save_type = context.get("save_type")
            if ctx_save_type not in condition.saves:
                return False
        return True

    # Handle SizeCondition
    if isinstance(condition, SizeCondition):
        # Get the size to compare based on subject
        if condition.subject == "self":
            subject_size = context.get("actor_size", 1)
        elif condition.subject == "target":
            subject_size = context.get("target_size", 1)
        else:  # source
            subject_size = context.get("source_size", 1)

        # Convert size class to numeric for comparison
        size_map = {
            "size_half": 0.5, "size_1": 1, "size_2": 2,
            "size_3": 3, "size_4": 4, "size_5": 5
        }
        condition_size = size_map.get(condition.size, 1)

        # Apply comparator
        if condition.comparator == "lt":
            return subject_size < condition_size
        if condition.comparator == "lte":
            return subject_size <= condition_size
        if condition.comparator == "gt":
            return subject_size > condition_size
        if condition.comparator == "gte":
            return subject_size >= condition_size
        if condition.comparator == "eq":
            return subject_size == condition_size
        return False

    # Handle ReactionCondition
    if isinstance(condition, ReactionCondition):
        # Check reaction_id
        if condition.reaction_id:
            return context.get("reaction_id") == condition.reaction_id
        # Check is_attack
        if condition.is_attack is not None:
            return context.get("is_attack_reaction") == condition.is_attack
        return True

    # Handle ConditionGroup
    if isinstance(condition, ConditionGroup):
        # all_of: all conditions must be met
        if condition.all_of:
            if not all(_evaluate_condition(c, context) for c in condition.all_of):
                return False
        # any_of: at least one condition must be met
        if condition.any_of:
            if not any(_evaluate_condition(c, context) for c in condition.any_of):
                return False
        # none_of: no conditions must be met
        if condition.none_of:
            if any(_evaluate_condition(c, context) for c in condition.none_of):
                return False
        return True

    # Unknown condition type - conservatively return False
    return False


def _get_talent_accuracy_modifiers(
    actor: CombatantState,
    is_melee: bool = False,
    is_ranged: bool = False,
    is_tech: bool = False,
    context: dict | None = None,
) -> tuple[int, int]:
    """Get accuracy and difficulty modifiers from talent effects.

    Evaluates accuracy_mods from all talent effects on the actor.

    Args:
        actor: The combatant whose talents to evaluate
        is_melee: Whether this is a melee attack
        is_ranged: Whether this is a ranged attack
        is_tech: Whether this is a tech attack
        context: Additional context for condition evaluation

    Returns:
        Tuple of (accuracy_mod, difficulty_mod)
    """
    accuracy_mod = 0
    difficulty_mod = 0
    ctx = context or {}

    # Add attack type to context
    ctx["is_melee"] = is_melee
    ctx["is_ranged"] = is_ranged
    ctx["is_tech"] = is_tech

    for effect in actor.talent_effects:
        for acc_mod in effect.accuracy_mods:
            # Check if this modifier applies to the attack type
            applies = acc_mod.applies_to == "all"
            if not applies and is_melee and acc_mod.applies_to == "melee":
                applies = True
            if not applies and is_ranged and acc_mod.applies_to == "ranged":
                applies = True
            if not applies and is_tech and acc_mod.applies_to == "tech":
                applies = True

            if not applies:
                continue

            # Check condition
            if not _evaluate_condition(acc_mod.condition, ctx):
                continue

            # Apply modifier (positive = accuracy, negative = difficulty)
            if acc_mod.value > 0:
                accuracy_mod += acc_mod.value
            else:
                difficulty_mod += abs(acc_mod.value)

    # Also check frame trait effects
    for effect in actor.frame_trait_effects:
        for acc_mod in effect.accuracy_mods:
            applies = acc_mod.applies_to == "all"
            if not applies and is_melee and acc_mod.applies_to == "melee":
                applies = True
            if not applies and is_ranged and acc_mod.applies_to == "ranged":
                applies = True
            if not applies and is_tech and acc_mod.applies_to == "tech":
                applies = True

            if not applies:
                continue

            if not _evaluate_condition(acc_mod.condition, ctx):
                continue

            if acc_mod.value > 0:
                accuracy_mod += acc_mod.value
            else:
                difficulty_mod += abs(acc_mod.value)

    # Check active core power effects
    if actor.core_power_active and actor.core_power_effects:
        for acc_mod in actor.core_power_effects.accuracy_mods:
            applies = acc_mod.applies_to == "all"
            if not applies and is_melee and acc_mod.applies_to == "melee":
                applies = True
            if not applies and is_ranged and acc_mod.applies_to == "ranged":
                applies = True
            if not applies and is_tech and acc_mod.applies_to == "tech":
                applies = True

            if not applies:
                continue

            if not _evaluate_condition(acc_mod.condition, ctx):
                continue

            if acc_mod.value > 0:
                accuracy_mod += acc_mod.value
            else:
                difficulty_mod += abs(acc_mod.value)

    return accuracy_mod, difficulty_mod


def _get_talent_check_modifiers(
    actor: CombatantState,
    check_type: str,
    check_kind: str = "check",
    context: dict | None = None,
) -> tuple[int, int]:
    """Get accuracy and difficulty modifiers from talent effects for checks/saves.

    Evaluates check_mods from all talent effects on the actor.

    Args:
        actor: The combatant whose talents to evaluate
        check_type: Type of check (e.g., "hull", "agility", "systems", "engineering")
        check_kind: Kind of check ("check" or "save")
        context: Additional context for condition evaluation

    Returns:
        Tuple of (accuracy_mod, difficulty_mod)
    """
    accuracy_mod = 0
    difficulty_mod = 0
    ctx = context or {}

    for effect in actor.talent_effects:
        for check_mod in effect.check_mods:
            # Check if this modifier applies to this check type
            if check_mod.check_types and check_type not in check_mod.check_types:
                continue
            # Check if this modifier applies to this check kind
            if check_mod.check_kinds and check_kind not in check_mod.check_kinds:
                continue
            # Check condition
            if not _evaluate_condition(check_mod.condition, ctx):
                continue

            # Apply modifier
            if check_mod.value > 0:
                accuracy_mod += check_mod.value
            else:
                difficulty_mod += abs(check_mod.value)

    # Also check frame trait effects
    for effect in actor.frame_trait_effects:
        for check_mod in effect.check_mods:
            if check_mod.check_types and check_type not in check_mod.check_types:
                continue
            if check_mod.check_kinds and check_kind not in check_mod.check_kinds:
                continue
            if not _evaluate_condition(check_mod.condition, ctx):
                continue

            if check_mod.value > 0:
                accuracy_mod += check_mod.value
            else:
                difficulty_mod += abs(check_mod.value)

    return accuracy_mod, difficulty_mod


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
                    sitrep_resolution=scenario.sitrep_resolution,
                    pending_decisions=list(scenario.pending_decisions),
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
        sitrep_resolution=scenario.sitrep_resolution,
        pending_decisions=list(scenario.pending_decisions),
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

    # Actor must have a position to check cover
    if actor.position is None:
        return scenario, False, "No position to check for cover"

    # Check for cover at actor's position using terrain
    has_hard_cover, is_in_soft_cover_area, cover_reason = get_cover_for_hiding(
        terrain=scenario.terrain,
        target_coord=actor.position.coord,
    )

    if has_hard_cover:
        return scenario, True, "Hidden behind hard cover"

    if is_in_soft_cover_area:
        return scenario, True, "Hidden in soft cover area"

    return scenario, False, cover_reason or "No cover available for hiding"


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
                sitrep_resolution=scenario.sitrep_resolution,
                pending_decisions=list(scenario.pending_decisions),
            )

    return scenario, effects


@dataclass
class KnockbackCollisionEffects:
    """Effects to apply from knockback collision.

    Per PR2 8642: "If it collides with an obstacle or another mech,
    it stops and is additionally knocked prone"
    """

    target_prone: bool = False
    secondary_target_id: str | None = None
    secondary_prone: bool = False


def _resolve_knockback_collision(
    scenario: MechCombatScenario,
    obstruction_coord: HexCoord | None,
    obstruction_type: Literal["unit", "terrain"] | None,
) -> KnockbackCollisionEffects:
    """Determine effects from knockback collision.

    Per PR2 rules:
    - Collision with obstacle/mech: target knocked prone (PR2 8642)
    - Collision with another character: that character makes HULL save or prone (PR2 6521)
      (Simplified: secondary always knocked prone for now)

    Args:
        scenario: Current combat scenario
        obstruction_coord: Coordinate of obstruction if any
        obstruction_type: Type of obstruction ("unit" or "terrain")

    Returns:
        KnockbackCollisionEffects with effects to apply
    """
    effects = KnockbackCollisionEffects()

    if obstruction_coord is None or obstruction_type is None:
        return effects

    # Target is always knocked prone on collision (PR2 8642)
    effects.target_prone = True

    if obstruction_type == "unit":
        # Find the unit at obstruction coord
        secondary = next(
            (c for c in scenario.combatants
             if c.position and c.position.coord == obstruction_coord),
            None
        )
        if secondary:
            effects.secondary_target_id = str(secondary.id)
            # Secondary must pass HULL save or be knocked prone (PR2 6521)
            # Simplified: always apply prone for now (save logic can be added later)
            effects.secondary_prone = True

    return effects


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
            sitrep_resolution=scenario.sitrep_resolution,
            pending_decisions=list(scenario.pending_decisions),
        )

    effect: dict = {
        "type": "knockback",
        "target_id": str(target.id),
        "spaces_requested": knockback_spaces,
        "spaces_moved": result.spaces_knocked,
        "final_position": {"q": final_position.q, "r": final_position.r},
        "blocked": result.obstructed,
    }

    # Handle collision effects (PR2 8642, 6521)
    if result.obstructed and result.obstruction_coord:
        collision_effects = _resolve_knockback_collision(
            scenario=scenario,
            obstruction_coord=result.obstruction_coord,
            obstruction_type=result.obstruction_type,
        )

        # Apply prone to knocked target
        if collision_effects.target_prone:
            scenario, added = _apply_statuses_to_target(
                scenario, str(target.id), ["prone"]
            )
            if added:
                effect["collision_prone"] = True

        # Apply prone to secondary unit (if knocked into another combatant)
        if collision_effects.secondary_target_id and collision_effects.secondary_prone:
            scenario, added = _apply_statuses_to_target(
                scenario, collision_effects.secondary_target_id, ["prone"]
            )
            if added:
                effect["secondary_prone_target"] = collision_effects.secondary_target_id

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
            sitrep_resolution=scenario.sitrep_resolution,
            pending_decisions=list(scenario.pending_decisions),
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
        sitrep_resolution=scenario.sitrep_resolution,
        pending_decisions=list(scenario.pending_decisions),
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


def _resolve_falling(
    scenario: MechCombatScenario,
    actor: CombatantState,
) -> tuple[MechCombatScenario, list[dict]]:
    """Resolve falling at end of turn (PR2 flight rules).

    Falling damage: 1 damage per altitude level fallen (armor-piercing).
    After falling, actor lands at elevation 0 and is no longer flying.

    Args:
        scenario: Current combat scenario
        actor: The combatant whose turn is ending

    Returns:
        Tuple of (updated scenario, list of falling effect dicts)
    """
    from core.shared.flying import calculate_fall_damage, FlyingStatus

    effects: list[dict] = []

    # Check if actor is in falling state
    if "falling" not in actor.statuses or actor.falling_from_altitude is None:
        return scenario, effects

    # Find actor index
    actor_idx = next((i for i, c in enumerate(scenario.combatants) if c.id == actor.id), -1)
    if actor_idx < 0:
        return scenario, effects

    # Calculate fall damage
    fall_result = calculate_fall_damage(actor.falling_from_altitude)

    updated_resources = actor.resources
    if fall_result.damage_taken > 0:
        # Apply damage (armor-piercing per FallingRules)
        new_hp = max(0, actor.resources.hp_current - fall_result.damage_taken)
        updated_resources = actor.resources.model_copy(update={"hp_current": new_hp})

        effects.append({
            "type": "falling_damage",
            "target_id": str(actor.id),
            "damage": fall_result.damage_taken,
            "fell_from_altitude": actor.falling_from_altitude,
            "armor_piercing": True,
        })

    # Clear falling status, reset altitude to 0
    new_statuses = [s for s in actor.statuses if s != "falling"]

    # Reset flying status to grounded
    updated_flying_status: FlyingStatus | None = None
    if actor.flying_status is not None:
        updated_flying_status = actor.flying_status.model_copy(update={
            "is_flying": False,
            "altitude_level": 0,
            "movement_mode": "ground",
        })
    else:
        # Create a grounded flying status if none exists
        updated_flying_status = FlyingStatus(
            is_flying=False,
            altitude_level=0,
            is_hover=False,
            movement_mode="ground",
        )

    # Update position elevation to 0
    updated_position = actor.position
    if actor.position is not None:
        updated_position = actor.position.model_copy(update={"elevation": 0})

    # Update actor
    updated_actor = actor.model_copy(update={
        "resources": updated_resources,
        "statuses": new_statuses,
        "falling_from_altitude": None,
        "flying_status": updated_flying_status,
        "position": updated_position,
    })

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
        sitrep_resolution=scenario.sitrep_resolution,
        pending_decisions=list(scenario.pending_decisions),
    )

    effects.append({
        "type": "fall_resolved",
        "target_id": str(actor.id),
        "landed_at_elevation": 0,
    })

    return scenario, effects


# =============================================================================
# Movement Resolution
# =============================================================================


def _apply_engagement_status(
    scenario: MechCombatScenario,
    mover: CombatantState,
    final_position: HexPosition,
    ignore_engagement: bool = False,
) -> tuple[MechCombatScenario, list[dict]]:
    """Refresh engagement statuses after movement.

    Per PR2 3817-3819, engagement is adjacency-driven; moving away should
    clear engaged status. Disengage prevents engagement with the mover.
    """
    effects: list[dict] = []
    updated_combatants = list(scenario.combatants)
    ignore_id = mover.id if ignore_engagement else None

    for idx, combatant in enumerate(scenario.combatants):
        if combatant.position is None:
            is_engaged = False
        elif ignore_id is not None and combatant.id == ignore_id:
            is_engaged = False
        else:
            is_engaged = False
            for other in scenario.combatants:
                if other.id == combatant.id:
                    continue
                if other.position is None:
                    continue
                if ignore_id is not None and other.id == ignore_id:
                    continue
                if not _is_hostile(combatant.id, other.id, scenario):
                    continue
                if combatant.position.coord.distance_to(other.position.coord) == 1:
                    is_engaged = True
                    break

        current_statuses = list(updated_combatants[idx].statuses)
        if is_engaged and "engaged" not in current_statuses:
            current_statuses.append("engaged")
            effects.append({
                "type": "status_applied",
                "target_id": str(combatant.id),
                "status": "engaged",
                "reason": "adjacent_to_hostile",
            })
        if not is_engaged and "engaged" in current_statuses:
            current_statuses = [s for s in current_statuses if s != "engaged"]
            effects.append({
                "type": "status_removed",
                "target_id": str(combatant.id),
                "status": "engaged",
                "reason": "no_adjacent_hostiles",
            })

        updated_combatants[idx] = updated_combatants[idx].model_copy(
            update={"statuses": current_statuses}
        )

    updated_scenario = MechCombatScenario(
        combatants=updated_combatants,
        grapples=list(scenario.grapples),
        rounds=list(scenario.rounds),
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables=dict(scenario.deployables),
        sitrep_resolution=scenario.sitrep_resolution,
        pending_decisions=list(scenario.pending_decisions),
    )

    return updated_scenario, effects


def _check_and_break_grapples_on_movement(
    scenario: MechCombatScenario,
    mover_id: str,
    new_position: HexCoord,
) -> tuple[MechCombatScenario, list[dict]]:
    """Check if movement breaks any grapples due to adjacency loss.

    Per PR2 4175: Grapple breaks if parties become non-adjacent.

    Args:
        scenario: Current combat scenario
        mover_id: ID of the combatant that moved
        new_position: The new position of the mover

    Returns:
        Tuple of (updated scenario, effects list)
    """
    effects: list[dict] = []
    grapples_to_remove: list[int] = []

    for idx, grapple in enumerate(scenario.grapples):
        # Check if the mover is part of this grapple
        if grapple.grappler_id != mover_id and grapple.target_id != mover_id:
            continue

        # Find the other party in the grapple
        other_id = grapple.target_id if grapple.grappler_id == mover_id else grapple.grappler_id
        other = next((c for c in scenario.combatants if c.id == other_id), None)

        if other is None or other.position is None:
            continue

        # Get sizes for adjacency check
        mover = next((c for c in scenario.combatants if c.id == mover_id), None)
        mover_size = mover.stats.size if mover and mover.stats else "size_1"
        other_size = other.stats.size if other.stats else "size_1"

        # Check if still adjacent
        still_adjacent = is_adjacent_by_size(
            new_position,
            other.position.coord,
            mover_size,
            other_size,
        )

        if not still_adjacent:
            grapples_to_remove.append(idx)
            effects.append({
                "type": "grapple_broken",
                "reason": "adjacency_lost",
                "grappler_id": grapple.grappler_id,
                "target_id": grapple.target_id,
                "mover_id": mover_id,
            })

    if grapples_to_remove:
        updated_grapples = [
            g for i, g in enumerate(scenario.grapples)
            if i not in grapples_to_remove
        ]
        scenario = MechCombatScenario(
            combatants=list(scenario.combatants),
            grapples=updated_grapples,
            rounds=list(scenario.rounds),
            terrain=scenario.terrain,
            environment=scenario.environment,
            deployables=dict(scenario.deployables),
            sitrep_resolution=scenario.sitrep_resolution,
            pending_decisions=list(scenario.pending_decisions),
        )

    return scenario, effects


def _resolve_movement(
    scenario: MechCombatScenario,
    actor: CombatantState,
    path: list[HexPosition],
    is_boost: bool = False,
    apply_damage_func=None,
    ignore_engagement: bool = False,
    prompt_dangerous_terrain: bool = False,
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
        prompt_dangerous_terrain: Whether to create a decision instead of auto-rolling (players only)

    Returns:
        Tuple of (updated scenario, effects list)
    """
    from core.shared.terrain import (
        terrain_index,
        calculate_movement_cost,
        resolve_dangerous_terrain,
    )
    from core.mech.combat_rules import DEFAULT_MECH_COMBAT_RULES

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
    current_round = scenario.rounds[-1].round_index if scenario.rounds else 1
    terrain_rules = DEFAULT_MECH_COMBAT_RULES.terrain
    check_once_per_round = terrain_rules.dangerous_terrain_check_once_per_round
    checked_this_round = (
        check_once_per_round
        and actor.dangerous_terrain_last_check_round == current_round
    )
    last_check_round = actor.dangerous_terrain_last_check_round

    for hex_pos in path:
        terrain_hex = terrain_idx.get(hex_pos.coord)
        if not terrain_hex or not terrain_hex.dangerous:
            continue

        if check_once_per_round and checked_this_round:
            continue

        skill_bonus = actor.stats.engineering_skill if actor.stats else 0

        if (
            prompt_dangerous_terrain
            and actor.side == "players"
            and not actor.ai_controlled
        ):
            from core.shared.decisions import (
                add_decision_to_scenario,
                check_dangerous_terrain_decision,
            )

            decision = check_dangerous_terrain_decision(
                combatant=actor,
                terrain_name="dangerous",
                check_target=10,
                current_round=current_round,
            )
            scenario = add_decision_to_scenario(scenario, decision)

            if check_once_per_round:
                checked_this_round = True
                last_check_round = current_round
            continue

        danger_result = resolve_dangerous_terrain(
            terrain=scenario.terrain,
            coord=hex_pos.coord,
            skill_bonus=skill_bonus,
            damage=terrain_rules.dangerous_terrain_damage,
            damage_type=terrain_rules.dangerous_terrain_damage_type,
            check_once_per_round=check_once_per_round,
            round_checked=current_round,
        )

        effects.append({
            "type": "dangerous_terrain",
            "coord": {"q": hex_pos.coord.q, "r": hex_pos.coord.r},
            "check_passed": danger_result.check_passed,
            "roll": danger_result.roll_result,
            "damage": danger_result.damage_dealt,
        })

        if check_once_per_round:
            checked_this_round = True
            last_check_round = current_round

        # Apply damage if check failed
        if danger_result.check_passed is False and danger_result.damage_dealt > 0:
            if apply_damage_func is not None:
                scenario, _change, _structure_result = apply_damage_func(
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
        updated_actor = next(
            (c for c in scenario.combatants if c.id == actor.id), actor
        )
        update_fields = {"position": final_position}
        if last_check_round is not None:
            update_fields["dangerous_terrain_last_check_round"] = last_check_round
        updated_actor = updated_actor.model_copy(update=update_fields)

        # Auto-retrieve thrown weapons when moving adjacent to their location.
        if updated_actor.inventory is not None:
            updated_mounts = []
            for mount in updated_actor.inventory.mounts:
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
            sitrep_resolution=scenario.sitrep_resolution,
            pending_decisions=list(scenario.pending_decisions),
        )

    # Linked grapple movement - smaller party moves with larger party (PR2 4168)
    # "The smaller party is immobilized, but moves when the larger party moves."
    from core.shared.grapple import SIZE_ORDER

    for grapple in scenario.grapples:
        # Check if actor is part of this grapple
        if grapple.grappler_id != actor.id and grapple.target_id != actor.id:
            continue

        # Find the other party
        other_id = grapple.target_id if grapple.grappler_id == actor.id else grapple.grappler_id
        other_idx = next((i for i, c in enumerate(scenario.combatants) if c.id == other_id), None)
        if other_idx is None:
            continue
        other = scenario.combatants[other_idx]
        if other.position is None:
            continue

        # Re-fetch the updated actor from scenario
        updated_actor_for_grapple = next(
            (c for c in scenario.combatants if c.id == actor.id), None
        )
        if updated_actor_for_grapple is None:
            continue

        # Check sizes - only drag if actor is LARGER
        actor_size = updated_actor_for_grapple.stats.size if updated_actor_for_grapple.stats else "size_1"
        other_size = other.stats.size if other.stats else "size_1"

        actor_size_val = SIZE_ORDER.get(actor_size, 1)
        other_size_val = SIZE_ORDER.get(other_size, 1)

        if actor_size_val <= other_size_val:
            continue  # Actor is same size or smaller, no linked movement

        # Move smaller party to maintain adjacency
        final_coord = final_position.coord
        adjacent_hexes = final_coord.neighbors()

        # Pick the adjacent hex closest to their original position
        old_coord = other.position.coord
        best_hex = min(adjacent_hexes, key=lambda h: h.distance_to(old_coord))

        # Update smaller party's position
        new_other_position = HexPosition(coord=best_hex, elevation=other.position.elevation)
        updated_other = other.model_copy(update={"position": new_other_position})
        updated_combatants = list(scenario.combatants)
        updated_combatants[other_idx] = updated_other
        scenario = MechCombatScenario(
            combatants=updated_combatants,
            grapples=list(scenario.grapples),
            rounds=list(scenario.rounds),
            terrain=scenario.terrain,
            environment=scenario.environment,
            deployables=dict(scenario.deployables),
            sitrep_resolution=scenario.sitrep_resolution,
            pending_decisions=list(scenario.pending_decisions),
        )

        effects.append({
            "type": "linked_grapple_movement",
            "larger_id": actor.id,
            "smaller_id": other_id,
            "from_coord": {"q": old_coord.q, "r": old_coord.r},
            "to_coord": {"q": best_hex.q, "r": best_hex.r},
        })

        break  # Only one linked movement per movement action

    # Check and break grapples if movement causes adjacency loss (PR2 4175)
    scenario, grapple_break_effects = _check_and_break_grapples_on_movement(
        scenario,
        actor.id,
        final_position.coord,
    )
    effects.extend(grapple_break_effects)

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
            sitrep_resolution=scenario.sitrep_resolution,
            pending_decisions=list(scenario.pending_decisions),
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
            sitrep_resolution=scenario.sitrep_resolution,
            pending_decisions=list(scenario.pending_decisions),
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
            sitrep_resolution=scenario.sitrep_resolution,
            pending_decisions=list(scenario.pending_decisions),
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
        sitrep_resolution=scenario.sitrep_resolution,
        pending_decisions=list(scenario.pending_decisions),
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

        # Check if any path hex is adjacent to mine (considering mover's size)
        mine_coord = mine.position.coord
        # Mover's size determines trigger radius - larger mechs trigger at greater distance
        mover_size = mover.stats.size if mover.stats else "size_1"
        adj_dist = adjacency_distance(mover_size, "size_1")  # Mines are size 1
        mine_trigger_hexes = {(h.q, h.r) for h in hexes_in_radius(mine_coord, adj_dist)}

        if path_coords & mine_trigger_hexes:
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
            sitrep_resolution=scenario.sitrep_resolution,
            pending_decisions=list(scenario.pending_decisions),
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
    apply_typed_damage_func=None,
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
        apply_typed_damage_func: Optional typed damage application function
        profile_id: Optional weapon profile ID for weapons with multiple profiles

    Returns:
        Tuple of (updated scenario, AttackOutcome with hit/miss/damage info)
    """
    from core.shared.rolls import resolve_attack
    from core.mech.combat_models import AttackOutcome, ResourceChange

    # Lazy import to avoid circular dependency
    if apply_typed_damage_func is None:
        from core.mech.combat_execution import apply_typed_damage
        apply_typed_damage_func = apply_typed_damage

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
    primary_damage_type = _get_primary_damage_type(weapon_profile)

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

    # Get talent/frame effect modifiers (Phase 32)
    talent_acc_mod, talent_diff_mod = _get_talent_accuracy_modifiers(
        attacker,
        is_melee=not is_ranged_attack,
        is_ranged=is_ranged_attack,
        is_tech=smart_attack,
        context={"is_outgoing": True},
    )

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

    # Combine all accuracy/difficulty modifiers (including talents, Phase 32)
    final_accuracy_bonus = accuracy_bonus + attacker_acc_mod + target_acc_mod + talent_acc_mod
    final_difficulty_bonus = (
        difficulty_bonus + attacker_diff_mod + target_diff_mod + cover_difficulty + talent_diff_mod
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
            "talent_acc": talent_acc_mod,
            "talent_diff": talent_diff_mod,
        },
    })

    # Initialize outcome tracking
    damage_dealt = 0
    damage_breakdown = DamageBreakdown()
    resource_change: ResourceChange | None = None
    structure_check: dict | None = None

    # Apply damage if hit
    if attack_result.hit:
        base_components, _overkill_heat = _roll_weapon_damage_components(
            weapon_profile, apply_overkill=has_overkill
        )

        scaled_components: list[tuple[str, int]] = []
        for damage_type, amount in base_components:
            scaled = amount
            if attack_result.is_critical:
                scaled *= 2
            scaled_components.append((damage_type, scaled))

        if "exposed" in target.statuses:
            effects.append({
                "type": "exposed_multiplier",
                "target_id": target.id,
                "multiplier": 2,
            })

        if reliable_value is not None:
            total_scaled_damage = sum(
                amount for dmg_type, amount in scaled_components if dmg_type != "heat"
            )
            if total_scaled_damage < reliable_value:
                scaled_components.append(
                    (primary_damage_type, reliable_value - total_scaled_damage)
                )

        if "shredded" in target.statuses:
            effects.append({
                "type": "shredded_armor_bypass",
                "target_id": target.id,
                "armor_bypassed": target.stats.armor if target.stats else 0,
            })

        damage_context = _build_damage_context(
            attacker=attacker,
            target=target,
            is_melee=not is_ranged_attack,
            is_ranged=is_ranged_attack,
            is_tech=smart_attack,
        )
        target_resistances = _collect_damage_resistances(target, damage_context)
        heat_multiplier = _collect_heat_resistance_multiplier(target, damage_context)

        scenario, change, breakdown, structure_result, _overheat_result = apply_typed_damage_func(
            scenario,
            target.id,
            scaled_components,
            armor_piercing=armor_piercing,
            attacker_id=attacker.id,
            resistances=target_resistances,
            heat_resistance_multiplier=heat_multiplier,
        )
        resource_change = change
        damage_dealt = (
            breakdown.kinetic + breakdown.explosive + breakdown.energy + breakdown.burn
        )
        damage_breakdown = breakdown

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

        if has_lock_on:
            scenario = _remove_status_from_target(scenario, target.id, "lock_on")
            effects.append({
                "type": "lock_on_consumed",
                "target_id": target.id,
            })
    elif reliable_value is not None:
        if "shredded" in target.statuses:
            effects.append({
                "type": "shredded_armor_bypass",
                "target_id": target.id,
                "armor_bypassed": target.stats.armor if target.stats else 0,
            })

        damage_context = _build_damage_context(
            attacker=attacker,
            target=target,
            is_melee=not is_ranged_attack,
            is_ranged=is_ranged_attack,
            is_tech=smart_attack,
        )
        target_resistances = _collect_damage_resistances(target, damage_context)
        heat_multiplier = _collect_heat_resistance_multiplier(target, damage_context)

        scenario, change, breakdown, structure_result, _overheat_result = apply_typed_damage_func(
            scenario,
            target.id,
            [(primary_damage_type, reliable_value)],
            armor_piercing=armor_piercing,
            attacker_id=attacker.id,
            resistances=target_resistances,
            heat_resistance_multiplier=heat_multiplier,
        )
        resource_change = change
        damage_dealt = (
            breakdown.kinetic + breakdown.explosive + breakdown.energy + breakdown.burn
        )
        damage_breakdown = breakdown
        effects.append({
            "type": "reliable_damage",
            "target_id": target.id,
            "amount": reliable_value,
        })

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

    return scenario, AttackOutcome(
        hit=attack_result.hit,
        critical=attack_result.is_critical,
        damage_dealt=damage_dealt,
        damage_breakdown=damage_breakdown,
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
    "_roll_weapon_damage_components",
    "_get_primary_damage_type",
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
    # Damage/Resistance Helpers
    "_build_damage_context",
    "_collect_damage_resistances",
    "_collect_heat_resistance_multiplier",
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
