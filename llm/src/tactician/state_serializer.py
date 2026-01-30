"""Combat state serializer for LLM tactician.

Converts MechCombatScenario state into a structured, LLM-readable format.
"""

from __future__ import annotations

from typing import Any
from core.mech.combat_state import (
    MechCombatScenario,
    CombatantState,
    CombatTurn,
)
from core.mech.combat_models import AvailableAction
from core.mech.combat_execution import (
    get_current_actor,
    get_available_actions,
)
from core.mech.action_economy import (
    ActionEconomyState,
    use_full_action,
    use_quick_action,
    use_overcharge,
    use_reaction,
)
from core.mech.compendium import get_weapon_definition
from core.mech.weapon import WeaponDamage, resolve_weapon_profile


def get_potential_targets(scenario: MechCombatScenario, actor_id: str) -> list[str]:
    """Get IDs of potential targets for an actor (enemies within sensor range)."""
    actor = None
    for c in scenario.combatants:
        if c.id == actor_id:
            actor = c
            break
    if actor is None or actor.position is None:
        return []

    sensor_range = actor.stats.sensor_range
    potential = []
    for target in scenario.combatants:
        if target.id == actor_id:
            continue
        if target.side == actor.side:
            continue
        if target.resources.hp_current <= 0:
            continue
        if target.position is None:
            continue
        distance = actor.position.coord.distance_to(target.position.coord)
        if distance <= sensor_range:
            potential.append(target.id)
    return potential


def compute_damage_stats(damage_components: list[WeaponDamage]) -> dict[str, Any]:
    """Compute min, max, average damage across all damage components."""
    total_min = 0
    total_max = 0
    total_avg = 0.0
    damage_types = []
    for dmg in damage_components:
        # Dice component
        dice_min = dmg.dice.min_value() if dmg.dice else 0
        dice_max = dmg.dice.max_value() if dmg.dice else 0
        dice_avg = dmg.dice.average() if dmg.dice else 0.0
        # Flat component
        flat = dmg.flat
        total_min += dice_min + flat
        total_max += dice_max + flat
        total_avg += dice_avg + flat
        if dmg.damage_type:
            damage_types.append(dmg.damage_type)
    # Deduplicate damage types
    unique_types = list(set(damage_types))
    return {
        "min": total_min,
        "max": total_max,
        "average": total_avg,
        "damage_types": unique_types,
    }


def compute_average_weapon_damage(combatant: CombatantState) -> dict[str, Any] | None:
    """Compute average damage for the combatant's first usable weapon.

    Returns dict with min, max, average damage, or None if no weapons.
    """
    if combatant.inventory is None:
        return None
    for mount in combatant.inventory.mounts:
        if mount.destroyed:
            continue
        for weapon_state in mount.weapons:
            if weapon_state.destroyed:
                continue
            try:
                weapon_def = get_weapon_definition(weapon_state.weapon_id)
                if weapon_def is None:
                    continue
                # Resolve to weapon profile
                profile = resolve_weapon_profile(weapon_def)
                damage_stats = compute_damage_stats(profile.damage)
                # Include weapon name and id for reference
                damage_stats["weapon_id"] = weapon_state.weapon_id
                damage_stats["weapon_name"] = profile.name
                return damage_stats
            except Exception:
                # Weapon not found or other error
                continue
    return None


def compute_action_parameters(
    scenario: MechCombatScenario,
    actor_id: str,
    action: AvailableAction,
) -> dict[str, Any]:
    """Enrich an AvailableAction with target parameters.

    Returns dict with:
    - targets: list of potential target IDs with distances
    - damage_prediction: estimated damage for weapon actions
    """
    actor = None
    for c in scenario.combatants:
        if c.id == actor_id:
            actor = c
            break
    if actor is None:
        return {"targets": [], "damage_prediction": None}

    result = {"targets": [], "damage_prediction": None}

    # Compute potential targets if action requires target
    if action.requires_target:
        potential_targets = get_potential_targets(scenario, actor_id)
        for target_id in potential_targets:
            target = None
            for c in scenario.combatants:
                if c.id == target_id:
                    target = c
                    break
            if target is None or target.position is None or actor.position is None:
                continue
            distance = actor.position.coord.distance_to(target.position.coord)
            result["targets"].append(
                {
                    "id": target_id,
                    "distance": distance,
                }
            )

    # Compute damage prediction for weapon-based actions
    if action.requires_weapon:
        damage_info = compute_average_weapon_damage(actor)
        if damage_info:
            result["damage_prediction"] = damage_info

    return result


def compute_economy_from_turn(turn: CombatTurn) -> ActionEconomyState:
    """Compute current action economy based on actions taken in this turn."""
    economy = ActionEconomyState()
    for action in turn.actions:
        if action.granted_by_overcharge:
            economy = use_overcharge(economy)
        if action.action_type == "full":
            economy = use_full_action(economy)
        elif action.action_type == "quick":
            economy = use_quick_action(economy)
        elif action.action_type == "reaction":
            economy = use_reaction(economy)
        # free actions don't consume economy
    return economy


def serialize_combat_state(scenario: MechCombatScenario) -> dict[str, Any]:
    """Serialize combat scenario into LLM-readable structured dict.

    Output includes:
    - current actor
    - all combatants (position, HP, heat, conditions)
    - available actions for current actor
    - terrain/hazards on the map
    - turn order, round number
    - positions use hex coordinates matching the combat grid
    - actions include their parameters (targets, ranges, damage predictions)

    Args:
        scenario: Current combat scenario

    Returns:
        Structured dict suitable for LLM consumption
    """
    # Infer current round and turn index
    if not scenario.rounds:
        current_round = 1
        current_turn_index = 0
        current_turn = None
        current_round_data = None
    else:
        current_round_data = scenario.rounds[-1]
        current_round = current_round_data.round_index
        if not current_round_data.turns:
            current_turn_index = 0
            current_turn = None
        else:
            # Assume current turn is the last turn in the list (ongoing)
            current_turn_index = len(current_round_data.turns) - 1
            current_turn = current_round_data.turns[current_turn_index]

    # Determine which actors have acted this round
    acted_ids = set()
    if current_round_data and current_round_data.turns:
        for turn in current_round_data.turns:
            acted_ids.add(turn.actor_id)

    # Get current actor
    current_actor = get_current_actor(scenario, current_round, current_turn_index)

    # Compute economy based on current turn's actions
    if current_turn is not None:
        economy = compute_economy_from_turn(current_turn)
    else:
        economy = ActionEconomyState()

    # Get available actions for current actor (if any)
    available_actions_result = None
    if current_actor is not None:
        available_actions_result = get_available_actions(
            scenario, current_actor.id, economy
        )

    # Serialize combatants
    combatants = []
    for c in scenario.combatants:
        combatant_dict = {
            "id": c.id,
            "name": c.name,
            "side": c.side,
            "kind": c.kind,
            "position": c.position.model_dump(mode="json") if c.position else None,
            "stats": {
                "size": c.stats.size,
                "hp_max": c.stats.hp_max,
                "evasion": c.stats.evasion,
                "e_defense": c.stats.e_defense,
                "armor": c.stats.armor,
                "speed": c.stats.speed,
                "sensor_range": c.stats.sensor_range,
                "tech_attack": c.stats.tech_attack,
                "grit": c.stats.grit,
                "engineering_skill": c.stats.engineering_skill,
            },
            "resources": {
                "hp_current": c.resources.hp_current,
                "heat_current": c.resources.heat_current,
                "heat_cap": c.resources.heat_cap,
                "structure_current": c.resources.structure_current,
                "stress_current": c.resources.stress_current,
                "repairs_remaining": c.resources.repairs_remaining,
                "burn_marked": c.resources.burn_marked,
            },
            "statuses": c.statuses,
            "conditions": c.conditions,
            "ai_controlled": c.ai_controlled,
        }
        combatants.append(combatant_dict)

    # Serialize terrain
    terrain = None
    if scenario.terrain:
        tiles = []
        for tile in scenario.terrain.tiles:
            tiles.append(
                {
                    "coord": tile.coord.model_dump(mode="json"),
                    "elevation": tile.elevation,
                    "blocks_line_of_sight": tile.blocks_line_of_sight,
                    "provides_soft_cover": tile.provides_soft_cover,
                    "provides_hard_cover": tile.provides_hard_cover,
                    "hard_cover_size": tile.hard_cover_size,
                    "difficult": tile.difficult,
                    "dangerous": tile.dangerous,
                }
            )
        terrain = {
            "type": "terrain_map",
            "tiles": tiles,
            "hazards": [t for t in tiles if t["dangerous"]],
        }

    # Serialize deployables
    deployables = []
    for deployable_id, deployable in scenario.deployables.items():
        deployables.append(
            {
                "id": deployable_id,
                "name": deployable.name,
                "kind": deployable.kind,
                "position": deployable.position.model_dump(mode="json")
                if deployable.position
                else None,
                "hp": deployable.hp,
                "max_hp": deployable.max_hp,
                "armor": deployable.armor,
                "evasion": deployable.evasion,
                "is_destroyed": deployable.is_destroyed,
                "is_active": deployable.is_active,
            }
        )

    # Serialize grapples
    grapples = []
    for grapple in scenario.grapples:
        grapples.append(
            {
                "grappler_id": grapple.grappler_id,
                "target_id": grapple.target_id,
            }
        )

    # Serialize turn order with side and acted status
    turn_order = []
    side_order = {"players": 0, "hostiles": 1, "neutral": 2}
    for c in scenario.combatants:
        if c.resources.hp_current > 0:
            turn_order.append(
                {
                    "id": c.id,
                    "side": c.side,
                    "has_acted": c.id in acted_ids,
                }
            )
    # Sort by side, then by id
    turn_order.sort(key=lambda x: (side_order.get(x["side"], 3), x["id"]))

    # Serialize available actions with parameters
    actions = []
    if available_actions_result is not None and current_actor is not None:
        for action_list in [
            available_actions_result.full_actions,
            available_actions_result.quick_actions,
            available_actions_result.free_actions,
            available_actions_result.reactions,
            available_actions_result.protocols,
        ]:
            for action in action_list:
                action_dict = {
                    "action_id": action.action_id,
                    "action_name": action.action_name,
                    "action_type": action.action_type,
                    "is_available": action.is_available,
                    "unavailable_reason": action.unavailable_reason,
                    "requires_target": action.requires_target,
                    "requires_weapon": action.requires_weapon,
                    "requires_system": action.requires_system,
                    "requires_path": action.requires_path,
                    "max_targets": action.max_targets,
                    # Add parameters (targets, ranges, damage predictions)
                    "parameters": compute_action_parameters(
                        scenario, current_actor.id, action
                    ),
                }
                actions.append(action_dict)

    # Build final output
    result = {
        "current_actor": current_actor.id if current_actor else None,
        "combatants": combatants,
        "terrain": terrain,
        "deployables": deployables,
        "grapples": grapples,
        "turn_order": turn_order,
        "round_number": current_round,
        "current_turn_index": current_turn_index,
        "available_actions": actions,
        "action_economy": {
            "full_actions_remaining": economy.full_actions_remaining,
            "quick_actions_remaining": economy.quick_actions_remaining,
            "can_overcharge": economy.can_overcharge,
            "reactions_remaining_this_turn": economy.reactions_remaining_this_turn,
        },
    }
    return result
