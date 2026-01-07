"""Combat resolution helpers for Lancer TTRPG.

Provides reusable helpers for common combat resolution patterns that tie
together existing primitives (rolls, damage, saves, conditions, effects).

Per PR2 528-620 combat rules:
- Attack roll: 1d20 + attack bonus vs defense
- Critical hits: roll all damage dice twice, pick highest (PR2 3965-3969)
- Damage resolution: raw damage → reductions → HP/heat
- Saves: HULL/AGI/SYS/ENG vs target
- Status interactions: condition modifiers apply at each step

Reactions (PR2 4381-4401):
- Brace: 1/round, resistance to triggering attack, +1 difficulty on others
- Overwatch: 1/round, skirmish when enemy enters threat

Drones (PR2 5070-5088):
- Turret drones make reaction attacks when allies hit within range 10
- Other drones provide various support effects
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import BaseModel

from core.shared.enums import (
    ActionType,
    DamageType,
    SaveType,
    StatusType,
)
from core.shared.dice import roll_dice, round_up
from core.shared.rolls import (
    AttackResolutionResult,
    resolve_attack,
)
from core.shared.saves import resolve_save, SaveRequest, SaveResult


HexCoord = tuple[int, int]


AttackPatternType = Literal["single", "line", "cone", "blast", "burst"]


class AttackPattern(BaseModel):
    """Area attack patterns with geometric specification (PR2 3985-4012)."""

    type: AttackPatternType = "single"
    size: int | None = None
    origin: HexCoord | None = None


class GeometryValidationResult(BaseModel):
    """Result of validating attack pattern geometry."""

    is_valid: bool
    affected_spaces: list[HexCoord] = Field(default_factory=list)
    affected_target_ids: list[str] = Field(default_factory=list)
    obstruction_coords: list[HexCoord] = Field(default_factory=list)
    reason: str = ""


def validate_attack_geometry(
    pattern: AttackPattern,
    attacker_position: HexCoord,
    target_positions: dict[str, HexCoord],
    obstructions: list[HexCoord] | None = None,
) -> GeometryValidationResult:
    """Validate area attack pattern geometry per PR2 3985-4012."""
    if pattern.type == "single":
        return GeometryValidationResult(
            is_valid=True,
            affected_spaces=[attacker_position],
            affected_target_ids=list(target_positions.keys()),
            reason="Single target: no geometry needed",
        )

    if pattern.size is None:
        return GeometryValidationResult(
            is_valid=False,
            reason=f"Pattern type '{pattern.type}' requires size parameter",
        )

    if obstructions is None:
        obstructions = []

    origin = pattern.origin or attacker_position
    affected_spaces: list[HexCoord] = []
    affected_target_ids: list[str] = []

    if pattern.type == "line":
        affected_spaces = _compute_line_spaces(origin, pattern.size)
    elif pattern.type == "cone":
        affected_spaces = _compute_cone_spaces(origin, pattern.size)
    elif pattern.type == "blast":
        affected_spaces = _compute_blast_spaces(origin, pattern.size)
    elif pattern.type == "burst":
        affected_spaces = _compute_burst_spaces(origin, pattern.size)

    for target_id, target_pos in target_positions.items():
        if target_pos in affected_spaces:
            if target_pos not in obstructions:
                affected_target_ids.append(target_id)

    return GeometryValidationResult(
        is_valid=True,
        affected_spaces=affected_spaces,
        affected_target_ids=affected_target_ids,
        reason=f"{pattern.type} pattern with size {pattern.size}",
    )


def _compute_line_spaces(origin: HexCoord, length: int) -> list[HexCoord]:
    """Compute spaces in a line pattern (PR2 3988-3989)."""
    spaces: list[HexCoord] = []
    for i in range(length + 1):
        spaces.append((origin[0] + i, origin[1]))
    return spaces


def _compute_cone_spaces(origin: HexCoord, size: int) -> list[HexCoord]:
    """Compute spaces in a cone pattern (PR2 3990-3992)."""
    spaces: list[HexCoord] = []
    for row in range(size + 1):
        width = row * 2 + 1
        for col in range(width):
            q = origin[0] - row + col
            r = origin[1] + row
            spaces.append((q, r))
    return spaces


def _compute_blast_spaces(origin: HexCoord, radius: int) -> list[HexCoord]:
    """Compute spaces in a blast pattern (PR2 3993-3995)."""
    spaces: list[HexCoord] = []
    for dq in range(-radius, radius + 1):
        for dr in range(-radius, radius + 1):
            if abs(dq) + abs(dr) <= radius:
                spaces.append((origin[0] + dq, origin[1] + dr))
    return spaces


def _compute_burst_spaces(origin: HexCoord, radius: int) -> list[HexCoord]:
    """Compute spaces in a burst pattern (PR2 3996-3999)."""
    spaces: list[HexCoord] = []
    for dq in range(-radius, radius + 1):
        for dr in range(-radius, radius + 1):
            if abs(dq) + abs(dr) <= radius:
                spaces.append((origin[0] + dq, origin[1] + dr))
    spaces.append(origin)
    return spaces


CriticalEffectType = Literal[
    "knockback",
    "prone_save",
    "shredded",
    "impaired",
    "immobilized",
    "stunned_save",
    "pull",
    "heat",
    "burn",
]


class CriticalEffect(BaseModel):
    """Critical hit follow-up effects - applied automatically on natural 20."""

    type: CriticalEffectType
    value: int | None = None


class CriticalDamageResult(BaseModel):
    """Critical hit damage calculation (roll twice, pick highest per PR2 3965-3969)."""

    base_damage: int
    bonus_damage: int
    rolled_once: list[int] = Field(default_factory=list)
    rolled_twice: list[int] = Field(default_factory=list)
    highest_selected: list[int] = Field(default_factory=list)
    total_damage: int
    is_critical: bool = True


def calculate_critical_damage(
    base_damage: int,
    bonus_damage: int,
    is_critical: bool,
    force_rolls: list[int] | None = None,
) -> CriticalDamageResult:
    """Calculate damage with critical hit doubling (PR2 3965-3969)."""
    if not is_critical:
        rolled_once = force_rolls if force_rolls else [roll_dice(f"{base_damage}d6")]
        total = sum(rolled_once) + bonus_damage
        return CriticalDamageResult(
            base_damage=base_damage,
            bonus_damage=bonus_damage,
            rolled_once=rolled_once,
            total_damage=total,
            is_critical=False,
        )

    if force_rolls:
        rolled_once = list(force_rolls[:base_damage])
        rolled_again = list(force_rolls[base_damage : base_damage * 2])
    else:
        rolled_once = [roll_dice(f"{base_damage}d6")]
        rolled_again = [roll_dice(f"{base_damage}d6")]

    all_rolls = rolled_once + rolled_again
    all_rolls.sort(reverse=True)
    highest_selected = all_rolls[:base_damage]
    total = sum(highest_selected) + bonus_damage

    return CriticalDamageResult(
        base_damage=base_damage,
        bonus_damage=bonus_damage,
        rolled_once=rolled_once,
        rolled_twice=all_rolls,
        highest_selected=highest_selected,
        total_damage=total,
        is_critical=True,
    )


class TargetAttackResult(BaseModel):
    """Result for one target."""

    target_id: str
    target_position: HexCoord | None = None
    attack_roll: int = 0
    is_hit: bool = False
    is_critical: bool = False
    miss_by: int = 0
    damage_result: CriticalDamageResult | None = None
    damage_applied: int = 0
    heat_applied: int = 0
    knockback_result: object | None = None
    conditions_applied: list[StatusType] = Field(default_factory=list)
    save_result: SaveResult | None = None
    resolution_notes: list[str] = Field(default_factory=list)


class AttackSequenceInput(BaseModel):
    """Complete input for attack resolution."""

    attacker_id: str
    position: HexCoord | None = None
    target_ids: list[str]
    pattern: AttackPattern | None = None
    attack_bonus: int
    defense_value: int
    accuracy_bonus: int = 0
    difficulty_bonus: int = 0
    base_damage: int
    damage_type: DamageType
    armor_piercing: int = 0
    bonus_damage: int = 0
    critical_effects: list[CriticalEffect] = Field(default_factory=list)
    save_target: int | None = None
    save_type: SaveType | None = None
    save_on_miss: bool = False
    drone_assisted: bool = False
    drone_id: str | None = None
    drone_bonus: int = 0
    target_positions: dict[str, HexCoord] = Field(default_factory=dict)
    force_attack_roll: int | None = None
    force_accuracy_rolls: list[int] | None = None
    force_difficulty_rolls: list[int] | None = None
    force_damage_rolls: list[int] | None = None


class AttackSequenceResult(BaseModel):
    """Complete attack sequence result."""

    attacker_id: str
    pattern: AttackPattern | None = None
    target_results: list[TargetAttackResult]
    total_targets: int
    targets_hit: int
    targets_critical: int
    total_damage_dealt: int
    total_heat_dealt: int
    overwatch_triggers: list[dict] = Field(default_factory=list)
    drone_triggers: list[dict] = Field(default_factory=list)
    conditions_on_targets: dict[str, list[StatusType]] = Field(default_factory=dict)
    positions_changed: dict[str, object] = Field(default_factory=dict)


def resolve_attack_sequence(input: AttackSequenceInput) -> AttackSequenceResult:
    """Resolve complete attack sequence per PR2 rules."""
    target_results: list[TargetAttackResult] = []
    total_damage = 0
    total_heat = 0
    targets_hit = 0
    targets_critical = 0
    conditions_on_targets: dict[str, list[StatusType]] = {}
    positions_changed: dict[str, HexCoord] = {}

    target_positions = input.target_positions or {
        tid: (0, 0) for tid in input.target_ids
    }

    for target_id in input.target_ids:
        target_pos = target_positions.get(target_id, (0, 0))
        result = _resolve_single_target_attack(input, target_id, target_pos)
        target_results.append(result)

        if result.is_hit:
            targets_hit += 1
            total_damage += result.damage_applied
            total_heat += result.heat_applied

            if target_id not in conditions_on_targets:
                conditions_on_targets[target_id] = []
            conditions_on_targets[target_id].extend(result.conditions_applied)

            if (
                result.knockback_result
                and hasattr(result.knockback_result, "end_position")
                and result.knockback_result.end_position
            ):
                positions_changed[target_id] = result.knockback_result.end_position

        if result.is_critical:
            targets_critical += 1

    return AttackSequenceResult(
        attacker_id=input.attacker_id,
        pattern=input.pattern,
        target_results=target_results,
        total_targets=len(input.target_ids),
        targets_hit=targets_hit,
        targets_critical=targets_critical,
        total_damage_dealt=total_damage,
        total_heat_dealt=total_heat,
        conditions_on_targets=conditions_on_targets,
        positions_changed=positions_changed,
    )


def _resolve_single_target_attack(
    input: AttackSequenceInput,
    target_id: str,
    target_position: HexCoord,
) -> TargetAttackResult:
    """Resolve attack against a single target."""
    result = TargetAttackResult(target_id=target_id, target_position=target_position)

    attack_bonus = input.attack_bonus
    if input.drone_assisted:
        attack_bonus += input.drone_bonus

    attack_result = resolve_attack(
        attack_bonus=attack_bonus,
        target_defense=input.defense_value,
        accuracy_bonus=input.accuracy_bonus,
        difficulty_bonus=input.difficulty_bonus,
        forced_roll=input.force_attack_roll,
        forced_accuracy_rolls=input.force_accuracy_rolls,
        forced_difficulty_rolls=input.force_difficulty_rolls,
    )

    result.attack_roll = attack_result.roll
    result.is_hit = attack_result.hit
    result.is_critical = attack_result.is_critical
    result.miss_by = attack_result.miss_by
    result.resolution_notes.append(
        f"Attack roll {attack_result.roll}+{attack_bonus} vs {input.defense_value}"
    )

    if not result.is_hit:
        if input.save_on_miss and input.save_target is not None and input.save_type:
            save_req = SaveRequest(
                save_type=input.save_type,
                save_target=input.save_target,
            )
            result.save_result = resolve_save(save_req)
            result.resolution_notes.append(f"Save on miss: {result.save_result.degree}")
        return result

    multi_target = len(input.target_ids) > 1
    effective_bonus_damage = input.bonus_damage
    if multi_target and effective_bonus_damage > 0:
        effective_bonus_damage = round_up(input.bonus_damage / len(input.target_ids))
        result.resolution_notes.append(
            f"Multi-target bonus damage halved to {effective_bonus_damage}"
        )

    crit_damage = calculate_critical_damage(
        base_damage=input.base_damage,
        bonus_damage=effective_bonus_damage,
        is_critical=result.is_critical,
        force_rolls=input.force_damage_rolls,
    )
    result.damage_result = crit_damage
    result.damage_applied = crit_damage.total_damage

    result.resolution_notes.append(
        f"Damage: {crit_damage.total_damage} ({input.damage_type})"
    )

    if result.is_critical and input.critical_effects:
        attacker_pos = input.position or (0, 0)
        for effect in input.critical_effects:
            effect_result = _apply_critical_effect(
                effect, target_id, target_position, attacker_pos
            )
            if effect_result.get("condition"):
                result.conditions_applied.append(effect_result["condition"])
            if effect_result.get("knockback"):
                result.knockback_result = effect_result["knockback"]
            if effect_result.get("save"):
                result.save_result = effect_result["save"]
            result.resolution_notes.append(f"Critical effect: {effect.type}")

    return result


def _apply_critical_effect(
    effect: CriticalEffect,
    target_id: str,
    target_position: HexCoord,
    attacker_position: HexCoord,
) -> dict:
    """Apply a critical hit effect."""
    from core.shared.involuntary_movement import resolve_knockback
    from core.mech.grid import HexCoord as MechHexCoord

    result: dict = {}

    if effect.type == "knockback" and effect.value:
        source_hex = MechHexCoord(q=attacker_position[0], r=attacker_position[1])
        target_hex = MechHexCoord(q=target_position[0], r=target_position[1])
        kb_result = resolve_knockback(
            source=source_hex,
            target=target_hex,
            spaces=effect.value,
        )
        result["knockback"] = kb_result

    elif effect.type == "prone_save":
        save_req = SaveRequest(
            save_type="hull",
            save_target=10,
        )
        save_result = resolve_save(save_req)
        result["save"] = save_result
        if not save_result.success:
            result["condition"] = "prone"

    elif effect.type == "shredded":
        result["condition"] = "shredded"

    elif effect.type == "impaired":
        result["condition"] = "impaired"

    elif effect.type == "immobilized":
        result["condition"] = "immobilized"

    elif effect.type == "stunned_save":
        save_req = SaveRequest(
            save_type="systems",
            save_target=10,
        )
        save_result = resolve_save(save_req)
        result["save"] = save_result
        if not save_result.success:
            result["condition"] = "stunned"

    elif effect.type == "pull":
        source_hex = MechHexCoord(q=target_position[0], r=target_position[1])
        target_hex = MechHexCoord(q=target_position[0], r=target_position[1])
        kb_result = resolve_knockback(
            source=source_hex,
            target=target_hex,
            spaces=effect.value or 1,
        )
        result["knockback"] = kb_result

    return result


class MovementPath(BaseModel):
    """Movement path with threat zone analysis."""

    start: HexCoord
    end: HexCoord
    spaces: int
    path_hexes: list[HexCoord]
    threat_zones_entered: list[dict] = Field(default_factory=list)
    overwatch_triggers: list[dict] = Field(default_factory=list)
    difficult_terrain_spaces: int = 0
    dangerous_terrain_spaces: int = 0
    engagement_penalty: bool = False


class MovementInput(BaseModel):
    """Movement resolution input."""

    mover_id: str
    start_position: HexCoord
    end_position: HexCoord
    speed: int
    can_fly: bool = False
    is_disengaging: bool = False
    nearby_enemies: list[dict] = Field(default_factory=list)
    terrain_map: dict[str, str] | None = None
    is_prone: bool = False
    is_slowed: bool = False
    is_immobilized: bool = False
    is_stunned: bool = False
    force_path: list[HexCoord] | None = None


class MovementResult(BaseModel):
    """Complete movement result."""

    mover_id: str
    start: HexCoord
    end: HexCoord
    spaces_moved: int
    path_valid: bool
    path_hexes: list[HexCoord]
    obstructed: bool = False
    obstruction_hex: HexCoord | None = None
    difficult_terrain_penalty: int = 0
    dangerous_terrain_checks: list[dict] = Field(default_factory=list)
    overwatch_triggers: list[dict] = Field(default_factory=list)
    prone_from_fall: bool = False
    fall_damage: int = 0
    disengage_used: bool = False
    reactions_avoided: list[str] = Field(default_factory=list)


def resolve_movement(input: MovementInput) -> MovementResult:
    """Resolve movement with overwatch integration per PR2 rules."""
    path_hexes: list[HexCoord]
    if input.force_path:
        path_hexes = input.force_path
    else:
        path_hexes = _compute_straight_path(input.start_position, input.end_position)

    spaces_moved = len(path_hexes) - 1
    if spaces_moved > input.speed:
        spaces_moved = input.speed

    difficult_terrain = 0
    dangerous_checks: list[dict] = []

    if input.terrain_map:
        for i, hex_pos in enumerate(path_hexes[1:], 1):
            hex_key = f"{hex_pos[0]},{hex_pos[1]}"
            terrain = input.terrain_map.get(hex_key, "")
            if terrain == "difficult":
                difficult_terrain += 1
            elif terrain == "dangerous":
                dangerous_checks.append(
                    {
                        "hex": hex_pos,
                        "check_result": "safe",
                    }
                )

    overwatch_triggers: list[dict] = []
    if not input.is_disengaging and input.nearby_enemies:
        for enemy in input.nearby_enemies:
            enemy_pos = enemy.get("position", (0, 0))
            threat = enemy.get("threat", 1)
            weapons = enemy.get("weapons", [])

            for i, hex_pos in enumerate(path_hexes[1:], 1):
                dist = _hex_distance(hex_pos, enemy_pos)
                if dist <= threat:
                    has_overwatch = any(
                        w.get("overwatch_available", False) for w in weapons
                    )
                    if has_overwatch:
                        overwatch_triggers.append(
                            {
                                "enemy_id": enemy.get("id"),
                                "weapon_id": weapons[0].get("id") if weapons else None,
                                "at_hex": hex_pos,
                                "at_step": i,
                            }
                        )

    disengage_used = input.is_disengaging
    reactions_avoided: list[str] = []
    if disengage_used:
        reactions_avoided = [e.get("id") for e in input.nearby_enemies]
        overwatch_triggers = []

    return MovementResult(
        mover_id=input.mover_id,
        start=input.start_position,
        end=path_hexes[-1] if path_hexes else input.end_position,
        spaces_moved=spaces_moved,
        path_valid=True,
        path_hexes=path_hexes,
        difficult_terrain_penalty=difficult_terrain,
        dangerous_terrain_checks=dangerous_checks,
        overwatch_triggers=overwatch_triggers,
        disengage_used=disengage_used,
        reactions_avoided=reactions_avoided,
    )


def _compute_straight_path(start: HexCoord, end: HexCoord) -> list[HexCoord]:
    """Compute straight-line path between two hexes."""
    path: list[HexCoord] = [start]
    if start == end:
        return path

    dq = end[0] - start[0]
    dr = end[1] - start[1]
    steps = max(abs(dq), abs(dr))

    if steps == 0:
        return path

    for i in range(1, steps + 1):
        new_q = start[0] + (dq // steps) * i
        new_r = start[1] + (dr // steps) * i
        path.append((new_q, new_r))

    return path


def _hex_distance(a: HexCoord, b: HexCoord) -> int:
    """Calculate hex distance between two positions."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs((a[0] + a[1]) - (b[0] + b[1])))


class FullActionTurnInput(BaseModel):
    """Full action turn input."""

    actor_id: str
    position: HexCoord
    action_choice: Literal["full", "two_quick"]
    full_action_type: Literal["attack", "move", "tech", "other"] | None = None
    quick_actions: list[dict] = Field(default_factory=list)
    attack_input: AttackSequenceInput | None = None
    movement_input: MovementInput | None = None
    heat: int = 0
    hp: int = 0
    statuses: list[StatusType] = Field(default_factory=list)
    brace_available: bool = True
    overwatch_available: bool = True
    enemies: list[dict] = Field(default_factory=list)
    allies: list[dict] = Field(default_factory=list)


class FullActionTurnResult(BaseModel):
    """Full action turn result."""

    actor_id: str
    action_type: str
    actions_taken: list[str] = Field(default_factory=list)
    movement_result: MovementResult | None = None
    attack_result: AttackSequenceResult | None = None
    brace_used: bool = False
    brace_details: dict | None = None
    overwatch_triggered_by_enemy: bool = False
    overwatch_triggers_from_this_actor: list[dict] = Field(default_factory=list)
    position_changed: bool = False
    new_position: HexCoord | None = None
    status_changes: dict = Field(default_factory=dict)
    resource_changes: dict = Field(default_factory=dict)
    reactions_remaining_this_round: int = 1
    can_take_actions: bool = True


def resolve_full_action_turn(input: FullActionTurnInput) -> FullActionTurnResult:
    """Resolve complete full action turn per PR2 action economy."""
    actions_taken: list[str] = []
    position_changed = False
    new_position = input.position

    movement_result: MovementResult | None = None
    attack_result: AttackSequenceResult | None = None

    brace_used = False
    brace_details: dict | None = None

    if input.action_choice == "full":
        if input.full_action_type == "move" and input.movement_input:
            movement_result = resolve_movement(input.movement_input)
            actions_taken.append("move")
            position_changed = True
            new_position = movement_result.end

        elif input.full_action_type == "attack" and input.attack_input:
            attack_result = resolve_attack_sequence(input.attack_input)
            actions_taken.append("attack")

    elif input.action_choice == "two_quick":
        for qa in input.quick_actions:
            if qa.get("type") == "move" and input.movement_input:
                movement_result = resolve_movement(input.movement_input)
                actions_taken.append(f"quick_{qa.get('type')}")
                position_changed = True
                new_position = movement_result.end
            elif qa.get("type") == "attack" and input.attack_input:
                attack_result = resolve_attack_sequence(input.attack_input)
                actions_taken.append(f"quick_{qa.get('type')}")

    overwatch_triggers_from_this_actor: list[dict] = []
    if attack_result:
        overwatch_triggers_from_this_actor = attack_result.overwatch_triggers

    return FullActionTurnResult(
        actor_id=input.actor_id,
        action_type=input.action_choice,
        actions_taken=actions_taken,
        movement_result=movement_result,
        attack_result=attack_result,
        brace_used=brace_used,
        brace_details=brace_details,
        overwatch_triggered_by_enemy=False,
        overwatch_triggers_from_this_actor=overwatch_triggers_from_this_actor,
        position_changed=position_changed,
        new_position=new_position,
        status_changes={},
        resource_changes={},
        reactions_remaining_this_round=1 if input.brace_available else 0,
        can_take_actions=True,
    )


class StatusCheckInput(BaseModel):
    """Check status effects on actions."""

    statuses: list[StatusType]
    attempted_action: Literal[
        "attack",
        "move",
        "tech_action",
        "reaction",
        "free_action",
        "stand_up",
        "boost",
        "overcharge",
    ]
    context: dict = Field(default_factory=dict)


class StatusInteractionResult(BaseModel):
    """Result of status check."""

    can_perform: bool
    difficulty_modifier: int = 0
    cannot_take_action: bool = False
    cannot_take_reactions: bool = False
    cannot_move: bool = False
    movement_speed_cap: int | None = None
    only_regular_move: bool = False
    cannot_be_targeted: bool = False
    area_attacks_can_target: bool = True
    only_improvised_or_grapple: bool = False
    cannot_overcharge: bool = False
    reasons: list[str] = Field(default_factory=list)


def check_status_effects(input: StatusCheckInput) -> StatusInteractionResult:
    """Check if statuses prevent/difficulty actions per PR2."""
    can_perform = True
    difficulty_modifier = 0
    reasons: list[str] = []

    cannot_take_action = False
    cannot_take_reactions = False
    cannot_move = False
    movement_speed_cap: int | None = None
    only_regular_move = False
    cannot_be_targeted = False
    area_attacks_can_target = True
    only_improvised_or_grapple = False
    cannot_overcharge = False

    for status in input.statuses:
        if status == "jammed":
            if input.attempted_action == "reaction":
                cannot_take_reactions = True
                reasons.append("JAMMED: cannot take reactions")
            if input.attempted_action in ["attack"]:
                only_improvised_or_grapple = True
                reasons.append("JAMMED: only improvised attacks or grapples")

        elif status == "stunned":
            if input.attempted_action in [
                "attack",
                "move",
                "tech_action",
                "boost",
                "overcharge",
            ]:
                cannot_take_action = True
                reasons.append(
                    "STUNNED: cannot take actions (except mount/dismount/eject)"
                )

        elif status == "immobilized":
            if input.attempted_action == "move":
                cannot_move = True
                reasons.append("IMMOBILIZED: cannot move voluntarily")
            if input.attempted_action == "stand_up":
                cannot_take_action = True
                reasons.append("IMMOBILIZED: cannot stand up from prone")

        elif status == "slowed":
            if input.attempted_action == "move":
                movement_speed_cap = 0
                only_regular_move = True
                reasons.append("SLOWED: max voluntary movement = 0, only regular move")

        elif status == "engaged":
            if input.attempted_action == "attack":
                difficulty_modifier += 1
                reasons.append("ENGAGED: +1 difficulty on ranged attacks")

        elif status == "hidden":
            if input.attempted_action == "attack":
                cannot_be_targeted = True
                area_attacks_can_target = False
                reasons.append("HIDDEN: cannot be directly targeted")

        elif status == "braced":
            if input.attempted_action == "reaction":
                cannot_take_reactions = True
                reasons.append("BRACED: cannot take reactions until end of next turn")

        elif status == "impaired":
            if input.attempted_action in ["attack", "tech_action"]:
                difficulty_modifier += 1
                reasons.append("IMPAIRED: +1 difficulty on attacks and tech actions")

        elif status == "down":
            cannot_take_action = True
            reasons.append("DOWN: unconscious, cannot take actions")

    can_perform = not cannot_take_action and not cannot_take_reactions
    if input.attempted_action == "move":
        can_perform = can_perform and not cannot_move

    return StatusInteractionResult(
        can_perform=can_perform,
        difficulty_modifier=difficulty_modifier,
        cannot_take_action=cannot_take_action,
        cannot_take_reactions=cannot_take_reactions,
        cannot_move=cannot_move,
        movement_speed_cap=movement_speed_cap,
        only_regular_move=only_regular_move,
        cannot_be_targeted=cannot_be_targeted,
        area_attacks_can_target=area_attacks_can_target,
        only_improvised_or_grapple=only_improvised_or_grapple,
        cannot_overcharge=cannot_overcharge,
        reasons=reasons,
    )


class TurretDroneAttackInput(BaseModel):
    """Turret drone reaction attack input."""

    drone_id: str
    owner_id: str
    drone_position: HexCoord
    ally_attack_hit: bool
    ally_id: str
    ally_position: HexCoord
    target_id: str
    target_position: HexCoord
    drone_base_damage: int = 3
    drone_attack_bonus: int = 0
    drone_tier: int = 1
    force_rolls: dict = Field(default_factory=dict)


def resolve_turret_drone_attack(input: TurretDroneAttackInput) -> AttackSequenceResult:
    """Resolve turret drone reaction attack (PR2 7344-7358)."""
    if not input.ally_attack_hit:
        return AttackSequenceResult(
            attacker_id=input.drone_id,
            target_results=[],
            total_targets=0,
            targets_hit=0,
            targets_critical=0,
            total_damage_dealt=0,
            total_heat_dealt=0,
        )

    range_check = _hex_distance(input.drone_position, input.ally_position)
    if range_check > 10:
        return AttackSequenceResult(
            attacker_id=input.drone_id,
            target_results=[],
            total_targets=0,
            targets_hit=0,
            targets_critical=0,
            total_damage_dealt=0,
            total_heat_dealt=0,
        )

    tier_damage = input.drone_base_damage + (input.drone_tier - 1) * 2

    attack_input = AttackSequenceInput(
        attacker_id=input.drone_id,
        position=input.drone_position,
        target_ids=[input.target_id],
        attack_bonus=input.drone_attack_bonus,
        defense_value=10,
        base_damage=tier_damage,
        damage_type="kinetic",
        drone_assisted=False,
        target_positions={input.target_id: input.target_position},
        force_attack_roll=input.force_rolls.get("attack"),
    )

    return resolve_attack_sequence(attack_input)


class LatchDroneInput(BaseModel):
    """Latch drone mount attack input."""

    drone_id: str
    owner_id: str
    target_mech_id: str
    mode: Literal["mount", "active"]
    mount_attack_bonus: int = 0
    mount_damage: int = 0
    buff_type: Literal["evasion", "defense", "heat_cap"] | None = None
    buff_value: int | None = None
    target_is_stunned: bool = False
    target_is_destroyed: bool = False


class LatchDroneResult(BaseModel):
    """Result of latch drone action."""

    success: bool
    mode: str
    mount_hit: bool = False
    mount_damage: int = 0
    mount_critical: bool = False
    conditions_cleared: list[StatusType] = Field(default_factory=list)
    buff_applied: bool = False
    buff_details: dict | None = None


def resolve_latch_drone(input: LatchDroneInput) -> LatchDroneResult:
    """Resolve latch drone mount attack or active buff (PR2 7814-7832)."""
    result = LatchDroneResult(success=True, mode=input.mode)

    if input.mode == "mount":
        if input.target_is_stunned:
            result.mount_hit = True
            result.mount_damage = input.mount_damage
            result.conditions_cleared = ["stunned", "slowed"]
            result.buff_applied = True
            result.buff_details = {
                "type": "evasion",
                "value": 2,
            }
        else:
            result.mount_hit = False

    elif input.mode == "active":
        if input.buff_type and input.buff_value:
            result.buff_applied = True
            result.buff_details = {
                "type": input.buff_type,
                "value": input.buff_value,
            }

    return result


class RestockDroneInput(BaseModel):
    """Restock drone activation input."""

    drone_id: str
    owner_id: str
    activating_combatant_id: str
    activating_combatant_position: HexCoord
    action_choice: Literal["cool", "reload", "clear_condition"]
    condition_to_clear: StatusType | None = None


class RestockDroneResult(BaseModel):
    """Result of restock drone activation."""

    action: str
    heat_cleared: int = 0
    weapons_reloaded: list[str] = Field(default_factory=list)
    condition_cleared: StatusType | None = None
    success: bool = True


def resolve_restock_drone(input: RestockDroneInput) -> RestockDroneResult:
    """Resolve restock drone action (PR2 7834-7843)."""
    result = RestockDroneResult(action=input.action_choice)

    if input.action_choice == "cool":
        result.heat_cleared = 6

    elif input.action_choice == "reload":
        result.weapons_reloaded = ["primary"]

    elif input.action_choice == "clear_condition" and input.condition_to_clear:
        result.condition_cleared = input.condition_to_clear

    return result


class ICEOUTDroneInput(BaseModel):
    """ICEOUT drone burst area input."""

    drone_id: str
    owner_id: str
    drone_position: HexCoord
    zone_target_ids: list[str] = Field(default_factory=list)


class ICEOUTDroneResult(BaseModel):
    """Result of ICEOUT drone activation."""

    success: bool
    zone_created: bool = True
    zone_center: HexCoord
    zone_radius: int = 1
    tech_immunity_granted: bool = True
    affected_targets: list[str] = Field(default_factory=list)


def resolve_iceout_drone(input: ICEOUTDroneInput) -> ICEOUTDroneResult:
    """Resolve ICEOUT drone burst 1 zone with tech immunity (PR2 8653-8657)."""
    return ICEOUTDroneResult(
        success=True,
        zone_created=True,
        zone_center=input.drone_position,
        zone_radius=1,
        tech_immunity_granted=True,
        affected_targets=input.zone_target_ids,
    )


class TrackingDroneInput(BaseModel):
    """Tracking drone tech attack input."""

    drone_id: str
    owner_id: str
    drone_position: HexCoord
    target_id: str
    target_position: HexCoord
    tech_attack_bonus: int = 0
    target_e_defense: int = 10


class TrackingDroneResult(BaseModel):
    """Result of tracking drone tech attack."""

    hit: bool
    revealed_info: list[str] = Field(default_factory=list)
    hide_negated: bool = False
    invis_negated: bool = False


def resolve_tracking_drone(input: TrackingDroneInput) -> TrackingDroneResult:
    """Resolve tracking drone tech attack (PR2 8779-8789)."""
    attack_bonus = input.tech_attack_bonus
    hit = attack_bonus + 10 >= input.target_e_defense

    return TrackingDroneResult(
        hit=hit,
        revealed_info=["position", "defenses", "conditions"],
        hide_negated=hit,
        invis_negated=hit,
    )


class HiveDroneInput(BaseModel):
    """Hive drone burst 2 area input."""

    drone_id: str
    owner_id: str
    drone_position: HexCoord
    zone_target_ids: list[str] = Field(default_factory=list)


class HiveDroneResult(BaseModel):
    """Result of hive drone activation."""

    success: bool
    zone_created: bool = True
    zone_center: HexCoord
    zone_radius: int = 2
    soft_cover_granted: bool = True
    entry_damage: int = 0
    affected_targets: list[str] = Field(default_factory=list)


def resolve_hive_drone(input: HiveDroneInput) -> HiveDroneResult:
    """Resolve hive drone burst 2 zone (PR2 6787-6792)."""
    return HiveDroneResult(
        success=True,
        zone_created=True,
        zone_center=input.drone_position,
        zone_radius=2,
        soft_cover_granted=True,
        entry_damage=0,
        affected_targets=input.zone_target_ids,
    )
