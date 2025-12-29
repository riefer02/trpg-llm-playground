"""Helpers to build combat actions from weapon definitions."""

from __future__ import annotations

from typing import Iterable

from core.shared.enums import ActionType, AttackType, RangeType
from core.shared.effects import ReactionTriggerEvent
from core.mech.combat_rules import AttackPatternDefinition
from core.mech.combat_state import ActionUse
from core.mech.grid import HexCoord, HexPosition
from core.mech.weapon import MechWeaponDefinition, WeaponTag


def build_action_use_from_weapon(
    *,
    action_id: str,
    action_type: ActionType,
    weapon: MechWeaponDefinition,
    target_id: str | None = None,
    target_position: HexPosition | None = None,
    target_ids: list[str] | None = None,
    target_positions: list[HexPosition] | None = None,
    range_type: RangeType | None = None,
    attack_type_override: AttackType | None = None,
    weapon_count: int | None = None,
    uses_aux_bonus_attack: bool | None = None,
    area_origin: HexPosition | None = None,
    area_direction: HexCoord | None = None,
    area_affected: list[HexCoord] | None = None,
    reaction_trigger: ReactionTriggerEvent | None = None,
    heat_generated: int | None = None,
) -> ActionUse:
    """Create an ActionUse enriched with weapon tags and patterns."""
    weapon_tags = [tag.tag for tag in weapon.tags]
    pattern = _pattern_from_tags(weapon.tags)
    attack_type = attack_type_override
    if attack_type is None and weapon.weapon_type == "melee":
        attack_type = "melee"

    preferred_range = range_type
    if preferred_range is None and action_id == "overwatch":
        preferred_range = "threat"
    range_spaces = _resolve_weapon_range(
        weapon,
        preferred=preferred_range,
        area_pattern=pattern,
    )

    return ActionUse(
        action_id=action_id,
        action_type=action_type,
        target_id=target_id,
        target_position=target_position,
        target_ids=target_ids or [],
        target_positions=target_positions or [],
        range_spaces=range_spaces,
        attack_type_override=attack_type,
        weapon_tags=weapon_tags,
        area_pattern=pattern,
        area_origin=area_origin,
        area_direction=area_direction,
        area_affected=area_affected or [],
        weapon_count=weapon_count,
        uses_superheavy=weapon.size == "superheavy",
        uses_aux_bonus_attack=uses_aux_bonus_attack,
        reaction_trigger=reaction_trigger,
        heat_generated=heat_generated,
    )


def build_action_use_from_weapon_id(
    *,
    action_id: str,
    action_type: ActionType,
    weapon_id: str,
    weapon_definitions: dict[str, MechWeaponDefinition] | None = None,
    **kwargs,
) -> ActionUse:
    """Create an ActionUse from a weapon ID and definition map."""
    if weapon_definitions is None:
        from core.mech.compendium import WEAPON_DEFINITIONS_BY_ID

        weapon_definitions = WEAPON_DEFINITIONS_BY_ID

    weapon = weapon_definitions.get(weapon_id)
    if not weapon:
        raise ValueError(f"Unknown weapon id: {weapon_id}")

    return build_action_use_from_weapon(
        action_id=action_id,
        action_type=action_type,
        weapon=weapon,
        **kwargs,
    )


def _pattern_from_tags(tags: Iterable[WeaponTag]) -> AttackPatternDefinition | None:
    for tag in tags:
        if tag.tag in {"line", "cone", "blast", "burst"} and tag.value is not None:
            return AttackPatternDefinition(pattern=tag.tag, size=tag.value)
    return None


def _resolve_weapon_range(
    weapon: MechWeaponDefinition,
    preferred: RangeType | None,
    area_pattern: AttackPatternDefinition | None,
) -> int | None:
    if area_pattern and area_pattern.pattern == "burst":
        return None

    range_by_type = {entry.range_type: entry.value for entry in weapon.ranges}
    if preferred and preferred in range_by_type:
        return range_by_type[preferred]

    if "range" in range_by_type:
        return range_by_type["range"]
    if "threat" in range_by_type:
        return range_by_type["threat"]

    threat_tag = next((tag for tag in weapon.tags if tag.tag == "threat"), None)
    if threat_tag and threat_tag.value is not None:
        return threat_tag.value

    return None
