"""Validation helpers for dice pool and countdown effects."""

from __future__ import annotations

from typing import Literal

from core.shared.models import FrozenModel
from core.shared.effects import DicePoolEffect, CountdownDieEffect, MechanicalEffect


class EffectValidationIssue(FrozenModel):
    """Issue raised when validating effect definitions."""

    severity: Literal["warning", "error"]
    message: str
    context: str | None = None


def merge_dice_pools_by_name(
    pools: list[DicePoolEffect],
) -> tuple[list[DicePoolEffect], list[EffectValidationIssue]]:
    """Merge dice pools with the same name and surface incompatibilities."""
    merged: dict[str, DicePoolEffect] = {}
    issues: list[EffectValidationIssue] = []
    for pool in pools:
        existing = merged.get(pool.pool_name)
        if not existing:
            merged[pool.pool_name] = pool
            continue
        merged[pool.pool_name] = _merge_dice_pool(existing, pool, issues)
    return list(merged.values()), issues


def merge_countdown_dice_by_name(
    countdown_dice: list[CountdownDieEffect],
) -> tuple[list[CountdownDieEffect], list[EffectValidationIssue]]:
    """Merge countdown dice with the same name and surface incompatibilities."""
    merged: dict[str, CountdownDieEffect] = {}
    issues: list[EffectValidationIssue] = []
    for die in countdown_dice:
        existing = merged.get(die.die_name)
        if not existing:
            merged[die.die_name] = die
            continue
        merged[die.die_name] = _merge_countdown_die(existing, die, issues)
    return list(merged.values()), issues


def validate_mechanical_effect(effect: MechanicalEffect) -> list[EffectValidationIssue]:
    """Validate effect definitions and return any dice pool/countdown issues."""
    issues: list[EffectValidationIssue] = []
    merged_pools, pool_issues = merge_dice_pools_by_name(effect.dice_pools)
    issues.extend(pool_issues)
    issues.extend(_validate_dice_pool_state(merged_pools))
    merged_countdown, countdown_issues = merge_countdown_dice_by_name(effect.countdown_dice)
    issues.extend(countdown_issues)
    issues.extend(_validate_countdown_state(merged_countdown))
    return issues


def merge_dice_pools_from_effects(
    effects: list[MechanicalEffect],
) -> tuple[list[DicePoolEffect], list[EffectValidationIssue]]:
    """Merge dice pools across multiple effects to resolve shared pool names."""
    pools: list[DicePoolEffect] = []
    for effect in effects:
        pools.extend(effect.dice_pools)
    return merge_dice_pools_by_name(pools)


def merge_countdown_dice_from_effects(
    effects: list[MechanicalEffect],
) -> tuple[list[CountdownDieEffect], list[EffectValidationIssue]]:
    """Merge countdown dice across multiple effects to resolve shared die names."""
    countdown_dice: list[CountdownDieEffect] = []
    for effect in effects:
        countdown_dice.extend(effect.countdown_dice)
    return merge_countdown_dice_by_name(countdown_dice)


def validate_mechanical_effects(effects: list[MechanicalEffect]) -> list[EffectValidationIssue]:
    """Validate multiple effects by resolving shared dice pools/countdown dice."""
    issues: list[EffectValidationIssue] = []
    merged_pools, pool_issues = merge_dice_pools_from_effects(effects)
    issues.extend(pool_issues)
    issues.extend(_validate_dice_pool_state(merged_pools))
    merged_countdown, countdown_issues = merge_countdown_dice_from_effects(effects)
    issues.extend(countdown_issues)
    issues.extend(_validate_countdown_state(merged_countdown))
    return issues


def _merge_dice_pool(
    existing: DicePoolEffect,
    incoming: DicePoolEffect,
    issues: list[EffectValidationIssue],
) -> DicePoolEffect:
    resolved: dict[str, object] = {}
    for field in (
        "die_size",
        "max_dice",
        "starting_dice",
        "weapon_id",
        "expires_on_scene_end",
        "lost_on_rest",
        "lost_on_full_repair",
    ):
        existing_set = field in existing.model_fields_set
        incoming_set = field in incoming.model_fields_set
        if existing_set and incoming_set and getattr(existing, field) != getattr(incoming, field):
            issues.append(
                EffectValidationIssue(
                    severity="error",
                    message=(
                        f"Dice pool '{existing.pool_name}' has conflicting {field}: "
                        f"{getattr(existing, field)!r} vs {getattr(incoming, field)!r}"
                    ),
                    context=existing.pool_name,
                )
            )
        if incoming_set and not existing_set:
            resolved[field] = getattr(incoming, field)
        else:
            resolved[field] = getattr(existing, field)
    condition_set_existing = "condition" in existing.model_fields_set
    condition_set_incoming = "condition" in incoming.model_fields_set
    if condition_set_existing and condition_set_incoming and existing.condition != incoming.condition:
        issues.append(
            EffectValidationIssue(
                severity="warning",
                message=(
                    f"Dice pool '{existing.pool_name}' has conflicting conditions: "
                    f"{existing.condition!r} vs {incoming.condition!r}"
                ),
                context=existing.pool_name,
            )
        )
    condition = existing.condition if condition_set_existing else (
        incoming.condition if condition_set_incoming else None
    )
    return DicePoolEffect(
        pool_name=existing.pool_name,
        die_size=resolved["die_size"],
        max_dice=resolved["max_dice"],
        starting_dice=resolved["starting_dice"],
        gain_triggers=[*existing.gain_triggers, *incoming.gain_triggers],
        spend_options=[*existing.spend_options, *incoming.spend_options],
        weapon_id=resolved["weapon_id"],
        expires_on_scene_end=resolved["expires_on_scene_end"],
        lost_on_rest=resolved["lost_on_rest"],
        lost_on_full_repair=resolved["lost_on_full_repair"],
        condition=condition,
    )


def _merge_countdown_die(
    existing: CountdownDieEffect,
    incoming: CountdownDieEffect,
    issues: list[EffectValidationIssue],
) -> CountdownDieEffect:
    resolved: dict[str, object] = {}
    for field in (
        "die_size",
        "starting_value",
        "minimum_value",
        "spend_requires_value",
        "reset_value",
        "expires_on_scene_end",
        "lost_on_rest",
        "lost_on_full_repair",
    ):
        existing_set = field in existing.model_fields_set
        incoming_set = field in incoming.model_fields_set
        if existing_set and incoming_set and getattr(existing, field) != getattr(incoming, field):
            issues.append(
                EffectValidationIssue(
                    severity="error",
                    message=(
                        f"Countdown die '{existing.die_name}' has conflicting {field}: "
                        f"{getattr(existing, field)!r} vs {getattr(incoming, field)!r}"
                    ),
                    context=existing.die_name,
                )
            )
        if incoming_set and not existing_set:
            resolved[field] = getattr(incoming, field)
        else:
            resolved[field] = getattr(existing, field)
    condition_set_existing = "condition" in existing.model_fields_set
    condition_set_incoming = "condition" in incoming.model_fields_set
    if condition_set_existing and condition_set_incoming and existing.condition != incoming.condition:
        issues.append(
            EffectValidationIssue(
                severity="warning",
                message=(
                    f"Countdown die '{existing.die_name}' has conflicting conditions: "
                    f"{existing.condition!r} vs {incoming.condition!r}"
                ),
                context=existing.die_name,
            )
        )
    condition = existing.condition if condition_set_existing else (
        incoming.condition if condition_set_incoming else None
    )
    return CountdownDieEffect(
        die_name=existing.die_name,
        die_size=resolved["die_size"],
        starting_value=resolved["starting_value"],
        minimum_value=resolved["minimum_value"],
        decrement_triggers=[*existing.decrement_triggers, *incoming.decrement_triggers],
        spend_options=[*existing.spend_options, *incoming.spend_options],
        spend_requires_value=resolved["spend_requires_value"],
        reset_value=resolved["reset_value"],
        expires_on_scene_end=resolved["expires_on_scene_end"],
        lost_on_rest=resolved["lost_on_rest"],
        lost_on_full_repair=resolved["lost_on_full_repair"],
        condition=condition,
    )


def _validate_dice_pool_state(pools: list[DicePoolEffect]) -> list[EffectValidationIssue]:
    issues: list[EffectValidationIssue] = []
    for pool in pools:
        if not pool.spend_options:
            continue
        if pool.max_dice is not None:
            for spend_option in pool.spend_options:
                if spend_option.dice_cost is None:
                    continue
                if spend_option.dice_cost > pool.max_dice:
                    issues.append(
                        EffectValidationIssue(
                            severity="error",
                            message=(
                                f"Dice pool '{pool.pool_name}' cannot spend {spend_option.dice_cost} "
                                f"because max_dice is {pool.max_dice}"
                            ),
                            context=pool.pool_name,
                        )
                    )
        if pool.starting_dice == 0 and not pool.gain_triggers:
            issues.append(
                EffectValidationIssue(
                    severity="warning",
                    message=(
                        f"Dice pool '{pool.pool_name}' has spend options but no starting dice "
                        "or gain triggers"
                    ),
                    context=pool.pool_name,
                )
            )
    return issues


def _validate_countdown_state(
    countdown_dice: list[CountdownDieEffect],
) -> list[EffectValidationIssue]:
    issues: list[EffectValidationIssue] = []
    for die in countdown_dice:
        if die.spend_requires_value > die.die_size:
            issues.append(
                EffectValidationIssue(
                    severity="error",
                    message=(
                        f"Countdown die '{die.die_name}' requires value {die.spend_requires_value} "
                        f"but die_size is {die.die_size}"
                    ),
                    context=die.die_name,
                )
            )
        if die.spend_options and not die.decrement_triggers and die.starting_value > die.spend_requires_value:
            issues.append(
                EffectValidationIssue(
                    severity="warning",
                    message=(
                        f"Countdown die '{die.die_name}' has spend options but no decrement triggers"
                    ),
                    context=die.die_name,
                )
            )
    return issues
