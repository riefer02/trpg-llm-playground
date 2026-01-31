"""Core model serialization helpers for API responses."""

from typing import Any

from core.character import Character, MechConfiguration
from core.mech.build import MechDerivedStats
from core.pilot import Pilot


def _serialize_mech_config(mech: MechConfiguration) -> dict[str, Any]:
    return {
        "id": mech.id,
        "name": mech.name,
        "frame_id": mech.frame_id,
        "build": mech.build.model_dump(mode="json"),
        "damage_state": mech.damage_state.model_dump(mode="json")
        if mech.damage_state
        else None,
    }


def _serialize_mech_stats(stats: MechDerivedStats) -> dict[str, Any]:
    return {
        "hp": stats.hp,
        "armor": stats.armor,
        "evasion": stats.evasion,
        "e_defense": stats.e_defense,
        "speed": stats.speed,
        "sensor_range": stats.sensor_range,
        "tech_attack": stats.tech_attack,
        "heat_cap": stats.heat_cap,
        "repair_cap": stats.repair_cap,
        "system_points": stats.system_points,
        "save_target": stats.save_target,
        "size": stats.size,
    }


def _serialize_pilot_base(core_pilot: Pilot) -> dict[str, Any]:
    return {
        "callsign": core_pilot.callsign,
        "name": core_pilot.name,
        "level": core_pilot.level,
        "xp": core_pilot.xp,
        "xp_to_next_level": core_pilot.xp_to_next_level,
        "level_progress": core_pilot.level_progress,
        "skills": core_pilot.skills.as_dict(),
        "triggers": [t.model_dump() for t in core_pilot.triggers],
        "talents": [t.model_dump() for t in core_pilot.talents],
        "licenses": [lic.model_dump() for lic in core_pilot.licenses],
        "core_bonuses": [cb.model_dump() for cb in core_pilot.core_bonuses],
        "background": core_pilot.background.model_dump()
        if core_pilot.background
        else None,
        "pilot_gear": core_pilot.pilot_gear.model_dump()
        if core_pilot.pilot_gear
        else None,
        "notes": core_pilot.notes,
        "salvage": core_pilot.salvage,
    }


def serialize_pilot_response_fields(core_pilot: Pilot) -> dict[str, Any]:
    data = _serialize_pilot_base(core_pilot)
    data.update(
        {
            "grit": core_pilot.grit,
            "hp": core_pilot.hp,
            "armor": core_pilot.armor,
            "evasion": core_pilot.evasion,
            "e_defense": core_pilot.e_defense,
            "speed": core_pilot.speed,
            "save_target": core_pilot.save_target,
            "attack_bonus": core_pilot.attack_bonus,
        }
    )
    return data


def serialize_character_response_fields(core_char: Character) -> dict[str, Any]:
    data = _serialize_pilot_base(core_char.pilot)
    data.update(
        {
            "pilot_id": core_char.pilot.id,
            "grit": core_char.pilot.grit,
            "pilot_hp": core_char.pilot.hp,
            "mechs": [_serialize_mech_config(m) for m in core_char.mechs],
            "active_mech_id": core_char.active_mech_id,
            "active_mech_stats": _serialize_mech_stats(core_char.active_mech_stats)
            if core_char.active_mech_stats
            else None,
            "core_bonus_effects": [
                effect.model_dump() for effect in core_char.core_bonus_effects
            ],
        }
    )
    return data
