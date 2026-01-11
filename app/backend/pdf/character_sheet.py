"""HTML template rendering for character PDF exports."""

from __future__ import annotations

from html import escape
from pathlib import Path
from string import Template
from typing import Iterable

from core.character import Character
from core.mech.compendium import (
    SYSTEM_DEFINITIONS_BY_ID,
    WEAPON_DEFINITIONS_BY_ID,
    get_frame_definition,
)
from core.pilot.gear import get_pilot_gear_definition

_TEMPLATE = Template(
    (Path(__file__).parent / "templates" / "character_sheet.html").read_text()
)


def render_character_sheet_pdf(character: Character) -> bytes:
    """Render a character sheet PDF using WeasyPrint."""
    from weasyprint import HTML

    html = render_character_sheet_html(character)
    return HTML(string=html).write_pdf()


def render_character_sheet_html(character: Character) -> str:
    """Render the HTML for a character sheet."""
    pilot = character.pilot
    active_mech = (
        character.get_mech(character.active_mech_id)
        if character.active_mech_id
        else None
    )
    stats = character.active_mech_stats

    frame_name = "None"
    if active_mech:
        frame_def = get_frame_definition(active_mech.frame_id)
        frame_name = frame_def.name if frame_def else _format_label(active_mech.frame_id)

    pilot_gear = pilot.pilot_gear

    pilot_gear_clothing = _gear_name(pilot_gear.clothing) if pilot_gear else "None"
    pilot_gear_armor = _gear_name(pilot_gear.armor) if pilot_gear else "None"

    pilot_weapon_names = []
    if pilot_gear and pilot_gear.weapons:
        pilot_weapon_names = [_gear_name(weapon_id) for weapon_id in pilot_gear.weapons]

    pilot_item_names = []
    if pilot_gear and pilot_gear.gear:
        pilot_item_names = [_gear_name(item_id) for item_id in pilot_gear.gear]

    mech_weapons = []
    mech_systems = []
    sp_spent = 0

    if active_mech:
        for weapon in active_mech.build.weapons:
            definition = WEAPON_DEFINITIONS_BY_ID.get(weapon.weapon_id)
            name = definition.name if definition else _format_label(weapon.weapon_id)
            mech_weapons.append(
                f"Mount {weapon.mount_index + 1}: {name} ({weapon.weapon_size})"
            )

        for system in active_mech.build.systems:
            definition = SYSTEM_DEFINITIONS_BY_ID.get(system.system_id)
            name = definition.name if definition else _format_label(system.system_id)
            sp_cost = system.sp_cost if system.sp_cost is not None else (definition.sp_cost if definition else 0)
            mech_systems.append(f"{name} (SP {sp_cost})")
            sp_spent += sp_cost

    sp_limit = stats.system_points if stats else 0
    sp_usage = f"{sp_spent} / {sp_limit}"

    return _TEMPLATE.safe_substitute(
        callsign=_safe_text(pilot.callsign),
        name=_safe_text(pilot.name or "Unnamed"),
        level=pilot.level,
        background=_safe_text(pilot.background.name if pilot.background else "None"),
        grit=pilot.grit,
        pilot_hp=pilot.hp,
        notes=_safe_text(pilot.notes or "None"),
        skill_hull=pilot.skills.hull,
        skill_agility=pilot.skills.agility,
        skill_systems=pilot.skills.systems,
        skill_engineering=pilot.skills.engineering,
        active_mech_name=_safe_text(active_mech.name if active_mech else "None"),
        active_mech_frame=_safe_text(frame_name),
        mech_hp=stats.hp if stats else "-",
        mech_armor=stats.armor if stats else "-",
        mech_evasion=stats.evasion if stats else "-",
        mech_e_defense=stats.e_defense if stats else "-",
        mech_speed=stats.speed if stats else "-",
        mech_heat_cap=stats.heat_cap if stats else "-",
        mech_system_points=stats.system_points if stats else "-",
        mech_save_target=stats.save_target if stats else "-",
        mech_size=_safe_text(stats.size.replace("size_", "") if stats else "-"),
        pilot_gear_clothing=_safe_text(pilot_gear_clothing),
        pilot_gear_armor=_safe_text(pilot_gear_armor),
        pilot_gear_weapons=_render_list(pilot_weapon_names),
        pilot_gear_items=_render_list(pilot_item_names),
        mech_weapons=_render_list(mech_weapons),
        mech_systems=_render_list(mech_systems),
        mech_sp_usage=_safe_text(sp_usage),
        licenses=_render_list(
            [
                f"{_format_license(lic.license_id)} - Rank {lic.rank}"
                for lic in pilot.licenses
            ]
        ),
        core_bonuses=_render_list(
            [_format_label(cb.core_bonus_id) for cb in pilot.core_bonuses]
        ),
        talents=_render_list(
            [
                f"{_format_label(talent.talent_id)} - Rank {talent.rank}"
                for talent in pilot.talents
            ]
        ),
        triggers=_render_list(
            [
                f"{_format_label(trigger.trigger_id)} - +{trigger.rank}"
                for trigger in pilot.triggers
            ]
        ),
    )


def _safe_text(value: str) -> str:
    return escape(value)


def _gear_name(gear_id: str | None) -> str:
    if not gear_id:
        return "None"
    definition = get_pilot_gear_definition(gear_id)
    return definition.name if definition else _format_label(gear_id)


def _render_list(items: Iterable[str]) -> str:
    values = list(items)
    if not values:
        return '<div class="empty">None</div>'
    list_items = "\n".join(f"<li>{escape(item)}</li>" for item in values)
    return f"<ul>{list_items}</ul>"


def _format_label(value: str) -> str:
    return value.replace("gms_", "GMS ").replace("_", " ")


def _format_license(value: str) -> str:
    return value.replace("_", " ").upper()
