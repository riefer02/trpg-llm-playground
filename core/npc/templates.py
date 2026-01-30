"""NPC template utilities for typed Lancer mechanics.

This module provides helpers for filtering NPC templates by role,
creating template variants, and working with NPC special classes.
"""

from core.shared.models import FrozenModel
from core.shared.id_helpers import TemplateIdField
from core.npc.models import NPCTemplate, NPCStats, NPCStatsBase, NPCRole


def get_templates_by_role(role: NPCRole) -> list[NPCTemplate]:
    """Get all NPC templates of a specific role.

    Args:
        role: The role to filter by (striker, defender, controller, supporter)

    Returns:
        List of templates with the specified role
    """
    from core.npc.compendium import NPC_TEMPLATES

    return [t for t in NPC_TEMPLATES if t.role == role]


def get_striker_templates() -> list[NPCTemplate]:
    """Get all striker NPCs (highest targeting priority)."""
    return get_templates_by_role("striker")


def get_defender_templates() -> list[NPCTemplate]:
    """Get all defender NPCs (protect allies, high armor)."""
    return get_templates_by_role("defender")


def get_controller_templates() -> list[NPCTemplate]:
    """Get all controller NPCs (area denial, debuffs)."""
    return get_templates_by_role("controller")


def get_supporter_templates() -> list[NPCTemplate]:
    """Get all supporter NPCs (healing, buffs)."""
    return get_templates_by_role("supporter")


class NPCTemplateVariant(FrozenModel):
    """A variant of a base template with modified stats.

    Variants allow creating elite, veteran, or other enhanced versions
    of base templates without duplicating template definitions.
    """

    base_template_id: TemplateIdField
    variant_name: str
    hp_modifier: int = 0
    evasion_modifier: int = 0
    e_defense_modifier: int = 0
    armor_modifier: int = 0
    speed_modifier: int = 0
    sensor_modifier: int = 0
    save_modifier: int = 0
    tech_attack_modifier: int = 0
    role_override: NPCRole | None = None


def create_variant(template: NPCTemplate, variant: NPCTemplateVariant) -> NPCTemplate:
    """Create a variant of an NPC template.

    Args:
        template: The base template
        variant: The variant specification

    Returns:
        A new NPCTemplate with modified stats
    """
    if variant.base_template_id != template.id:
        raise ValueError(
            f"Variant base_template_id '{variant.base_template_id}' does not match template id '{template.id}'"
        )

    new_base = NPCStatsBase(
        size=template.stats.base.size,
        hp_base=template.stats.base.hp_base + variant.hp_modifier,
        evasion_base=template.stats.base.evasion_base + variant.evasion_modifier,
        e_defense_base=template.stats.base.e_defense_base + variant.e_defense_modifier,
        armor_base=template.stats.base.armor_base + variant.armor_modifier,
        speed_base=template.stats.base.speed_base + variant.speed_modifier,
        sensor_range=template.stats.base.sensor_range + variant.sensor_modifier,
        save_bonus=template.stats.base.save_bonus + variant.save_modifier,
        tech_attack=template.stats.base.tech_attack + variant.tech_attack_modifier,
    )

    return NPCTemplate(
        id=f"{template.id}_{variant.variant_name}",
        name=f"{template.name} {variant.variant_name.title()}",
        description=template.description,
        npc_class=template.npc_class,
        tier=template.tier,
        role=variant.role_override or template.role,
        stats=NPCStats(
            base=new_base,
            scaling=template.stats.scaling,
        ),
        abilities=list(template.abilities),
        gear=list(template.gear),
        effects=template.effects,
        tags=list(template.tags),
    )


ELITE_VARIANT = NPCTemplateVariant(
    base_template_id="",
    variant_name="elite",
    hp_modifier=5,
    evasion_modifier=1,
    e_defense_modifier=1,
    armor_modifier=0,
    save_modifier=1,
)

VETERAN_VARIANT = NPCTemplateVariant(
    base_template_id="",
    variant_name="veteran",
    hp_modifier=8,
    evasion_modifier=2,
    e_defense_modifier=2,
    armor_modifier=1,
    save_modifier=2,
    tech_attack_modifier=1,
)

BOSS_VARIANT = NPCTemplateVariant(
    base_template_id="",
    variant_name="boss",
    hp_modifier=15,
    evasion_modifier=2,
    e_defense_modifier=2,
    armor_modifier=2,
    save_modifier=3,
    tech_attack_modifier=1,
)


def create_elite_variant(template: NPCTemplate) -> NPCTemplate:
    """Create an elite variant of a template."""
    variant = NPCTemplateVariant(
        base_template_id=template.id,
        variant_name="elite",
        hp_modifier=5,
        evasion_modifier=1,
        e_defense_modifier=1,
        save_modifier=1,
    )
    return create_variant(template, variant)


def create_veteran_variant(template: NPCTemplate) -> NPCTemplate:
    """Create a veteran variant of a template."""
    variant = NPCTemplateVariant(
        base_template_id=template.id,
        variant_name="veteran",
        hp_modifier=8,
        evasion_modifier=2,
        e_defense_modifier=2,
        armor_modifier=1,
        save_modifier=2,
        tech_attack_modifier=1,
    )
    return create_variant(template, variant)


def create_boss_variant(template: NPCTemplate) -> NPCTemplate:
    """Create a boss variant of a template."""
    variant = NPCTemplateVariant(
        base_template_id=template.id,
        variant_name="boss",
        hp_modifier=15,
        evasion_modifier=2,
        e_defense_modifier=2,
        armor_modifier=2,
        save_modifier=3,
        tech_attack_modifier=1,
    )
    return create_variant(template, variant)


from core.npc.enums import NPCSpecialClass


def get_special_class_description(npc_class: NPCSpecialClass) -> str:
    """Get a description of a special NPC class.

    Args:
        npc_class: The special class identifier

    Returns:
        Description of the class
    """
    descriptions = {
        "human": "Pilot-scale human opponent (not a mech)",
        "infantry_squad": "Group of infantry soldiers operating together",
        "monstrosity": "Biological or alien threat with unusual abilities",
        "ultra": "Boss-level threat with multiple phases or abilities",
        "elite": "Enhanced version of a standard enemy type",
        "grunt": "Basic enemy soldier with minimal special abilities",
        "veteran": "Experienced combatant with improved stats",
        "exotic": "Non-standard enemy with unusual mechanics",
        "drone": "Autonomous robotic unit",
        "mercenary": "Professional soldier fighting for pay",
        "commander": "Leadership unit that buffs nearby allies",
        "pirate": "Ragged but dangerous raider",
        "spacer": "Zero-g specialist with space/void experience",
        "vehicle": "Non-mech ground vehicle",
        "ship": "Starship-scale vessel",
    }
    return descriptions.get(npc_class, "Unknown special class")
