"""Example NPC builds and combat scenarios for validation and reference."""

from core.npc.state import NPCState
from core.npc.compendium import (
    get_npc_template,
    NPC_TEMPLATES,
)
from core.npc.validation import validate_npc_in_combat, validate_npc_template


def _assert_no_errors(validation) -> None:
    """Assert that validation has no error-level issues."""
    errors = [issue for issue in validation.issues if issue.severity == "error"]
    assert not errors, f"Validation errors: {errors}"


def create_npc_by_id(
    template_id: str, instance_id: str, name: str | None = None
) -> NPCState:
    """Create an NPC instance from a template ID.

    Args:
        template_id: The template ID to use
        instance_id: Unique ID for this instance
        name: Optional name override

    Returns:
        A new NPCState instance

    Raises:
        ValueError: If template_id not found
    """
    template = get_npc_template(template_id)
    if template is None:
        raise ValueError(f"NPC template '{template_id}' not found")
    return NPCState.from_template(template, instance_id, name)


def create_gms_grunt_squad() -> list[NPCState]:
    """Create a standard GMS grunt squad (4 NPCs)."""
    return [
        create_npc_by_id("gms_grunt_t1", "grunt_1", "Security Alpha"),
        create_npc_by_id("gms_grunt_t1", "grunt_2", "Security Beta"),
        create_npc_by_id("gms_grunt_t1", "grunt_3", "Security Gamma"),
        create_npc_by_id("gms_grunt_t1", "grunt_4", "Security Delta"),
    ]


def create_ipsn_raid_party() -> list[NPCState]:
    """Create an IPS-N raid party (mixed tiers)."""
    return [
        create_npc_by_id("ipsn_grunt_t1", "raider_1", "Raider Lead"),
        create_npc_by_id("ipsn_grunt_t1", "raider_2", "Raider Scout"),
        create_npc_by_id("ipsn_boss_t3", "raider_boss", "IPS-N Dreadnought"),
    ]


def create_horus_encounter() -> list[NPCState]:
    """Create a HORUS-themed encounter."""
    return [
        create_npc_by_id("horus_elite_t3", "harbinger_1", "HORUS Harbinger"),
        create_npc_by_id("ssc_specialist_t2", "specter_1", "SSC Specter"),
    ]


def create_boss_encounter() -> list[NPCState]:
    """Create a boss-level encounter."""
    return [
        create_npc_by_id("ha_boss_t3", "emperor", "HA Emperor"),
        create_npc_by_id("gms_grunt_t2", "elite_guard_1", "GMS Vanguard"),
        create_npc_by_id("gms_grunt_t2", "elite_guard_2", "GMS Vanguard"),
    ]


def evaluate_gms_grunt_example() -> dict:
    """Evaluate a GMS grunt NPC build.

    Returns:
        Dict with validation results and NPC state
    """
    npc = create_npc_by_id("gms_grunt_t1", "test_grunt")
    validation = validate_npc_in_combat(npc)
    return {
        "npc_id": npc.id,
        "npc_name": npc.name,
        "npc_class": npc.npc_class,
        "tier": npc.tier,
        "hp_max": npc.stats.hp_max,
        "evasion": npc.stats.evasion,
        "e_defense": npc.stats.e_defense,
        "armor": npc.stats.armor,
        "valid": validation.valid,
        "issues": len(validation.issues),
    }


def evaluate_ipsn_boss_example() -> dict:
    """Evaluate an IPS-N boss NPC build.

    Returns:
        Dict with validation results and NPC state
    """
    npc = create_npc_by_id("ipsn_boss_t3", "test_dreadnought")
    validation = validate_npc_in_combat(npc)
    return {
        "npc_id": npc.id,
        "npc_name": npc.name,
        "npc_class": npc.npc_class,
        "tier": npc.tier,
        "hp_max": npc.stats.hp_max,
        "evasion": npc.stats.evasion,
        "e_defense": npc.stats.e_defense,
        "armor": npc.stats.armor,
        "structures": npc.structure_current,
        "valid": validation.valid,
        "issues": len(validation.issues),
    }


def evaluate_all_npc_templates() -> dict:
    """Evaluate all NPC templates in the compendium.

    Returns:
        Dict with template count and any validation failures
    """
    failures = []
    for template in NPC_TEMPLATES:
        template_validation = validate_npc_template(template)
        if not template_validation.valid:
            failures.append(
                {
                    "template_id": template.id,
                    "issues": [
                        i.code
                        for i in template_validation.issues
                        if i.severity == "error"
                    ],
                }
            )
    return {
        "template_count": len(NPC_TEMPLATES),
        "failures": failures,
        "all_valid": len(failures) == 0,
    }
