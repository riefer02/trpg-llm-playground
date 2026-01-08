"""Example mech builds for validation and reference."""

from core.mech.build import MechBuild, build_mech_from_compendium, compute_mech_stats
from core.mech.frame import MechFrameDefinition
from core.mech.build_validation import MechBuildValidation, validate_mech_build
from core.mech.compendium import (
    get_frame_definition,
    SYSTEM_DEFINITIONS_BY_ID,
    WEAPON_DEFINITIONS_BY_ID,
)
from core.pilot.skill import SkillSet
from core.shared.effects import MechanicalEffect, StatModifier, ReactionTriggerEffect
from core.mech.combat_state import (
    MechCombatScenario,
    CombatantState,
    CombatStats,
    CombatResources,
    CombatRound,
    CombatTurn,
    ActionUse,
    WeaponState,
    WeaponMountState,
    MechSystemState,
    MechInventory,
)
from core.mech.validation.combat_validation import (
    CombatValidation,
    validate_combat_scenario,
)
from core.mech.grid import HexCoord, HexPosition
from core.mech.terrain import TerrainHex, TerrainMap
from core.mech.combat_action_builder import build_action_use_from_weapon
from core.mech.weapon import MechWeaponDefinition, WeaponRange, WeaponTag
from core.shared.rolls import ContestedCheck, SkillCheck, RollModifiers


def build_example_everest_frame() -> MechFrameDefinition:
    """Fetch the GMS Everest frame definition."""
    frame = get_frame_definition("gms_everest")
    if not frame:
        raise ValueError("GMS Everest frame definition not found")
    return frame


def build_example_raleigh_frame() -> MechFrameDefinition:
    """Fetch the IPS-N Raleigh frame definition."""
    frame = get_frame_definition("ipsn_raleigh")
    if not frame:
        raise ValueError("IPS-N Raleigh frame definition not found")
    return frame


def build_oda_ll0_mech_example() -> tuple[
    MechFrameDefinition, MechBuild, SkillSet, int, list[MechanicalEffect]
]:
    """Build Oda's LL0 Everest mech loadout."""
    frame = build_example_everest_frame()
    build = build_mech_from_compendium(
        frame_id=frame.id,
        weapon_mounts=[
            (0, "anti_material_rifle"),
            (1, "assault_rifle"),
            (2, "tactical_knife"),
            (2, "tactical_knife"),
        ],
        system_ids=[
            "gms_hex_charges",
            "gms_jump_jet_burst",
            "personalizations",
            "gms_custom_paint_job",
        ],
    )
    skills = SkillSet(hull=2, agility=0, systems=0, engineering=0)
    grit = 0
    bonus_effects = [MechanicalEffect(stat_mods=[StatModifier(stat="hp", value=2)])]
    return frame, build, skills, grit, bonus_effects


def build_oda_ll3_mech_example() -> tuple[
    MechFrameDefinition, MechBuild, SkillSet, int, list[MechanicalEffect]
]:
    """Build Oda's LL3 Raleigh mech loadout."""
    frame = build_example_raleigh_frame()
    build = build_mech_from_compendium(
        frame_id=frame.id,
        weapon_mounts=[
            (0, "anti_material_rifle"),
            (1, "assault_rifle"),
            (2, "hand_cannon"),
            (2, "hand_cannon"),
        ],
        system_ids=[
            "gms_hex_charges",
            "ipsn_breaching_charges",
            "gms_jump_jet_burst",
            "gms_custom_paint_job",
        ],
    )
    skills = SkillSet(hull=5, agility=0, systems=0, engineering=0)
    grit = 2
    bonus_effects = [MechanicalEffect(stat_mods=[StatModifier(stat="hp", value=5)])]
    return frame, build, skills, grit, bonus_effects


def evaluate_oda_ll0_mech_example() -> MechBuildValidation:
    """Validate the LL0 Everest build."""
    frame, build, skills, grit, effects = build_oda_ll0_mech_example()
    return validate_mech_build(
        frame,
        build,
        skills,
        grit,
        weapon_definitions=WEAPON_DEFINITIONS_BY_ID,
        system_definitions=SYSTEM_DEFINITIONS_BY_ID,
        bonus_effects=effects,
    )


def evaluate_oda_ll3_mech_example() -> MechBuildValidation:
    """Validate the LL3 Raleigh build."""
    frame, build, skills, grit, effects = build_oda_ll3_mech_example()
    return validate_mech_build(
        frame,
        build,
        skills,
        grit,
        weapon_definitions=WEAPON_DEFINITIONS_BY_ID,
        system_definitions=SYSTEM_DEFINITIONS_BY_ID,
        bonus_effects=effects,
    )


def compute_oda_ll0_stats() -> dict[str, int | str]:
    """Compute LL0 mech stats for the example."""
    frame, _, skills, grit, effects = build_oda_ll0_mech_example()
    stats = compute_mech_stats(frame, skills, grit, bonus_effects=effects)
    return stats.model_dump()


def compute_oda_ll3_stats() -> dict[str, int | str]:
    """Compute LL3 mech stats for the example."""
    frame, _, skills, grit, effects = build_oda_ll3_mech_example()
    stats = compute_mech_stats(frame, skills, grit, bonus_effects=effects)
    return stats.model_dump()


def build_example_combat_scenario() -> MechCombatScenario:
    """Build a small combat scenario for validation."""
    player = CombatantState(
        id="alpha",
        name="Alpha",
        side="players",
        kind="mech",
        stats=CombatStats(
            size="size_1",
            hp_max=12,
            evasion=8,
            e_defense=8,
            armor=0,
            speed=4,
            sensor_range=10,
            tech_attack=0,
        ),
        resources=CombatResources(
            hp_current=12,
            heat_current=0,
            heat_cap=6,
            structure_current=4,
            stress_current=4,
            repairs_remaining=4,
        ),
        position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        statuses=[],
        conditions=[],
    )
    hostile = CombatantState(
        id="bravo",
        name="Bravo",
        side="hostiles",
        kind="mech",
        stats=CombatStats(
            size="size_1",
            hp_max=10,
            evasion=10,
            e_defense=8,
            armor=1,
            speed=3,
            sensor_range=8,
            tech_attack=0,
        ),
        resources=CombatResources(
            hp_current=10,
            heat_current=0,
            heat_cap=6,
            structure_current=4,
            stress_current=4,
            repairs_remaining=4,
        ),
        position=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
        statuses=[],
        conditions=[],
    )
    round_one = CombatRound(
        round_index=1,
        turns=[
            CombatTurn(
                actor_id="alpha",
                move_used=True,
                actions=[
                    ActionUse(
                        action_id="skirmish",
                        action_type="quick",
                        target_id="bravo",
                        attack_type_override="ranged",
                        range_spaces=10,
                        weapon_count=1,
                        uses_superheavy=False,
                        uses_aux_bonus_attack=False,
                    ),
                    ActionUse(
                        action_id="lock_on",
                        action_type="quick",
                        target_id="bravo",
                    ),
                    ActionUse(
                        action_id="overcharge", action_type="free", heat_generated=1
                    ),
                    ActionUse(
                        action_id="boost",
                        action_type="quick",
                        granted_by_overcharge=True,
                    ),
                ],
            ),
            CombatTurn(
                actor_id="bravo",
                move_used=True,
                actions=[
                    ActionUse(
                        action_id="barrage",
                        action_type="full",
                        target_id="alpha",
                        attack_type_override="ranged",
                        range_spaces=8,
                        weapon_count=2,
                        uses_superheavy=False,
                        uses_aux_bonus_attack=False,
                    ),
                ],
            ),
        ],
    )
    return MechCombatScenario(combatants=[player, hostile], rounds=[round_one])


def evaluate_example_combat_scenario() -> CombatValidation:
    """Validate the example combat scenario."""
    scenario = build_example_combat_scenario()
    return validate_combat_scenario(scenario)


def build_example_combat_scenario_with_reaction_metadata() -> MechCombatScenario:
    """Build a combat scenario that uses reaction trigger + heat metadata."""
    player = CombatantState(
        id="alpha",
        name="Alpha",
        side="players",
        kind="mech",
        stats=CombatStats(
            size="size_1",
            hp_max=10,
            evasion=10,
            e_defense=8,
            armor=0,
            speed=4,
            sensor_range=10,
            tech_attack=0,
        ),
        resources=CombatResources(
            hp_current=10,
            heat_current=0,
            heat_cap=6,
            structure_current=4,
            stress_current=4,
            repairs_remaining=4,
        ),
        position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        statuses=[],
        conditions=[],
        reaction_triggers=[
            ReactionTriggerEffect(
                reaction_id="overwatch",
                trigger_events=["enemy_enters_threat"],
                uses_per="round",
            )
        ],
    )
    hostile = CombatantState(
        id="bravo",
        name="Bravo",
        side="hostiles",
        kind="mech",
        stats=CombatStats(
            size="size_1",
            hp_max=8,
            evasion=9,
            e_defense=8,
            armor=0,
            speed=3,
            sensor_range=8,
            tech_attack=0,
        ),
        resources=CombatResources(
            hp_current=8,
            heat_current=0,
            heat_cap=6,
            structure_current=4,
            stress_current=4,
            repairs_remaining=4,
        ),
        position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        statuses=[],
        conditions=[],
    )
    round_one = CombatRound(
        round_index=1,
        turns=[
            CombatTurn(
                actor_id="alpha",
                move_used=False,
                actions=[
                    ActionUse(
                        action_id="overcharge", action_type="free", heat_generated=1
                    ),
                    ActionUse(
                        action_id="overwatch",
                        action_type="reaction",
                        target_id="bravo",
                        range_spaces=1,
                        used_as_reaction=True,
                        reaction_trigger="enemy_enters_threat",
                    ),
                ],
            ),
        ],
    )
    return MechCombatScenario(combatants=[player, hostile], rounds=[round_one])


def evaluate_example_combat_scenario_with_reaction_metadata() -> CombatValidation:
    """Validate the reaction/heat metadata combat scenario."""
    scenario = build_example_combat_scenario_with_reaction_metadata()
    return validate_combat_scenario(scenario)


def build_example_combat_scenario_with_terrain() -> MechCombatScenario:
    """Build a combat scenario that exercises terrain LOS/cover validation."""
    scenario = build_example_combat_scenario()
    terrain = TerrainMap(
        tiles=[
            TerrainHex(
                coord=HexCoord(q=1, r=0),
                elevation=0,
                blocks_line_of_sight=True,
                provides_hard_cover=True,
                hard_cover_size="size_1",
            )
        ]
    )
    return MechCombatScenario(
        combatants=scenario.combatants,
        grapples=scenario.grapples,
        rounds=scenario.rounds,
        terrain=terrain,
    )


def evaluate_example_combat_scenario_with_terrain() -> CombatValidation:
    """Validate the terrain-heavy combat scenario."""
    scenario = build_example_combat_scenario_with_terrain()
    return validate_combat_scenario(scenario)


def build_example_combat_scenario_with_area() -> MechCombatScenario:
    """Build a combat scenario that exercises area patterns and tags."""
    scenario = build_example_combat_scenario()
    alpha_position = scenario.combatants[0].position
    if not alpha_position:
        raise ValueError("Scenario missing alpha position")
    charlie = CombatantState(
        id="charlie",
        name="Charlie",
        side="hostiles",
        kind="mech",
        stats=CombatStats(
            size="size_1",
            hp_max=8,
            evasion=9,
            e_defense=8,
            armor=0,
            speed=3,
            sensor_range=8,
            tech_attack=0,
        ),
        resources=CombatResources(
            hp_current=8,
            heat_current=0,
            heat_cap=6,
            structure_current=4,
            stress_current=4,
            repairs_remaining=4,
        ),
        position=HexPosition(coord=HexCoord(q=3, r=0), elevation=0),
        statuses=[],
        conditions=[],
    )
    line_direction = HexCoord(q=1, r=0)
    demo_weapon = MechWeaponDefinition(
        id="demo_blast_arcing",
        name="Demo Blast Arcing",
        size="main",
        weapon_type="launcher",
        damage_type="explosive",
        ranges=[WeaponRange(range_type="range", value=5)],
        tags=[WeaponTag(tag="blast", value=2), WeaponTag(tag="arcing")],
    )
    round_one = CombatRound(
        round_index=1,
        turns=[
            CombatTurn(
                actor_id="alpha",
                move_used=False,
                actions=[
                    build_action_use_from_weapon(
                        action_id="skirmish",
                        action_type="quick",
                        weapon=demo_weapon,
                        target_id="bravo",
                        target_ids=["bravo", "charlie"],
                        weapon_count=1,
                        uses_aux_bonus_attack=False,
                        area_direction=line_direction,
                        area_affected=[
                            HexCoord(q=1, r=0),
                            HexCoord(q=2, r=0),
                        ],
                    ),
                ],
            ),
        ],
    )
    return MechCombatScenario(
        combatants=[*scenario.combatants, charlie],
        grapples=scenario.grapples,
        rounds=[round_one],
        terrain=TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    elevation=0,
                    blocks_line_of_sight=True,
                    provides_hard_cover=True,
                    hard_cover_size="size_1",
                )
            ]
        ),
    )


def evaluate_example_combat_scenario_with_area() -> CombatValidation:
    """Validate the area-pattern combat scenario."""
    scenario = build_example_combat_scenario_with_area()
    return validate_combat_scenario(scenario)


def build_example_combat_scenario_with_line() -> MechCombatScenario:
    """Build a combat scenario that exercises line pattern direction."""
    scenario = build_example_combat_scenario()
    alpha_position = scenario.combatants[0].position
    if not alpha_position:
        raise ValueError("Scenario missing alpha position")
    line_direction = HexCoord(q=1, r=0)
    demo_weapon = MechWeaponDefinition(
        id="demo_line_weapon",
        name="Demo Line Weapon",
        size="main",
        weapon_type="launcher",
        damage_type="explosive",
        ranges=[WeaponRange(range_type="range", value=5)],
        tags=[WeaponTag(tag="line", value=3), WeaponTag(tag="arcing")],
    )
    round_one = CombatRound(
        round_index=1,
        turns=[
            CombatTurn(
                actor_id="alpha",
                move_used=False,
                actions=[
                    build_action_use_from_weapon(
                        action_id="skirmish",
                        action_type="quick",
                        weapon=demo_weapon,
                        target_id="bravo",
                        weapon_count=1,
                        uses_aux_bonus_attack=False,
                        area_direction=line_direction,
                        area_affected=[
                            HexCoord(q=1, r=0),
                            HexCoord(q=2, r=0),
                            HexCoord(q=3, r=0),
                        ],
                    ),
                ],
            ),
        ],
    )
    return MechCombatScenario(
        combatants=scenario.combatants,
        grapples=scenario.grapples,
        rounds=[round_one],
        terrain=scenario.terrain,
    )


def evaluate_example_combat_scenario_with_line() -> CombatValidation:
    """Validate the line-pattern combat scenario."""
    scenario = build_example_combat_scenario_with_line()
    return validate_combat_scenario(scenario)


def evaluate_structure_and_overheat_examples() -> dict[
    str, dict[str, str | int | list[int] | None]
]:
    """Evaluate deterministic structure and overheat resolution examples."""
    from core.mech.combat_resolution import (
        resolve_structure_damage,
        resolve_overheat,
        ResolutionSettings,
    )

    structure_result = resolve_structure_damage(
        remaining_structure=2,
        incoming_damage=8,
        hp_before=5,
        structure_damage_marked=2,
        settings=ResolutionSettings(forced_rolls=[1, 4]),
    )
    overheat_result = resolve_overheat(
        stress_marked=2,
        remaining_stress=2,
        settings=ResolutionSettings(forced_rolls=[1, 4]),
    )
    system_trauma_inventory = MechInventory(
        mounts=[
            WeaponMountState(
                mount_index=0,
                weapons=[WeaponState(weapon_id="demo_weapon", destroyed=True)],
            )
        ],
        systems=[MechSystemState(system_id="demo_system", destroyed=True)],
    )
    system_trauma_result = resolve_structure_damage(
        remaining_structure=2,
        incoming_damage=8,
        hp_before=5,
        structure_damage_marked=1,
        inventory=system_trauma_inventory,
        settings=ResolutionSettings(forced_rolls=[3], forced_system_trauma_roll=2),
    )

    return {
        "structure": {
            "rolls": structure_result.dice.rolls,
            "chosen": structure_result.dice.chosen,
            "outcome": structure_result.outcome.name,
            "direct_hit": structure_result.direct_hit_outcome.name
            if structure_result.direct_hit_outcome
            else None,
            "spillover": structure_result.spillover_damage,
        },
        "system_trauma": {
            "rolls": system_trauma_result.dice.rolls,
            "chosen": system_trauma_result.dice.chosen,
            "outcome": system_trauma_result.outcome.name,
            "direct_hit": (
                system_trauma_result.direct_hit_outcome.name
                if system_trauma_result.direct_hit_outcome
                else None
            ),
            "trauma_target": (
                system_trauma_result.system_trauma.resolved_target
                if system_trauma_result.system_trauma
                else None
            ),
            "fallback_reason": (
                system_trauma_result.system_trauma.fallback_reason
                if system_trauma_result.system_trauma
                else None
            ),
        },
        "overheat": {
            "rolls": overheat_result.dice.rolls,
            "chosen": overheat_result.dice.chosen,
            "outcome": overheat_result.outcome.name,
            "meltdown": overheat_result.meltdown_outcome.name
            if overheat_result.meltdown_outcome
            else None,
        },
    }


def build_example_combat_scenario_with_ai() -> MechCombatScenario:
    """Build a scenario where AI control blocks pilot actions."""
    scenario = build_example_combat_scenario()
    combatants = list(scenario.combatants)
    ai_combatant = combatants[0].model_copy(update={"ai_controlled": True})
    combatants[0] = ai_combatant
    combatants[1] = combatants[1].model_copy(
        update={"position": HexPosition(coord=HexCoord(q=1, r=0), elevation=0)}
    )
    round_one = CombatRound(
        round_index=1,
        turns=[
            CombatTurn(
                actor_id=ai_combatant.id,
                move_used=False,
                actions=[
                    ActionUse(
                        action_id="jockey",
                        action_type="full",
                        target_id="bravo",
                    )
                ],
            )
        ],
    )
    return MechCombatScenario(
        combatants=combatants,
        grapples=scenario.grapples,
        rounds=[round_one],
        terrain=scenario.terrain,
        environment=scenario.environment,
    )


def evaluate_example_combat_scenario_with_ai() -> CombatValidation:
    """Validate AI control restrictions in combat."""
    scenario = build_example_combat_scenario_with_ai()
    return validate_combat_scenario(scenario)


def build_example_combat_scenario_with_flight() -> MechCombatScenario:
    """Build a scenario that exercises flight movement rules."""
    scenario = build_example_combat_scenario()
    alpha = scenario.combatants[0]
    if not alpha.position:
        raise ValueError("Scenario missing alpha position")
    round_one = CombatRound(
        round_index=1,
        turns=[
            CombatTurn(
                actor_id=alpha.id,
                move_used=True,
                movement_mode="flight",
                movement_path=[
                    alpha.position,
                    HexPosition(coord=HexCoord(q=1, r=0), elevation=1),
                    HexPosition(coord=HexCoord(q=1, r=1), elevation=2),
                ],
                actions=[
                    ActionUse(
                        action_id="skirmish", action_type="quick", target_id="bravo"
                    ),
                ],
            ),
        ],
    )
    return MechCombatScenario(
        combatants=scenario.combatants,
        grapples=scenario.grapples,
        rounds=[round_one],
        terrain=scenario.terrain,
        environment=scenario.environment,
    )


def evaluate_example_combat_scenario_with_flight() -> CombatValidation:
    """Validate flight movement rules in combat."""
    scenario = build_example_combat_scenario_with_flight()
    return validate_combat_scenario(scenario)


def build_example_combat_scenario_with_grapple() -> MechCombatScenario:
    """Build a scenario that exercises grapple adjacency rules."""
    scenario = build_example_combat_scenario()
    alpha = scenario.combatants[0]
    bravo = scenario.combatants[1]
    if not alpha.position or not bravo.position:
        raise ValueError("Scenario missing positions")
    if not alpha.position.coord.is_adjacent(bravo.position.coord):
        bravo = bravo.model_copy(
            update={
                "position": HexPosition(
                    coord=HexCoord(
                        q=alpha.position.coord.q + 1, r=alpha.position.coord.r
                    ),
                    elevation=alpha.position.elevation,
                )
            }
        )
    grapple_turn = CombatTurn(
        actor_id=alpha.id,
        move_used=False,
        actions=[
            ActionUse(
                action_id="grapple",
                action_type="quick",
                target_id=bravo.id,
            ),
        ],
    )
    round_one = CombatRound(round_index=1, turns=[grapple_turn])
    return MechCombatScenario(
        combatants=[alpha, bravo],
        grapples=scenario.grapples,
        rounds=[round_one],
        terrain=scenario.terrain,
        environment=scenario.environment,
    )


def evaluate_example_combat_scenario_with_grapple() -> CombatValidation:
    """Validate grapple adjacency rules in combat."""
    scenario = build_example_combat_scenario_with_grapple()
    return validate_combat_scenario(scenario)


def build_example_combat_scenario_with_stabilize() -> MechCombatScenario:
    """Build a scenario that exercises stabilize option validation."""
    scenario = build_example_combat_scenario()
    alpha = scenario.combatants[0]
    stabilize_turn = CombatTurn(
        actor_id=alpha.id,
        move_used=False,
        actions=[
            ActionUse(
                action_id="stabilize",
                action_type="full",
                stabilize_primary="cool_heat",
                stabilize_secondary="reload_loading",
            ),
        ],
    )
    round_one = CombatRound(round_index=1, turns=[stabilize_turn])
    return MechCombatScenario(
        combatants=scenario.combatants,
        grapples=scenario.grapples,
        rounds=[round_one],
        terrain=scenario.terrain,
        environment=scenario.environment,
    )


def evaluate_example_combat_scenario_with_stabilize() -> CombatValidation:
    """Validate stabilize rules in combat."""
    scenario = build_example_combat_scenario_with_stabilize()
    return validate_combat_scenario(scenario)


def build_example_combat_scenario_with_seeking() -> MechCombatScenario:
    """Build a scenario that exercises seeking tag LOS/cover ignore rules."""
    scenario = build_example_combat_scenario()
    alpha_position = scenario.combatants[0].position
    if not alpha_position:
        raise ValueError("Scenario missing alpha position")
    seeking_weapon = MechWeaponDefinition(
        id="demo_seeking_weapon",
        name="Demo Seeking Weapon",
        size="main",
        weapon_type="launcher",
        damage_type="explosive",
        ranges=[WeaponRange(range_type="range", value=6)],
        tags=[WeaponTag(tag="seeking")],
    )
    round_one = CombatRound(
        round_index=1,
        turns=[
            CombatTurn(
                actor_id="alpha",
                move_used=False,
                actions=[
                    build_action_use_from_weapon(
                        action_id="skirmish",
                        action_type="quick",
                        weapon=seeking_weapon,
                        target_id="bravo",
                        weapon_count=1,
                    ),
                ],
            ),
        ],
    )
    return MechCombatScenario(
        combatants=scenario.combatants,
        grapples=scenario.grapples,
        rounds=[round_one],
        terrain=TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    elevation=0,
                    blocks_line_of_sight=True,
                    provides_hard_cover=True,
                )
            ]
        ),
        environment=scenario.environment,
    )


def evaluate_example_combat_scenario_with_seeking() -> CombatValidation:
    """Validate seeking tag in combat."""
    scenario = build_example_combat_scenario_with_seeking()
    return validate_combat_scenario(scenario)


def build_example_combat_scenario_with_search() -> MechCombatScenario:
    """Build a scenario that exercises search contested check validation."""
    scenario = build_example_combat_scenario()
    alpha = scenario.combatants[0]
    bravo = scenario.combatants[1].model_copy(update={"conditions": ["hidden"]})
    contested = ContestedCheck(
        attacker=SkillCheck(modifiers=RollModifiers()),
        defender=SkillCheck(modifiers=RollModifiers()),
    )
    round_one = CombatRound(
        round_index=1,
        turns=[
            CombatTurn(
                actor_id=alpha.id,
                move_used=False,
                actions=[
                    ActionUse(
                        action_id="search",
                        action_type="quick",
                        target_id=bravo.id,
                        contested_check=contested,
                    )
                ],
            )
        ],
    )
    return MechCombatScenario(
        combatants=[alpha, bravo],
        grapples=scenario.grapples,
        rounds=[round_one],
        terrain=scenario.terrain,
        environment=scenario.environment,
    )


def evaluate_example_combat_scenario_with_search() -> CombatValidation:
    """Validate search contested checks in combat."""
    scenario = build_example_combat_scenario_with_search()
    return validate_combat_scenario(scenario)


def build_example_combat_scenario_with_lock_on_consumption() -> MechCombatScenario:
    """Build a scenario that exercises lock on consumption validation."""
    scenario = build_example_combat_scenario()
    alpha = scenario.combatants[0].model_copy(update={"conditions": ["lock_on"]})
    bravo = scenario.combatants[1]
    round_one = CombatRound(
        round_index=1,
        turns=[
            CombatTurn(
                actor_id=bravo.id,
                move_used=False,
                actions=[
                    ActionUse(
                        action_id="skirmish",
                        action_type="quick",
                        target_id=alpha.id,
                        attack_type_override="ranged",
                        range_spaces=10,
                        weapon_count=1,
                        uses_superheavy=False,
                        uses_aux_bonus_attack=False,
                        consumes_lock_on=True,
                    )
                ],
            )
        ],
    )
    return MechCombatScenario(
        combatants=[alpha, bravo],
        grapples=scenario.grapples,
        rounds=[round_one],
        terrain=scenario.terrain,
        environment=scenario.environment,
    )


def evaluate_example_combat_scenario_with_lock_on_consumption() -> CombatValidation:
    """Validate lock on consumption in combat."""
    scenario = build_example_combat_scenario_with_lock_on_consumption()
    return validate_combat_scenario(scenario)
