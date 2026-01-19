"""
JSON Schema export utility for Lancer type definitions.

Exports Pydantic models to JSON Schema format for:
- Database schema generation
- API documentation
- Cross-language type sharing
- Validation in other systems
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from core.pilot import (
    Pilot,
    Skill,
    SkillSet,
    Background,
    BackgroundInvokeRule,
    Talent,
    TalentDefinition,
    TalentRank,
    License,
    LicenseDefinition,
    CoreBonus,
    CoreBonusDefinition,
    PilotTrigger,
    TriggerDefinition,
    LevelProgression,
    PilotProgressionRules,
    ProgressionIssue,
    ProgressionValidation,
    MissionCadenceRules,
    ReserveEntry,
    ReserveDefinition,
    DowntimeActionDefinition,
    DowntimeActionUse,
    DowntimePlan,
    DowntimeIssue,
    DowntimeValidation,
    PilotGearItemDefinition,
    PilotGearRules,
    PilotLoadout,
    PilotCombatBaseStats,
    PilotDamageSeverity,
    DownAndOutRule,
    PilotRestRule,
    PilotCombatRules,
)
from core.shared.dice import DiceExpression
from core.shared.damage import DamageBreakdown
from core.shared.effects import (
    MechanicalEffect,
    StatModifier,
    StatOverrideEffect,
    SpatialCondition,
    AttackContextCondition,
    CheckContextCondition,
    ReactionCondition,
    SizeCondition,
    ConditionGroup,
    DamageModifier,
    DirectDamage,
    RangeModifier,
    LimitedUseBonusEffect,
    LimitedUseRechargeEffect,
    ActionGrant,
    TargetMarkEffect,
    LeadershipDicePoolEffect,
    DicePoolGain,
    DicePoolSpendOption,
    DicePoolEffect,
    CountdownDieTrigger,
    CountdownDieEffect,
    TechRange,
    TechAction,
    TechAttackModifier,
    TechActionRestriction,
    ActionRestriction,
    LineOfSightRestriction,
    EffectChoice,
    Immunity,
    TagImmunityEffect,
    Resistance,
    DamageReduction,
    DamageReductionRollEffect,
    AccuracyModifier,
    MovementGrant,
    MoveAdjacentEffect,
    ForcedMovement,
    StatusToggleEffect,
    StatusGrant,
    StatusClear,
    StatusBreakCondition,
    StatusStackLimit,
    MovementScopedStatus,
    StatusRestriction,
    CoverRestriction,
    CoverGrant,
    IntelEffect,
    MovementRestrictionEffect,
    MovementSurfaceEffect,
    MovementModeAccessEffect,
    JumpDistanceEffect,
    MovementOverrideEffect,
    ResourceChange,
    ScaledResourceChange,
    OverchargeCostCapEffect,
    AttackTargetingEffect,
    AreaSelectionEffect,
    AreaAttackPattern,
    AttackRerollEffect,
    AttackOutcomeEffect,
    CriticalDamageOverrideEffect,
    AccuracyTradeEffect,
    DelayedImpactEffect,
    WeaponTagGrant,
    WeaponRangeSpec,
    WeaponSizeBonus,
    WeaponModEffect,
    WeaponGrantEffect,
    DeploymentEffect,
    PhaseShiftEffect,
    ZoneEffect,
    ZoneEndCondition,
    AttackCaptureEffect,
    ReloadEffect,
    DamageAbsorption,
    OutOfPlayEffect,
    ZeroHpSurvivalEffect,
    TetherEffect,
    AISystemLimitEffect,
    AIControlTransferEffect,
    EffectRemoval,
    MovementTrailEffect,
    HologramTrailEffect,
    TriggeredEffect,
    SaveCheck,
    RandomCheckEffect,
    RollPatternEffect,
    StatusTrigger,
    ModeEffect,
)
from core.shared.rolls import (
    AccuracyDifficulty,
    FlatBonus,
    RollModifiers,
    SkillCheck,
    AttackRoll,
    SaveRoll,
    ContestedCheck,
)
from core.shared.narrative import (
    NarrativeCheckTierRule,
    NarrativeHelpRule,
    NarrativePushRule,
    NarrativeCheckRules,
    NarrativeGoalOutcome,
    NarrativeGoalCondition,
    NarrativeGoal,
    NarrativeGoalAttempt,
    NarrativeGoalState,
    NarrativeGoalTracker,
    NarrativeResolutionRequirement,
    NarrativeComplication,
    NarrativeComplicationState,
    NarrativeCombatState,
)
from core.mech import (
    DynamicWeaponDefinition,
    MechWeaponDefinition,
    MechSystemDefinition,
    MimicGunProfileRule,
    MountlessWeaponDefinition,
    MountSlot,
    CoreSystemDefinition,
    MechFrameBaseStats,
    MechFrameDefinition,
    MechPilotingRules,
    CorePowerRules,
    SystemPointRules,
    WeaponProfile,
    WeaponProfileChoice,
    MountedWeapon,
    InstalledSystem,
    MechBuild,
    MechDerivedStats,
    MechBuildIssue,
    MechBuildValidation,
)
from core.mech.grid import HexCoord, HexPosition
from core.mech.terrain import TerrainHex, TerrainMap
from core.mech.statuses import StatusDefinition, StatusInstance
from core.mech.combat_rules import (
    TurnOrderRules,
    TurnActionRules,
    EngagementRules,
    ObstructionRules,
    TerrainRules,
    FallingRules,
    ZeroGRules,
    TeleportRules,
    FlightRules,
    AttackPatternDefinition,
    LineOfSightRules,
    ValidTargetRules,
    DamageResolutionRules,
    StructureDamageRules,
    OverheatRules,
    ReactorMeltdownRules,
    RestRepairRules,
    MechCombatRules,
)
from core.mech.combat_actions import (
    ActionRule,
    AttackActionProfile,
    MovementActionProfile,
    TechActionProfile,
    GrappleRule,
    StabilizeRule,
    PrepareRule,
    ShutdownRule,
    BootUpRule,
    MountRule,
    SelfDestructRule,
    OverchargeActionRule,
    FightRule,
    JockeyRule,
    JockeyOption,
)
from core.mech.combat_state import (
    CombatStats,
    CombatResources,
    CombatantState,
    GrappleLink,
    ActionUse,
    CombatTurn,
    CombatRound,
    MechCombatScenario,
)
from core.mech.validation.combat_validation import (
    CombatValidationIssue,
    CombatValidation,
)
from core.mech.combat_resolution import (
    DiceRollResult,
    StructureResolution,
    OverheatResolution,
    ResolutionSettings,
)
from core.mech.combat_execution import (
    ActionExecutionInput,
    ActionExecutionResult,
    TurnStartResult,
    TurnEndResult,
    ReactionInput,
    ReactionResult,
    AvailableAction,
    AvailableActionsResult,
    ResourceChange,
)
from core.mech.combat_models import (
    OverwatchOpportunityInfo,
)
from core.shared.overwatch import (
    OverwatchOpportunity,
    OverwatchTriggerResult,
)
from core.mech.action_economy import (
    ActionEconomyState,
    ActionEconomyResult,
)
from core.character import (
    Character,
    MechConfiguration,
)
from core.shared.campaign import (
    Campaign,
    Session,
    MissionPrepPlan,
    CampaignIdentity,
    CampaignLobbyState,
    MissionObjectiveBrief,
    MissionStakesBrief,
    ReservePlanEntry,
    SessionLifecycleCheckpoint,
    MissionOutcomeReport,
)
from core.shared.terrain_primitives import (
    MaterialProperties,
    TerrainPrimitive,
    FloorTile,
    Obstacle,
    SoftCoverZone,
    Hazard,
    Objective,
    DestructibleTerrainState,
    GeneratedTerrain,
)
from core.shared.terrain_generation import (
    TileSetConfig,
    TerrainGeneratorParams,
)
from core.shared.hide_search import (
    SoftCoverZoneState,
)
from core.shared.scenario import (
    SitrepTemplate,
    SitrepZone,
    VictoryCondition,
)
from core.shared.sitrep_resolution import (
    SitrepResolution,
    SitrepDeployment,
    SitrepVictoryCondition,
    ZoneControlStateTracker,
)
from core.gm_toolkit.encounter_builder import (
    PlayerPartyPower,
    EnemyForceRecommendation,
)


# All exportable models
EXPORTABLE_MODELS: dict[str, type[BaseModel]] = {
    # Pilot domain
    "Pilot": Pilot,
    "Skill": Skill,
    "SkillSet": SkillSet,
    "Background": Background,
    "BackgroundInvokeRule": BackgroundInvokeRule,
    "Talent": Talent,
    "TalentDefinition": TalentDefinition,
    "TalentRank": TalentRank,
    "License": License,
    "LicenseDefinition": LicenseDefinition,
    "CoreBonus": CoreBonus,
    "CoreBonusDefinition": CoreBonusDefinition,
    "PilotTrigger": PilotTrigger,
    "TriggerDefinition": TriggerDefinition,
    "LevelProgression": LevelProgression,
    "PilotProgressionRules": PilotProgressionRules,
    "ProgressionIssue": ProgressionIssue,
    "ProgressionValidation": ProgressionValidation,
    "MissionCadenceRules": MissionCadenceRules,
    "ReserveEntry": ReserveEntry,
    "ReserveDefinition": ReserveDefinition,
    "DowntimeActionDefinition": DowntimeActionDefinition,
    "DowntimeActionUse": DowntimeActionUse,
    "DowntimePlan": DowntimePlan,
    "DowntimeIssue": DowntimeIssue,
    "DowntimeValidation": DowntimeValidation,
    "PilotGearItemDefinition": PilotGearItemDefinition,
    "PilotGearRules": PilotGearRules,
    "PilotLoadout": PilotLoadout,
    "PilotCombatBaseStats": PilotCombatBaseStats,
    "PilotDamageSeverity": PilotDamageSeverity,
    "DownAndOutRule": DownAndOutRule,
    "PilotRestRule": PilotRestRule,
    "PilotCombatRules": PilotCombatRules,
    # Character domain (unified pilot + mech)
    "Character": Character,
    "MechConfiguration": MechConfiguration,
    # Campaign persistence
    "Campaign": Campaign,
    "Session": Session,
    "MissionPrepPlan": MissionPrepPlan,
    "CampaignIdentity": CampaignIdentity,
    "CampaignLobbyState": CampaignLobbyState,
    "MissionObjectiveBrief": MissionObjectiveBrief,
    "MissionStakesBrief": MissionStakesBrief,
    "ReservePlanEntry": ReservePlanEntry,
    "SessionLifecycleCheckpoint": SessionLifecycleCheckpoint,
    "MissionOutcomeReport": MissionOutcomeReport,
    # Shared - Dice
    "DiceExpression": DiceExpression,
    # Shared - Effects
    "MechanicalEffect": MechanicalEffect,
    "StatModifier": StatModifier,
    "StatOverrideEffect": StatOverrideEffect,
    "DamageModifier": DamageModifier,
    "DirectDamage": DirectDamage,
    "RangeModifier": RangeModifier,
    "LimitedUseBonusEffect": LimitedUseBonusEffect,
    "LimitedUseRechargeEffect": LimitedUseRechargeEffect,
    "ActionGrant": ActionGrant,
    "TargetMarkEffect": TargetMarkEffect,
    "LeadershipDicePoolEffect": LeadershipDicePoolEffect,
    "DicePoolGain": DicePoolGain,
    "DicePoolSpendOption": DicePoolSpendOption,
    "DicePoolEffect": DicePoolEffect,
    "CountdownDieTrigger": CountdownDieTrigger,
    "CountdownDieEffect": CountdownDieEffect,
    "TechRange": TechRange,
    "TechAction": TechAction,
    "TechAttackModifier": TechAttackModifier,
    "TechActionRestriction": TechActionRestriction,
    "ActionRestriction": ActionRestriction,
    "LineOfSightRestriction": LineOfSightRestriction,
    "EffectChoice": EffectChoice,
    "Immunity": Immunity,
    "TagImmunityEffect": TagImmunityEffect,
    "Resistance": Resistance,
    "DamageReduction": DamageReduction,
    "DamageReductionRollEffect": DamageReductionRollEffect,
    "AccuracyModifier": AccuracyModifier,
    "MovementGrant": MovementGrant,
    "MoveAdjacentEffect": MoveAdjacentEffect,
    "ForcedMovement": ForcedMovement,
    "StatusToggleEffect": StatusToggleEffect,
    "StatusGrant": StatusGrant,
    "StatusClear": StatusClear,
    "StatusBreakCondition": StatusBreakCondition,
    "StatusStackLimit": StatusStackLimit,
    "MovementScopedStatus": MovementScopedStatus,
    "StatusRestriction": StatusRestriction,
    "CoverRestriction": CoverRestriction,
    "CoverGrant": CoverGrant,
    "IntelEffect": IntelEffect,
    "MovementRestrictionEffect": MovementRestrictionEffect,
    "MovementSurfaceEffect": MovementSurfaceEffect,
    "MovementModeAccessEffect": MovementModeAccessEffect,
    "JumpDistanceEffect": JumpDistanceEffect,
    "MovementOverrideEffect": MovementOverrideEffect,
    "ResourceChange": ResourceChange,
    "ScaledResourceChange": ScaledResourceChange,
    "OverchargeCostCapEffect": OverchargeCostCapEffect,
    "AttackTargetingEffect": AttackTargetingEffect,
    "AreaSelectionEffect": AreaSelectionEffect,
    "AreaAttackPattern": AreaAttackPattern,
    "AttackRerollEffect": AttackRerollEffect,
    "AttackOutcomeEffect": AttackOutcomeEffect,
    "CriticalDamageOverrideEffect": CriticalDamageOverrideEffect,
    "AccuracyTradeEffect": AccuracyTradeEffect,
    "DelayedImpactEffect": DelayedImpactEffect,
    "WeaponTagGrant": WeaponTagGrant,
    "WeaponRangeSpec": WeaponRangeSpec,
    "WeaponSizeBonus": WeaponSizeBonus,
    "WeaponModEffect": WeaponModEffect,
    "WeaponGrantEffect": WeaponGrantEffect,
    "DeploymentEffect": DeploymentEffect,
    "PhaseShiftEffect": PhaseShiftEffect,
    "ZoneEffect": ZoneEffect,
    "ZoneEndCondition": ZoneEndCondition,
    "AttackCaptureEffect": AttackCaptureEffect,
    "ReloadEffect": ReloadEffect,
    "DamageAbsorption": DamageAbsorption,
    "OutOfPlayEffect": OutOfPlayEffect,
    "ZeroHpSurvivalEffect": ZeroHpSurvivalEffect,
    "TetherEffect": TetherEffect,
    "AISystemLimitEffect": AISystemLimitEffect,
    "AIControlTransferEffect": AIControlTransferEffect,
    "EffectRemoval": EffectRemoval,
    "MovementTrailEffect": MovementTrailEffect,
    "HologramTrailEffect": HologramTrailEffect,
    "TriggeredEffect": TriggeredEffect,
    "SaveCheck": SaveCheck,
    "RandomCheckEffect": RandomCheckEffect,
    "RollPatternEffect": RollPatternEffect,
    "StatusTrigger": StatusTrigger,
    "ModeEffect": ModeEffect,
    # Shared - Rolls
    "AccuracyDifficulty": AccuracyDifficulty,
    "FlatBonus": FlatBonus,
    "RollModifiers": RollModifiers,
    "SkillCheck": SkillCheck,
    "AttackRoll": AttackRoll,
    "SaveRoll": SaveRoll,
    "ContestedCheck": ContestedCheck,
    # Shared - Narrative
    "NarrativeCheckTierRule": NarrativeCheckTierRule,
    "NarrativeHelpRule": NarrativeHelpRule,
    "NarrativePushRule": NarrativePushRule,
    "NarrativeCheckRules": NarrativeCheckRules,
    "NarrativeGoalOutcome": NarrativeGoalOutcome,
    "NarrativeGoalCondition": NarrativeGoalCondition,
    "NarrativeGoal": NarrativeGoal,
    "NarrativeGoalAttempt": NarrativeGoalAttempt,
    "NarrativeGoalState": NarrativeGoalState,
    "NarrativeGoalTracker": NarrativeGoalTracker,
    "NarrativeResolutionRequirement": NarrativeResolutionRequirement,
    "NarrativeComplication": NarrativeComplication,
    "NarrativeComplicationState": NarrativeComplicationState,
    "NarrativeCombatState": NarrativeCombatState,
    # Mech domain
    "WeaponProfile": WeaponProfile,
    "WeaponProfileChoice": WeaponProfileChoice,
    "MimicGunProfileRule": MimicGunProfileRule,
    "DynamicWeaponDefinition": DynamicWeaponDefinition,
    "MountlessWeaponDefinition": MountlessWeaponDefinition,
    "MechWeaponDefinition": MechWeaponDefinition,
    "MechSystemDefinition": MechSystemDefinition,
    "MountSlot": MountSlot,
    "CoreSystemDefinition": CoreSystemDefinition,
    "MechFrameBaseStats": MechFrameBaseStats,
    "MechFrameDefinition": MechFrameDefinition,
    "MechPilotingRules": MechPilotingRules,
    "CorePowerRules": CorePowerRules,
    "SystemPointRules": SystemPointRules,
    "MountedWeapon": MountedWeapon,
    "InstalledSystem": InstalledSystem,
    "MechBuild": MechBuild,
    "MechDerivedStats": MechDerivedStats,
    "MechBuildIssue": MechBuildIssue,
    "MechBuildValidation": MechBuildValidation,
    # Mech combat (grid/status/rules/actions/state)
    "HexCoord": HexCoord,
    "HexPosition": HexPosition,
    "TerrainHex": TerrainHex,
    "TerrainMap": TerrainMap,
    "StatusDefinition": StatusDefinition,
    "StatusInstance": StatusInstance,
    "TurnOrderRules": TurnOrderRules,
    "TurnActionRules": TurnActionRules,
    "EngagementRules": EngagementRules,
    "ObstructionRules": ObstructionRules,
    "TerrainRules": TerrainRules,
    "FallingRules": FallingRules,
    "ZeroGRules": ZeroGRules,
    "TeleportRules": TeleportRules,
    "FlightRules": FlightRules,
    "AttackPatternDefinition": AttackPatternDefinition,
    "LineOfSightRules": LineOfSightRules,
    "ValidTargetRules": ValidTargetRules,
    "DamageResolutionRules": DamageResolutionRules,
    "StructureDamageRules": StructureDamageRules,
    "OverheatRules": OverheatRules,
    "ReactorMeltdownRules": ReactorMeltdownRules,
    "RestRepairRules": RestRepairRules,
    "MechCombatRules": MechCombatRules,
    "ActionRule": ActionRule,
    "AttackActionProfile": AttackActionProfile,
    "MovementActionProfile": MovementActionProfile,
    "TechActionProfile": TechActionProfile,
    "GrappleRule": GrappleRule,
    "StabilizeRule": StabilizeRule,
    "PrepareRule": PrepareRule,
    "ShutdownRule": ShutdownRule,
    "BootUpRule": BootUpRule,
    "MountRule": MountRule,
    "SelfDestructRule": SelfDestructRule,
    "OverchargeActionRule": OverchargeActionRule,
    "FightRule": FightRule,
    "JockeyRule": JockeyRule,
    "JockeyOption": JockeyOption,
    "CombatStats": CombatStats,
    "CombatResources": CombatResources,
    "CombatantState": CombatantState,
    "GrappleLink": GrappleLink,
    "ActionUse": ActionUse,
    "CombatTurn": CombatTurn,
    "CombatRound": CombatRound,
    "MechCombatScenario": MechCombatScenario,
    "CombatValidationIssue": CombatValidationIssue,
    "CombatValidation": CombatValidation,
    "DiceRollResult": DiceRollResult,
    "StructureResolution": StructureResolution,
    "OverheatResolution": OverheatResolution,
    "ResolutionSettings": ResolutionSettings,
    # Combat execution
    "ActionExecutionInput": ActionExecutionInput,
    "ActionExecutionResult": ActionExecutionResult,
    "DamageBreakdown": DamageBreakdown,
    "TurnStartResult": TurnStartResult,
    "TurnEndResult": TurnEndResult,
    "ReactionInput": ReactionInput,
    "ReactionResult": ReactionResult,
    "AvailableAction": AvailableAction,
    "AvailableActionsResult": AvailableActionsResult,
    "ResourceChange": ResourceChange,
    "OverwatchOpportunityInfo": OverwatchOpportunityInfo,
    # Overwatch trigger detection
    "OverwatchOpportunity": OverwatchOpportunity,
    "OverwatchTriggerResult": OverwatchTriggerResult,
    # Action economy
    "ActionEconomyState": ActionEconomyState,
    "ActionEconomyResult": ActionEconomyResult,
    # Terrain primitives and generation
    "MaterialProperties": MaterialProperties,
    "TerrainPrimitive": TerrainPrimitive,
    "FloorTile": FloorTile,
    "Obstacle": Obstacle,
    "SoftCoverZone": SoftCoverZone,
    "Hazard": Hazard,
    "Objective": Objective,
    "DestructibleTerrainState": DestructibleTerrainState,
    "GeneratedTerrain": GeneratedTerrain,
    "TileSetConfig": TileSetConfig,
    "TerrainGeneratorParams": TerrainGeneratorParams,
    "SoftCoverZoneState": SoftCoverZoneState,
    # SITREP and Mission Pipeline
    "SitrepTemplate": SitrepTemplate,
    "SitrepZone": SitrepZone,
    "VictoryCondition": VictoryCondition,
    "SitrepResolution": SitrepResolution,
    "SitrepDeployment": SitrepDeployment,
    "SitrepVictoryCondition": SitrepVictoryCondition,
    "ZoneControlStateTracker": ZoneControlStateTracker,
    "PlayerPartyPower": PlayerPartyPower,
    "EnemyForceRecommendation": EnemyForceRecommendation,
}


def export_schema(
    model: type[BaseModel], mode: str = "serialization"
) -> dict[str, Any]:
    """
    Export a single model to JSON Schema.

    Args:
        model: The Pydantic model class to export
        mode: "serialization" or "validation" schema mode

    Returns:
        JSON Schema as a dictionary
    """
    return model.model_json_schema(mode=mode)


def export_all_schemas(
    output_dir: Path | str = "schemas",
    mode: str = "serialization",
) -> dict[str, Path]:
    """
    Export all models to individual JSON Schema files.

    Args:
        output_dir: Directory to write schema files
        mode: "serialization" or "validation" schema mode

    Returns:
        Dictionary mapping model names to their schema file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: dict[str, Path] = {}

    for name, model in EXPORTABLE_MODELS.items():
        schema = export_schema(model, mode)
        file_path = output_dir / f"{name.lower()}.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)

        exported[name] = file_path
        print(f"Exported: {name} -> {file_path}")

    return exported


def export_combined_schema(
    output_path: Path | str = "schemas/lancer.json",
    mode: str = "serialization",
) -> Path:
    """
    Export all models to a single combined JSON Schema file.

    The combined schema uses $defs for shared definitions
    and allows referencing any model type.

    Args:
        output_path: Path to write the combined schema
        mode: "serialization" or "validation" schema mode

    Returns:
        Path to the created schema file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build combined schema with $defs
    combined_schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://lancer.schema/combined",
        "title": "Lancer TTRPG Schema",
        "description": "Combined JSON Schema for Lancer TTRPG type definitions",
        "$defs": {},
        "oneOf": [],
    }

    for name, model in EXPORTABLE_MODELS.items():
        schema = export_schema(model, mode)

        # Extract $defs from individual schemas and merge
        if "$defs" in schema:
            for def_name, def_schema in schema["$defs"].items():
                combined_schema["$defs"][def_name] = def_schema
            del schema["$defs"]

        # Add the main schema to $defs
        combined_schema["$defs"][name] = schema

        # Add reference to oneOf
        combined_schema["oneOf"].append({"$ref": f"#/$defs/{name}"})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined_schema, f, indent=2)

    print(f"Exported combined schema: {output_path}")
    return output_path


def print_schema(model: type[BaseModel], mode: str = "serialization") -> None:
    """Print a model's JSON Schema to stdout."""
    schema = export_schema(model, mode)
    print(json.dumps(schema, indent=2))


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export Lancer schemas to JSON Schema")
    parser.add_argument(
        "--output-dir",
        default="schemas",
        help="Directory for individual schema files",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also export a combined schema file",
    )
    parser.add_argument(
        "--mode",
        choices=["serialization", "validation"],
        default="serialization",
        help="JSON Schema mode",
    )
    parser.add_argument(
        "--model",
        choices=list(EXPORTABLE_MODELS.keys()),
        help="Export only a specific model",
    )
    args = parser.parse_args()

    if args.model:
        # Export single model
        model = EXPORTABLE_MODELS[args.model]
        print_schema(model, args.mode)
    else:
        # Export all models
        export_all_schemas(args.output_dir, args.mode)

        if args.combined:
            export_combined_schema(f"{args.output_dir}/lancer.json", args.mode)

        print(f"\nAll schemas exported to: {args.output_dir}/")
