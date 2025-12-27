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
    Talent,
    TalentDefinition,
    TalentRank,
    License,
    LicenseDefinition,
    CoreBonus,
    CoreBonusDefinition,
)
from core.shared.dice import DiceExpression
from core.shared.effects import (
    MechanicalEffect,
    StatModifier,
    DamageModifier,
    RangeModifier,
    ActionGrant,
    Immunity,
    Resistance,
    AccuracyModifier,
    MovementGrant,
    StatusGrant,
)


# All exportable models
EXPORTABLE_MODELS: dict[str, type[BaseModel]] = {
    # Pilot domain
    "Pilot": Pilot,
    "Skill": Skill,
    "SkillSet": SkillSet,
    "Background": Background,
    "Talent": Talent,
    "TalentDefinition": TalentDefinition,
    "TalentRank": TalentRank,
    "License": License,
    "LicenseDefinition": LicenseDefinition,
    "CoreBonus": CoreBonus,
    "CoreBonusDefinition": CoreBonusDefinition,
    # Shared - Dice
    "DiceExpression": DiceExpression,
    # Shared - Effects
    "MechanicalEffect": MechanicalEffect,
    "StatModifier": StatModifier,
    "DamageModifier": DamageModifier,
    "RangeModifier": RangeModifier,
    "ActionGrant": ActionGrant,
    "Immunity": Immunity,
    "Resistance": Resistance,
    "AccuracyModifier": AccuracyModifier,
    "MovementGrant": MovementGrant,
    "StatusGrant": StatusGrant,
}


def export_schema(model: type[BaseModel], mode: str = "serialization") -> dict[str, Any]:
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

