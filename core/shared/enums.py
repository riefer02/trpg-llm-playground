"""Shared enumerations and literal types for Lancer TTRPG."""

from typing import Literal

__all__ = [
    "ActionType",
    "AttackType",
    "DamageType",
    "SaveType",
    "RangeType",
    "SizeClass",
    "ManufacturerType",
    "MountType",
    "MountSlotType",
    "SystemType",
    "StatusType",
    "CoverType",
]

# Action Types (what kind of action is this?)
ActionType = Literal[
    "free",  # No action cost
    "quick",  # Quick action (2 per turn)
    "full",  # Full action (1 per turn, ends quick actions)
    "reaction",  # Triggered reaction
    "protocol",  # Start-of-turn protocol
    "move",  # Regular move action
]

# Attack Types
AttackType = Literal[
    "melee",
    "ranged",
    "tech",
]

# Damage Types
DamageType = Literal[
    "kinetic",  # Physical/ballistic damage
    "explosive",  # Area/blast damage
    "energy",  # Laser/beam damage
    "burn",  # Heat-based, causes heat buildup
]

# Save Types (for mech saves)
SaveType = Literal[
    "hull",
    "agility",
    "systems",
    "engineering",
]

# Range Types
RangeType = Literal[
    "range",  # Standard ranged (in spaces)
    "threat",  # Melee threat range
    "thrown",  # Thrown range
    "line",  # Line attack
    "cone",  # Cone attack
    "burst",  # Burst area
    "blast",  # Blast area
    "sensors",  # Sensor range
]

# Size Classes (for mechs)
SizeClass = Literal[
    "size_half",  # 1/2 size (small drones)
    "size_1",  # Size 1 (standard)
    "size_2",  # Size 2 (larger)
    "size_3",  # Size 3 (very large)
    "size_4",  # Size 4 (titanic)
    "size_5",  # Size 5 (massive)
]

# Mech Manufacturers (for licenses and core bonuses)
ManufacturerType = Literal[
    "GMS",  # General Massive Systems (default/universal)
    "IPS-N",  # Interplanetary Shipping-Northstar
    "SSC",  # Smith-Shimano Corpro
    "HORUS",  # HORUS (mysterious collective)
    "HA",  # Harrison Armory
]

# Mount Types (for weapons)
MountType = Literal[
    "aux",  # Auxiliary mount
    "main",  # Main mount
    "heavy",  # Heavy mount
    "superheavy",  # Superheavy mount (takes 2 mounts)
]

# Mount Slot Types (for frame mounts)
MountSlotType = Literal[
    "main",
    "heavy",
    "aux_aux",
    "main_aux",
    "flexible",
    "integrated",
]

# System Types
SystemType = Literal[
    "system",  # Standard system
    "tech",  # Tech action system
    "deployable",  # Creates deployable
    "drone",  # Drone system
    "shield",  # Shield system
    "ai",  # AI system
]

# Status/Condition Types
StatusType = Literal[
    "braced",
    "immobilized",
    "impaired",
    "jammed",
    "lock_on",
    "shredded",
    "slowed",
    "stunned",
    "prone",
    "hidden",
    "invisible",
    "shutdown",
    "exposed",
    "engaged",
    "burn",
    "unshackled",  # NHP has become unshackled per PR2 5081-5082
]

# Cover Types
CoverType = Literal[
    "none",
    "soft",
    "hard",
]
