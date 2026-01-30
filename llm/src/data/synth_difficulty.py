"""
Difficulty stratification for synthetic data generation.

Ensures diverse training coverage by explicitly generating questions
at different cognitive complexity levels:
- Basic: Direct factual recall
- Intermediate: Synthesis and comparison
- Advanced: Edge cases, multi-step reasoning, interactions
"""

import hashlib
from typing import Dict, Optional, Tuple


# Difficulty level definitions with prompt instructions
DIFFICULTY_LEVELS = {
    "basic": {
        "description": "Direct factual recall",
        "instructions": (
            "Generate simple, direct questions that test basic factual recall.\n"
            "Examples:\n"
            "- 'What is [term]?'\n"
            "- 'How many [X] does [Y] have?'\n"
            "- 'What happens when you [action]?'\n"
            "- 'List the [things] for [topic].'\n\n"
            "Answers should be concise and paraphrase the mechanical information."
        ),
        "answer_guidance": (
            "Keep answers brief and factual. Paraphrase the rules in your own words. "
            "Avoid over-explaining - just answer what was asked."
        ),
    },
    "intermediate": {
        "description": "Synthesis and comparison",
        "instructions": (
            "Generate questions requiring synthesis of 2-3 facts or comparison.\n"
            "Examples:\n"
            "- 'How does [X] interact with [Y]?'\n"
            "- 'Compare [A] and [B] for [purpose].'\n"
            "- 'What are the advantages of [choice] over [alternative]?'\n"
            "- 'When would you use [X] instead of [Y]?'\n\n"
            "Answers should connect multiple pieces of information from the context."
        ),
        "answer_guidance": (
            "Connect multiple facts from the context. Show relationships between concepts. "
            "Include relevant tradeoffs or considerations when comparing options."
        ),
    },
    "advanced": {
        "description": "Edge cases and complex reasoning",
        "instructions": (
            "Generate questions about edge cases, exceptions, or multi-step interactions.\n"
            "Examples:\n"
            "- 'What happens if [X] and [Y] both apply at the same time?'\n"
            "- 'Under what conditions would [rule] not apply?'\n"
            "- 'How would you resolve [complex scenario]?'\n"
            "- 'What are the exceptions to [general rule]?'\n\n"
            "Answers should demonstrate deep rules knowledge and careful reasoning."
        ),
        "answer_guidance": (
            "Address the complexity directly. Acknowledge edge cases and exceptions. "
            "If the rules are ambiguous, state what IS clear and what requires GM judgment. "
            "Show step-by-step reasoning for complex interactions."
        ),
    },
}

# Default distribution weights
DEFAULT_DISTRIBUTION = {
    "basic": 0.30,
    "intermediate": 0.50,
    "advanced": 0.20,
}


def select_difficulty(
    distribution: Dict[str, float],
    seed: int,
    chunk_index: int,
    question_index: int = 0,
) -> str:
    """
    Deterministically select a difficulty level based on weighted distribution.
    
    Args:
        distribution: Dict mapping difficulty -> weight (should sum to ~1.0)
        seed: Random seed for reproducibility
        chunk_index: Current chunk being processed
        question_index: Index of question within chunk (for variation)
        
    Returns:
        Selected difficulty level string
    """
    # Create deterministic hash
    hash_input = f"difficulty:{seed}:{chunk_index}:{question_index}".encode()
    hash_val = int(hashlib.sha256(hash_input).hexdigest()[:8], 16)
    normalized = hash_val / 0xFFFFFFFF  # 0.0 to 1.0
    
    # Normalize distribution
    total_weight = sum(distribution.values())
    if total_weight == 0:
        return "intermediate"  # Fallback
    
    # Select based on cumulative probability
    cumulative = 0.0
    for level, weight in distribution.items():
        cumulative += weight / total_weight
        if normalized < cumulative:
            return level
    
    # Fallback to last level
    return list(distribution.keys())[-1]


def get_difficulty_instructions(difficulty: str) -> str:
    """Get the prompt instructions for a difficulty level."""
    level_config = DIFFICULTY_LEVELS.get(difficulty, DIFFICULTY_LEVELS["intermediate"])
    return level_config["instructions"]


def get_difficulty_answer_guidance(difficulty: str) -> str:
    """Get the answer style guidance for a difficulty level."""
    level_config = DIFFICULTY_LEVELS.get(difficulty, DIFFICULTY_LEVELS["intermediate"])
    return level_config["answer_guidance"]


def build_difficulty_prompt_section(difficulty: str) -> str:
    """
    Build a complete prompt section for the given difficulty level.
    
    Returns a formatted string to inject into generation prompts.
    """
    level_config = DIFFICULTY_LEVELS.get(difficulty, DIFFICULTY_LEVELS["intermediate"])
    
    return (
        f"### Difficulty Level: {difficulty.upper()}\n"
        f"{level_config['instructions']}\n\n"
        f"### Answer Style\n"
        f"{level_config['answer_guidance']}"
    )


class DifficultyStats:
    """Tracks difficulty distribution across generated examples."""

    def __init__(self):
        self.counts: Dict[str, int] = {
            "basic": 0,
            "intermediate": 0,
            "advanced": 0,
        }

    def record(self, difficulty: str) -> None:
        if difficulty in self.counts:
            self.counts[difficulty] += 1
        else:
            self.counts[difficulty] = 1

    def total(self) -> int:
        return sum(self.counts.values())

    def distribution(self) -> Dict[str, float]:
        total = self.total()
        if total == 0:
            return {k: 0.0 for k in self.counts}
        return {k: v / total for k, v in self.counts.items()}

    def summary(self) -> Dict:
        return {
            "counts": dict(self.counts),
            "total": self.total(),
            "distribution": self.distribution(),
        }

    def print_summary(self, target_distribution: Optional[Dict[str, float]] = None) -> None:
        s = self.summary()
        print("\n--- Difficulty Distribution ---")
        print(f"Total: {s['total']}")
        for level in ["basic", "intermediate", "advanced"]:
            count = s["counts"].get(level, 0)
            pct = s["distribution"].get(level, 0.0)
            target = target_distribution.get(level, 0.0) if target_distribution else None
            if target is not None:
                print(f"  {level}: {count} ({pct:.1%}) [target: {target:.1%}]")
            else:
                print(f"  {level}: {count} ({pct:.1%})")


def validate_distribution(distribution: Dict[str, float]) -> Tuple[bool, str]:
    """
    Validate a difficulty distribution config.
    
    Returns (is_valid, error_message)
    """
    if not distribution:
        return False, "distribution cannot be empty"
    
    valid_levels = set(DIFFICULTY_LEVELS.keys())
    for level in distribution:
        if level not in valid_levels:
            return False, f"unknown difficulty level: {level}"
    
    total = sum(distribution.values())
    if total <= 0:
        return False, "distribution weights must sum to > 0"
    
    for level, weight in distribution.items():
        if weight < 0:
            return False, f"negative weight for {level}"
    
    return True, ""


# Task-type specific overrides
# Some task types may benefit from different difficulty distributions
TASK_TYPE_OVERRIDES = {
    "lore": {
        "basic": 0.50,
        "intermediate": 0.40,
        "advanced": 0.10,
    },
    "character_build": {
        "basic": 0.20,
        "intermediate": 0.50,
        "advanced": 0.30,
    },
    "gm_guidance": {
        "basic": 0.20,
        "intermediate": 0.45,
        "advanced": 0.35,
    },
}


def get_distribution_for_task(
    task_type: str,
    base_distribution: Dict[str, float],
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """
    Get the difficulty distribution for a specific task type.
    
    Uses task-specific override if defined, otherwise returns base distribution.
    """
    if overrides is None:
        overrides = TASK_TYPE_OVERRIDES
    
    return overrides.get(task_type, base_distribution)

