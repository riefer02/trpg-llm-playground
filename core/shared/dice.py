"""Dice expression types and utilities for Lancer TTRPG."""

from __future__ import annotations
import random
import re
from typing import Literal
from pydantic import BaseModel, field_validator

# Standard dice sizes used in Lancer
DieSize = Literal[3, 6, 20]


class DiceExpression(BaseModel):
    """
    Represents a dice expression like "1d6", "2d6+2", or "1d20".
    
    Lancer primarily uses d6 and d20:
    - d20 for skill checks (roll high)
    - d6 for damage (1d6 per damage die)
    - d3 rarely used (typically 1d6/2)
    """
    
    count: int = 1
    size: DieSize = 6
    modifier: int = 0
    
    model_config = {"frozen": True}
    
    @field_validator("count")
    @classmethod
    def count_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("dice count must be at least 1")
        return v
    
    @classmethod
    def parse(cls, expr: str) -> DiceExpression:
        """
        Parse a dice expression string.
        
        Examples:
            DiceExpression.parse("1d6") -> DiceExpression(count=1, size=6, modifier=0)
            DiceExpression.parse("2d6+2") -> DiceExpression(count=2, size=6, modifier=2)
            DiceExpression.parse("1d20-1") -> DiceExpression(count=1, size=20, modifier=-1)
        """
        expr = expr.lower().strip()
        match = re.match(r"(\d+)?d(\d+)([+-]\d+)?", expr)
        if not match:
            raise ValueError(f"Invalid dice expression: {expr}")
        
        count = int(match.group(1)) if match.group(1) else 1
        size = int(match.group(2))
        modifier = int(match.group(3)) if match.group(3) else 0
        
        return cls(count=count, size=size, modifier=modifier)  # type: ignore[arg-type]
    
    def __str__(self) -> str:
        base = f"{self.count}d{self.size}"
        if self.modifier > 0:
            return f"{base}+{self.modifier}"
        elif self.modifier < 0:
            return f"{base}{self.modifier}"
        return base
    
    def min_value(self) -> int:
        """Minimum possible roll."""
        return self.count + self.modifier
    
    def max_value(self) -> int:
        """Maximum possible roll."""
        return (self.count * self.size) + self.modifier
    
    def average(self) -> float:
        """Expected average roll."""
        return (self.count * (self.size + 1) / 2) + self.modifier


def roll_dice(expr: str | DiceExpression) -> int:
    """
    Roll dice and return the total.
    
    Args:
        expr: A dice expression string ("2d6+2") or DiceExpression object
        
    Returns:
        Total of all dice plus modifier
    """
    if isinstance(expr, str):
        expr = DiceExpression.parse(expr)
    
    total = sum(random.randint(1, expr.size) for _ in range(expr.count))
    return total + expr.modifier


def roll_with_advantage(size: DieSize = 20) -> tuple[int, int, int]:
    """
    Roll with accuracy (Lancer's advantage mechanic).
    
    In Lancer, accuracy adds +1d6 per accuracy point.
    The highest single d6 is added to the d20 roll.
    
    Returns:
        Tuple of (total, d20_roll, accuracy_bonus)
    """
    d20 = random.randint(1, size)
    # For simplicity, this rolls 1 accuracy die
    accuracy = random.randint(1, 6)
    return (d20 + accuracy, d20, accuracy)

