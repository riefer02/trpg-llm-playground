"""Narrative check tier rules and helpers for Lancer TTRPG."""

from typing import Literal
from pydantic import BaseModel, Field

from core.shared.rolls import RollType


NarrativeCheckTier = Literal["standard", "risky", "heroic"]


class NarrativeCheckTierRule(BaseModel):
    """Tier-specific narrative check behavior."""

    tier: NarrativeCheckTier
    success_threshold: int = Field(..., ge=0)
    consequence_threshold: int | None = Field(
        default=None,
        ge=0,
        description="Roll needed to avoid consequences on success",
    )
    allows_push: bool = True

    model_config = {"frozen": True}


class NarrativeHelpRule(BaseModel):
    """Rules for helping on a narrative check."""

    accuracy_bonus: int = Field(default=1, ge=0)
    helpers_share_consequences: bool = True

    model_config = {"frozen": True}


class NarrativePushRule(BaseModel):
    """Rules for pushing a narrative check to a higher tier."""

    from_tier: NarrativeCheckTier
    to_tier: NarrativeCheckTier
    requires_gm_approval: bool = False

    model_config = {"frozen": True}


class NarrativeCheckRules(BaseModel):
    """Combined narrative check rules."""

    default_target: int = Field(default=10, ge=0)
    tiers: list[NarrativeCheckTierRule] = Field(default_factory=list)
    help_rule: NarrativeHelpRule = Field(default_factory=NarrativeHelpRule)
    push_rules: list[NarrativePushRule] = Field(default_factory=list)
    voluntary_failure_allowed: bool = True
    voluntary_failure_applies_to: list[RollType] = Field(
        default_factory=lambda: ["skill_check", "save"],
    )

    model_config = {"frozen": True}


NARRATIVE_TIER_RULES: list[NarrativeCheckTierRule] = [
    NarrativeCheckTierRule(
        tier="standard",
        success_threshold=10,
        consequence_threshold=None,
        allows_push=True,
    ),
    NarrativeCheckTierRule(
        tier="risky",
        success_threshold=10,
        consequence_threshold=20,
        allows_push=True,
    ),
    NarrativeCheckTierRule(
        tier="heroic",
        success_threshold=20,
        consequence_threshold=20,
        allows_push=False,
    ),
]


NARRATIVE_TIER_RULES_BY_TIER = {rule.tier: rule for rule in NARRATIVE_TIER_RULES}


NARRATIVE_PUSH_RULES: list[NarrativePushRule] = [
    NarrativePushRule(from_tier="standard", to_tier="risky", requires_gm_approval=False),
    NarrativePushRule(from_tier="risky", to_tier="heroic", requires_gm_approval=True),
]


DEFAULT_NARRATIVE_CHECK_RULES = NarrativeCheckRules(
    tiers=NARRATIVE_TIER_RULES,
    push_rules=NARRATIVE_PUSH_RULES,
)


def get_narrative_tier_rule(tier: NarrativeCheckTier) -> NarrativeCheckTierRule | None:
    """Look up a tier rule by name."""
    return NARRATIVE_TIER_RULES_BY_TIER.get(tier)
