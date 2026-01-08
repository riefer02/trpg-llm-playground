"""Shared Pydantic base models."""

from pydantic import BaseModel, ConfigDict

__all__ = ["FrozenModel"]


class FrozenModel(BaseModel):
    """Immutable model base for rules data."""

    model_config = ConfigDict(frozen=True)
