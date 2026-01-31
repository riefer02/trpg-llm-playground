"""Tests for salvage system on Pilot model."""

import pytest
from pydantic import ValidationError

from core.pilot.pilot import Pilot


def test_salvage_default_zero() -> None:
    """Pilot should start with zero salvage."""
    pilot = Pilot(callsign="Test", name="Test Pilot")
    assert pilot.salvage == 0


def test_add_salvage_increases_value() -> None:
    """Adding salvage should increase the pilot's salvage amount."""
    pilot = Pilot(callsign="Test", name="Test Pilot")
    assert pilot.salvage == 0

    updated = pilot.add_salvage(100)
    assert updated.salvage == 100

    # Original unchanged (immutable)
    assert pilot.salvage == 0


def test_add_salvage_negative_raises() -> None:
    """Adding negative salvage should raise ValueError."""
    pilot = Pilot(callsign="Test", name="Test Pilot")
    with pytest.raises(ValueError):
        pilot.add_salvage(-50)


def test_spend_salvage_decreases_value() -> None:
    """Spending salvage should decrease the pilot's salvage amount."""
    pilot = Pilot(callsign="Test", name="Test Pilot").add_salvage(200)
    assert pilot.salvage == 200

    updated = pilot.spend_salvage(75)
    assert updated.salvage == 125

    # Original unchanged
    assert pilot.salvage == 200


def test_spend_salvage_insufficient_raises() -> None:
    """Spending more salvage than available should raise ValueError."""
    pilot = Pilot(callsign="Test", name="Test Pilot").add_salvage(50)
    assert pilot.salvage == 50

    with pytest.raises(ValueError):
        pilot.spend_salvage(100)


def test_spend_salvage_negative_raises() -> None:
    """Spending negative salvage should raise ValueError."""
    pilot = Pilot(callsign="Test", name="Test Pilot").add_salvage(100)
    with pytest.raises(ValueError):
        pilot.spend_salvage(-10)


def test_salvage_cannot_be_negative() -> None:
    """Directly setting salvage to negative via model creation should fail."""
    with pytest.raises(ValidationError):
        Pilot(callsign="Test", name="Test Pilot", salvage=-5)
