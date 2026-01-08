"""Integration tests for Lancer TTRPG mechanical system.

This module contains comprehensive integration tests demonstrating the complete
game loop from character creation through campaign completion.

Tests follow the session structure defined in PR2:
    Brief → Preparation → Boots on Ground → Narrative Play → Combat → Debrief → Downtime
"""

from core.tests.conftest import (
    integration_pilot_ll0,
    integration_pilot_ll3,
    integration_mech_everest,
    integration_mech_raleigh,
    integration_sitrep_template,
    integration_sitrep_control_template,
    integration_campaign,
    integration_active_session,
    integration_narrative_tracker,
    integration_pilot_with_talents,
    integration_mech_minimal,
)
