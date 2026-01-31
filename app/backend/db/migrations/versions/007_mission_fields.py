"""Add mission_id and mission_difficulty fields to combat sessions.

Revision ID: 007_mission_fields
Revises: 006_campaign_invite_note
Create Date: 2026-01-31
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "007_mission_fields"
down_revision = "006_campaign_invite_note"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add mission_id and mission_difficulty columns to combat_sessions."""
    op.add_column(
        "combat_sessions",
        sa.Column("mission_id", sa.String(), nullable=True),
    )
    op.add_column(
        "combat_sessions",
        sa.Column("mission_difficulty", sa.Integer(), nullable=True),
    )
    op.create_index("ix_combat_sessions_mission_id", "combat_sessions", ["mission_id"])


def downgrade() -> None:
    """Drop mission_id and mission_difficulty columns from combat_sessions."""
    op.drop_index("ix_combat_sessions_mission_id", table_name="combat_sessions")
    op.drop_column("combat_sessions", "mission_difficulty")
    op.drop_column("combat_sessions", "mission_id")
