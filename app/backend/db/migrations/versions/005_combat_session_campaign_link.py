"""Add campaign session linkage to combat sessions.

Revision ID: 005_combat_session_campaign_link
Revises: 004_campaign_memberships
Create Date: 2026-01-15
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "005_combat_session_campaign_link"
down_revision = "004_campaign_memberships"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add campaign_session_id column and index to combat sessions."""
    op.add_column(
        "combat_sessions",
        sa.Column("campaign_session_id", sa.String(), nullable=True),
    )
    op.create_index(
        op.f("ix_combat_sessions_campaign_session_id"),
        "combat_sessions",
        ["campaign_session_id"],
        unique=False,
    )


def downgrade() -> None:
    """Drop campaign_session_id column and index."""
    op.drop_index(
        op.f("ix_combat_sessions_campaign_session_id"),
        table_name="combat_sessions",
    )
    op.drop_column("combat_sessions", "campaign_session_id")
