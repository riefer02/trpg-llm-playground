"""Add invite note field to campaign invites.

Revision ID: 006_campaign_invite_note
Revises: 005_combat_session_campaign_link
Create Date: 2026-01-18
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "006_campaign_invite_note"
down_revision = "005_combat_session_campaign_link"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add invite_note column to campaign invites."""
    op.add_column(
        "campaign_invites",
        sa.Column("invite_note", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    """Drop invite_note column from campaign invites."""
    op.drop_column("campaign_invites", "invite_note")
