"""Add characters table.

Revision ID: 003_add_characters
Revises: 002_add_combat_sessions
Create Date: 2026-01-10
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "003_add_characters"
down_revision = "002_add_combat_sessions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create characters table."""
    op.create_table(
        "characters",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("callsign", sa.String(), nullable=False),
        sa.Column("data", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("campaign_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_characters_callsign"),
        "characters",
        ["callsign"],
        unique=False,
    )
    op.create_index(
        op.f("ix_characters_user_id"),
        "characters",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_characters_campaign_id"),
        "characters",
        ["campaign_id"],
        unique=False,
    )


def downgrade() -> None:
    """Drop characters table."""
    op.drop_index(op.f("ix_characters_campaign_id"), table_name="characters")
    op.drop_index(op.f("ix_characters_user_id"), table_name="characters")
    op.drop_index(op.f("ix_characters_callsign"), table_name="characters")
    op.drop_table("characters")
