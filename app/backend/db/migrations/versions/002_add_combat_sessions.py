"""Add combat_sessions table.

Revision ID: 002_add_combat_sessions
Revises: 001_initial
Create Date: 2026-01-10
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "002_add_combat_sessions"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create combat_sessions table."""
    op.create_table(
        "combat_sessions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("current_round", sa.Integer(), nullable=False),
        sa.Column("current_turn_index", sa.Integer(), nullable=False),
        sa.Column("scenario", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column("gm_user_id", sa.String(), nullable=False),
        sa.Column("campaign_id", sa.String(), nullable=True),
        sa.Column("notes", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_combat_sessions_campaign_id"),
        "combat_sessions",
        ["campaign_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_combat_sessions_gm_user_id"),
        "combat_sessions",
        ["gm_user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_combat_sessions_name"),
        "combat_sessions",
        ["name"],
        unique=False,
    )
    op.create_index(
        op.f("ix_combat_sessions_status"),
        "combat_sessions",
        ["status"],
        unique=False,
    )


def downgrade() -> None:
    """Drop combat_sessions table."""
    op.drop_index(op.f("ix_combat_sessions_status"), table_name="combat_sessions")
    op.drop_index(op.f("ix_combat_sessions_name"), table_name="combat_sessions")
    op.drop_index(op.f("ix_combat_sessions_gm_user_id"), table_name="combat_sessions")
    op.drop_index(op.f("ix_combat_sessions_campaign_id"), table_name="combat_sessions")
    op.drop_table("combat_sessions")
