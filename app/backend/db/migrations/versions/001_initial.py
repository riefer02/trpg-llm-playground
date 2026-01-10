"""Initial schema with pilots, mechs, campaigns.

Revision ID: 001_initial
Revises: None
Create Date: 2026-01-10
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create pilots, mechs, and campaigns tables."""
    # Pilots table
    op.create_table(
        "pilots",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("data", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("campaign_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_pilots_campaign_id"), "pilots", ["campaign_id"], unique=False)
    op.create_index(op.f("ix_pilots_name"), "pilots", ["name"], unique=False)
    op.create_index(op.f("ix_pilots_user_id"), "pilots", ["user_id"], unique=False)

    # Mechs table
    op.create_table(
        "mechs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("frame_id", sa.String(), nullable=False),
        sa.Column("data", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column("pilot_id", sa.String(), nullable=True),
        sa.Column("campaign_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_mechs_campaign_id"), "mechs", ["campaign_id"], unique=False)
    op.create_index(op.f("ix_mechs_frame_id"), "mechs", ["frame_id"], unique=False)
    op.create_index(op.f("ix_mechs_name"), "mechs", ["name"], unique=False)
    op.create_index(op.f("ix_mechs_pilot_id"), "mechs", ["pilot_id"], unique=False)

    # Campaigns table
    op.create_table(
        "campaigns",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("gm_user_id", sa.String(), nullable=False),
        sa.Column("settings", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_campaigns_gm_user_id"), "campaigns", ["gm_user_id"], unique=False)
    op.create_index(op.f("ix_campaigns_name"), "campaigns", ["name"], unique=False)


def downgrade() -> None:
    """Drop all initial tables."""
    op.drop_index(op.f("ix_campaigns_name"), table_name="campaigns")
    op.drop_index(op.f("ix_campaigns_gm_user_id"), table_name="campaigns")
    op.drop_table("campaigns")

    op.drop_index(op.f("ix_mechs_pilot_id"), table_name="mechs")
    op.drop_index(op.f("ix_mechs_name"), table_name="mechs")
    op.drop_index(op.f("ix_mechs_frame_id"), table_name="mechs")
    op.drop_index(op.f("ix_mechs_campaign_id"), table_name="mechs")
    op.drop_table("mechs")

    op.drop_index(op.f("ix_pilots_user_id"), table_name="pilots")
    op.drop_index(op.f("ix_pilots_name"), table_name="pilots")
    op.drop_index(op.f("ix_pilots_campaign_id"), table_name="pilots")
    op.drop_table("pilots")
