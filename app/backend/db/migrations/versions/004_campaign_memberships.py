"""Expand campaign persistence with memberships and invites.

Revision ID: 004_campaign_memberships
Revises: 003_add_characters
Create Date: 2026-01-15
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "004_campaign_memberships"
down_revision = "003_add_characters"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add campaign metadata columns and membership tables."""
    # Expand campaigns table
    op.add_column("campaigns", sa.Column("user_id", sa.String(), nullable=True))
    op.execute("UPDATE campaigns SET user_id = gm_user_id")
    op.alter_column("campaigns", "user_id", nullable=False)
    op.create_index(
        op.f("ix_campaigns_user_id"), "campaigns", ["user_id"], unique=False
    )

    op.add_column(
        "campaigns",
        sa.Column("status", sa.String(), nullable=False, server_default="active"),
    )
    op.add_column(
        "campaigns",
        sa.Column("visibility", sa.String(), nullable=False, server_default="private"),
    )
    op.add_column(
        "campaigns",
        sa.Column(
            "data",
            postgresql.JSON(astext_type=sa.Text()),
            nullable=True,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )

    # Drop legacy GM column/index
    op.drop_index(op.f("ix_campaigns_gm_user_id"), table_name="campaigns")
    op.drop_column("campaigns", "gm_user_id")

    # Campaign memberships
    op.create_table(
        "campaign_memberships",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("campaign_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False, server_default="player"),
        sa.Column("status", sa.String(), nullable=False, server_default="active"),
        sa.Column(
            "ready_state",
            sa.String(),
            nullable=False,
            server_default="not_ready",
        ),
        sa.Column("assigned_character_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["campaign_id"], ["campaigns.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("campaign_id", "user_id", name="uq_campaign_member"),
    )
    op.create_index(
        op.f("ix_campaign_memberships_campaign_id"),
        "campaign_memberships",
        ["campaign_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_campaign_memberships_user_id"),
        "campaign_memberships",
        ["user_id"],
        unique=False,
    )

    # Campaign invites
    op.create_table(
        "campaign_invites",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("campaign_id", sa.String(), nullable=False),
        sa.Column("invited_by_user_id", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False, server_default="player"),
        sa.Column("token", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("invited_email", sa.String(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("redeemed_by_user_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["campaign_id"], ["campaigns.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token", name="uq_campaign_invite_token"),
    )
    op.create_index(
        op.f("ix_campaign_invites_campaign_id"),
        "campaign_invites",
        ["campaign_id"],
        unique=False,
    )

    # Campaign character attachments
    op.create_table(
        "campaign_characters",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("campaign_id", sa.String(), nullable=False),
        sa.Column("character_id", sa.String(), nullable=False),
        sa.Column("added_by_user_id", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False, server_default="player"),
        sa.Column("notes", sa.String(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["campaign_id"], ["campaigns.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["character_id"], ["characters.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "campaign_id", "character_id", name="uq_campaign_character"
        ),
    )
    op.create_index(
        op.f("ix_campaign_characters_campaign_id"),
        "campaign_characters",
        ["campaign_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_campaign_characters_character_id"),
        "campaign_characters",
        ["character_id"],
        unique=False,
    )


def downgrade() -> None:
    """Drop membership tables and campaign metadata columns."""
    op.drop_index(
        op.f("ix_campaign_characters_character_id"), table_name="campaign_characters"
    )
    op.drop_index(
        op.f("ix_campaign_characters_campaign_id"), table_name="campaign_characters"
    )
    op.drop_table("campaign_characters")

    op.drop_index(
        op.f("ix_campaign_invites_campaign_id"), table_name="campaign_invites"
    )
    op.drop_table("campaign_invites")

    op.drop_index(
        op.f("ix_campaign_memberships_user_id"), table_name="campaign_memberships"
    )
    op.drop_index(
        op.f("ix_campaign_memberships_campaign_id"), table_name="campaign_memberships"
    )
    op.drop_table("campaign_memberships")

    op.drop_index(op.f("ix_campaigns_user_id"), table_name="campaigns")
    op.add_column(
        "campaigns",
        sa.Column("gm_user_id", sa.String(), nullable=False, server_default=""),
    )
    op.execute("UPDATE campaigns SET gm_user_id = user_id")
    op.alter_column("campaigns", "gm_user_id", server_default=None)
    op.create_index(
        op.f("ix_campaigns_gm_user_id"), "campaigns", ["gm_user_id"], unique=False
    )
    op.drop_column("campaigns", "user_id")

    op.drop_column("campaigns", "data")
    op.drop_column("campaigns", "visibility")
    op.drop_column("campaigns", "status")
