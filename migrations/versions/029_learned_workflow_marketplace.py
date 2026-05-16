"""029_learned_workflow_marketplace

Adds two columns to learned_workflows so an org can opt-in to publish
an APPROVED workflow to the cross-org marketplace, and another org
can install a fork of it:

  - marketplace_published: bool  (default false)
      When true, the row appears in /v1/learned-workflows/marketplace
      regardless of organization_id (read-only across tenants).
  - marketplace_source_workflow_id: uuid (nullable)
      On install, the new row's source_workflow_id points back to the
      published parent so we can show "Installed from <org>" and bump
      a counter on the source.
  - marketplace_install_count: int (default 0)
      Bumped on every install. Visible in marketplace listings.

Revises: 028_learned_workflow_versions
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "029_learned_workflow_marketplace"
down_revision = "028_learned_workflow_versions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "learned_workflows",
        sa.Column("marketplace_published", sa.Boolean, nullable=False, server_default=sa.text("false")),
    )
    op.add_column(
        "learned_workflows",
        sa.Column("marketplace_source_workflow_id", UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "learned_workflows",
        sa.Column("marketplace_install_count", sa.Integer, nullable=False, server_default=sa.text("0")),
    )
    # Filtered index so marketplace listings stay fast as the table grows.
    op.create_index(
        "ix_learned_workflows_marketplace_published",
        "learned_workflows",
        ["marketplace_published"],
        postgresql_where=sa.text("marketplace_published = true"),
    )
    # Walk-back index for "installed from"
    op.create_index(
        "ix_learned_workflows_marketplace_source",
        "learned_workflows",
        ["marketplace_source_workflow_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_learned_workflows_marketplace_source", table_name="learned_workflows")
    op.drop_index("ix_learned_workflows_marketplace_published", table_name="learned_workflows")
    op.drop_column("learned_workflows", "marketplace_install_count")
    op.drop_column("learned_workflows", "marketplace_source_workflow_id")
    op.drop_column("learned_workflows", "marketplace_published")
