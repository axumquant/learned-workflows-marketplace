"""028_learned_workflow_versions

Adds:
  - learned_workflow_versions table — every UPDATE to a workflow
    snapshots the previous state here so admins can roll back
  - learned_workflow_stats view — aggregates use_count / success_rate /
    last_used per workflow from learned_workflow_invocations

Revises: 027_learned_workflows
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "028_learned_workflow_versions"
down_revision = "027_learned_workflows"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "learned_workflow_versions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("workflow_id", UUID(as_uuid=True), sa.ForeignKey("learned_workflows.id", ondelete="CASCADE"), nullable=False),
        sa.Column("organization_id", UUID(as_uuid=True), nullable=False),
        sa.Column("version_number", sa.Integer, nullable=False),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("display_name", sa.Text, nullable=False),
        sa.Column("description", sa.Text, nullable=False, server_default=""),
        sa.Column("skill_prompt", sa.Text, nullable=False, server_default=""),
        sa.Column("parameters", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("actions", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("tags", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("portal", sa.Text, nullable=True),
        sa.Column("scope", sa.Text, nullable=False),
        sa.Column("visibility", sa.Text, nullable=False),
        sa.Column("change_summary", sa.Text, nullable=False, server_default=""),
        sa.Column("created_by_user_id", UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "ix_learned_workflow_versions_workflow_id_version",
        "learned_workflow_versions",
        ["workflow_id", "version_number"],
        unique=True,
    )
    op.create_index("ix_learned_workflow_versions_org_id", "learned_workflow_versions", ["organization_id"])

    # Add a current_version column to learned_workflows (defaults to 1
    # for all existing rows; new edits bump it).
    op.add_column(
        "learned_workflows",
        sa.Column("current_version", sa.Integer, nullable=False, server_default="1"),
    )

    # Stats view — postgres view that aggregates invocations.
    op.execute(
        """
        CREATE OR REPLACE VIEW learned_workflow_stats AS
        SELECT
            w.id                          AS workflow_id,
            w.organization_id,
            w.name,
            w.display_name,
            w.scope,
            w.visibility,
            w.enabled,
            w.current_version,
            COUNT(i.id)                   AS use_count,
            COUNT(i.id) FILTER (WHERE i.success = true) AS success_count,
            COUNT(i.id) FILTER (WHERE i.success = false) AS failure_count,
            CASE WHEN COUNT(i.id) > 0
                 THEN ROUND( (COUNT(i.id) FILTER (WHERE i.success = true)::numeric)
                           / COUNT(i.id)::numeric, 4)
                 ELSE 0 END               AS success_rate,
            MAX(i.created_at)             AS last_used_at,
            COUNT(DISTINCT i.invoked_by_user_id) FILTER (WHERE i.invoked_by_user_id IS NOT NULL) AS unique_users
        FROM learned_workflows w
        LEFT JOIN learned_workflow_invocations i ON i.workflow_id = w.id
        GROUP BY w.id, w.organization_id, w.name, w.display_name, w.scope,
                 w.visibility, w.enabled, w.current_version;
        """
    )


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS learned_workflow_stats;")
    op.drop_column("learned_workflows", "current_version")
    op.drop_index("ix_learned_workflow_versions_org_id", table_name="learned_workflow_versions")
    op.drop_index("ix_learned_workflow_versions_workflow_id_version", table_name="learned_workflow_versions")
    op.drop_table("learned_workflow_versions")
