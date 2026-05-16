"""Learned workflows — reusable MCP tools synthesized from automation runs.

The ``learned_workflows`` table backs the "Save as Workflow" flow in the
Automation Lab overlay and the catalog browser in Extension Settings.

Each row is a reusable browser-automation macro:

  - ``name`` is a stable snake_case identifier used as the MCP tool name
    when the autopilot agent calls it.
  - ``display_name`` is the human-readable label shown in the catalog.
  - ``description`` and ``skill_prompt`` are LLM-generated (Ollama Cloud,
    deepseek-v4-pro) from the action sequence and page context, then
    reviewed/edited by the user before save.
  - ``parameters`` lists the runtime inputs (e.g. ``zip``, ``mbi``) that
    get substituted into action ``value`` fields at execution time.
  - ``actions`` is the recorded sequence of ``ExecutorAction`` items,
    with ``{{param}}`` placeholders where typed values came from user
    input or seed_data.
  - ``tags`` are auto-generated tags for search and filtering.
  - ``portal`` ties this workflow to a specific portal (sunfire,
    enrollhere, etc.) so the catalog can scope by site.
  - ``scope`` is ``user`` or ``org`` — org-scoped workflows go through
    an approval queue before they're invokable by other org members.
  - ``visibility`` is ``private``, ``pending_approval``, ``approved``,
    or ``rejected``. User-scoped workflows skip approval and go
    straight to ``approved``. Org-scoped start at ``pending_approval``
    until an admin approves.
  - ``enabled`` is a soft kill-switch — disabled tools are hidden from
    the agent and invoke UI but kept in the catalog for re-enable.

Tenant isolation: organization_id is required and RLS-enforced.

Revision ID: 027_learned_workflows
Revises: 026_integration_sync_jobs_errors_compat
Create Date: 2026-05-15
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "027_learned_workflows"
down_revision = "026_integration_sync_jobs_errors_compat"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "learned_workflows",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("organization_id", UUID(as_uuid=True), nullable=False),
        sa.Column("created_by_user_id", UUID(as_uuid=True), nullable=False),
        # Stable machine-callable identifier (snake_case). Unique per org.
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("display_name", sa.String(length=200), nullable=False, server_default=""),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        # Skill prompt injected into the autopilot agent's system prompt
        # when this workflow is in the agent's available toolset.
        sa.Column("skill_prompt", sa.Text(), nullable=False, server_default=""),
        # Portal/provider key (sunfire, enrollhere, etc.)
        sa.Column("portal", sa.String(length=64), nullable=False, server_default=""),
        # JSONB: list of {name, type, description, pattern, required}
        sa.Column(
            "parameters",
            JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        # JSONB: list of ExecutorAction with {{param}} placeholders
        sa.Column(
            "actions",
            JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "tags",
            sa.ARRAY(sa.String(length=64)),
            nullable=False,
            server_default=sa.text("ARRAY[]::varchar[]"),
        ),
        # Sharing scope: "user" or "org"
        sa.Column(
            "scope",
            sa.String(length=16),
            nullable=False,
            server_default="user",
        ),
        # Approval workflow state
        sa.Column(
            "visibility",
            sa.String(length=24),
            nullable=False,
            server_default="approved",
        ),
        sa.Column("approved_by_user_id", UUID(as_uuid=True), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rejection_reason", sa.Text(), nullable=False, server_default=""),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        # Provenance — link back to the run that produced this workflow
        sa.Column("source_run_id", UUID(as_uuid=True), nullable=True),
        # Usage stats — incremented when the agent invokes this workflow
        sa.Column("use_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        # Embedding metadata — set after Qdrant upsert succeeds
        sa.Column("qdrant_point_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Unique workflow name per org (excluding soft-deleted)
    op.execute(
        "CREATE UNIQUE INDEX ix_learned_workflows_org_name "
        "ON learned_workflows (organization_id, name) "
        "WHERE deleted_at IS NULL"
    )
    op.create_index(
        "ix_learned_workflows_org_portal",
        "learned_workflows",
        ["organization_id", "portal"],
    )
    op.create_index(
        "ix_learned_workflows_org_scope_visibility",
        "learned_workflows",
        ["organization_id", "scope", "visibility"],
    )
    op.create_index(
        "ix_learned_workflows_created_by",
        "learned_workflows",
        ["created_by_user_id"],
    )

    # Tenant isolation via RLS
    op.execute("ALTER TABLE public.learned_workflows ENABLE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY learned_workflows_tenant_isolation
        ON public.learned_workflows
        FOR ALL
        USING (
            organization_id = (
                (auth.jwt() -> 'app_metadata' ->> 'organization_id')::uuid
            )
        )
        WITH CHECK (
            organization_id = (
                (auth.jwt() -> 'app_metadata' ->> 'organization_id')::uuid
            )
        )
        """
    )

    # Invocation log — captures each call to a learned workflow so the
    # portal can show stats and the platform can detect drift.
    op.create_table(
        "learned_workflow_invocations",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("organization_id", UUID(as_uuid=True), nullable=False),
        sa.Column("workflow_id", UUID(as_uuid=True), nullable=False),
        sa.Column("invoked_by_user_id", UUID(as_uuid=True), nullable=False),
        sa.Column(
            "params",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("success", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("actions_executed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "metadata",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_learned_workflow_invocations_workflow",
        "learned_workflow_invocations",
        ["workflow_id", "created_at"],
    )
    op.create_index(
        "ix_learned_workflow_invocations_org_user",
        "learned_workflow_invocations",
        ["organization_id", "invoked_by_user_id"],
    )
    op.execute("ALTER TABLE public.learned_workflow_invocations ENABLE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY learned_workflow_invocations_tenant_isolation
        ON public.learned_workflow_invocations
        FOR ALL
        USING (
            organization_id = (
                (auth.jwt() -> 'app_metadata' ->> 'organization_id')::uuid
            )
        )
        WITH CHECK (
            organization_id = (
                (auth.jwt() -> 'app_metadata' ->> 'organization_id')::uuid
            )
        )
        """
    )


def downgrade() -> None:
    op.execute(
        "DROP POLICY IF EXISTS learned_workflow_invocations_tenant_isolation "
        "ON public.learned_workflow_invocations"
    )
    op.drop_index("ix_learned_workflow_invocations_org_user", table_name="learned_workflow_invocations")
    op.drop_index("ix_learned_workflow_invocations_workflow", table_name="learned_workflow_invocations")
    op.drop_table("learned_workflow_invocations")

    op.execute(
        "DROP POLICY IF EXISTS learned_workflows_tenant_isolation ON public.learned_workflows"
    )
    op.drop_index("ix_learned_workflows_created_by", table_name="learned_workflows")
    op.drop_index("ix_learned_workflows_org_scope_visibility", table_name="learned_workflows")
    op.drop_index("ix_learned_workflows_org_portal", table_name="learned_workflows")
    op.execute("DROP INDEX IF EXISTS ix_learned_workflows_org_name")
    op.drop_table("learned_workflows")
