"""REST API for learned workflows — the reusable browser-automation tools
the user synthesizes from Automation Lab runs.

Endpoints:

  POST   /v1/learned-workflows/synthesize-from-run/{run_id}
                                          → propose a workflow (NOT saved)
  POST   /v1/learned-workflows            → save a (potentially edited) workflow
  GET    /v1/learned-workflows            → list workflows visible to the caller
  GET    /v1/learned-workflows/{id}       → fetch one
  PATCH  /v1/learned-workflows/{id}       → update fields (creator or org admin)
  DELETE /v1/learned-workflows/{id}       → soft-delete (creator or org admin)
  POST   /v1/learned-workflows/{id}/enable
  POST   /v1/learned-workflows/{id}/disable
  POST   /v1/learned-workflows/{id}/approve  → org admin approves a pending workflow
  POST   /v1/learned-workflows/{id}/reject   → org admin rejects with reason
  POST   /v1/learned-workflows/{id}/invoke   → expand params into executable
                                                action list (returned to caller)
  POST   /v1/learned-workflows/search        → semantic search (Qdrant — TODO)

Auth: every route requires an authenticated TenantContext. Org id is
ALWAYS sourced from the JWT, never from the request body. Tenant
isolation is enforced both in code AND by Postgres RLS (migration 027).
"""

from __future__ import annotations

import re
from typing import Annotated, Any
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from learned_workflows_marketplace.auth import TenantContext, get_tenant_context
from learned_workflows_marketplace.learned_workflows import (
    VALID_SCOPES,
    VALID_VISIBILITIES,
    LearnedWorkflow,
    get_learned_workflow_repository,
    normalize_workflow_name,
)

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1/learned-workflows", tags=["learned-workflows"])
Tenant = Annotated[TenantContext, Depends(get_tenant_context)]


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------


class SynthesizeRequest(BaseModel):
    """Optional body for the synthesize endpoint — currently empty; future
    knobs (force-scope, hint-tags) can land here."""

    scope: str = Field(default="user")


class SynthesisResponse(BaseModel):
    name: str
    display_name: str
    description: str
    skill_prompt: str
    portal: str
    parameters: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    tags: list[str]
    confidence: float
    reasoning: str
    source_run_id: str


class CreateWorkflowRequest(BaseModel):
    name: str
    display_name: str
    description: str = ""
    skill_prompt: str = ""
    portal: str = ""
    parameters: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    scope: str = "user"
    source_run_id: str | None = None


class WorkflowResponse(BaseModel):
    workflow: dict[str, Any]


class WorkflowListResponse(BaseModel):
    workflows: list[dict[str, Any]]
    total: int


class UpdateWorkflowRequest(BaseModel):
    display_name: str | None = None
    description: str | None = None
    skill_prompt: str | None = None
    parameters: list[dict[str, Any]] | None = None
    actions: list[dict[str, Any]] | None = None
    tags: list[str] | None = None
    enabled: bool | None = None
    scope: str | None = None


class RejectRequest(BaseModel):
    reason: str = ""


class InvokeRequest(BaseModel):
    params: dict[str, Any] = Field(default_factory=dict)


class InvokeResponse(BaseModel):
    workflow_id: str
    workflow_name: str
    actions: list[dict[str, Any]]
    missing_params: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Authorization helpers
# ---------------------------------------------------------------------------


def _is_org_admin(ctx: TenantContext) -> bool:
    """Crude admin check until a real role system lands.

    Treats users whose ``role`` claim contains ``admin``/``owner``/``superadmin``
    as org admins. The default (no role) is non-admin. This is intentionally
    permissive in dev — production should switch to a proper RBAC check.
    """
    user = ctx.user
    if user is None:
        return False
    role = (getattr(user, "role", "") or "").lower()
    return any(token in role for token in ("admin", "owner", "superadmin"))


def _can_edit(workflow: dict[str, Any], ctx: TenantContext) -> bool:
    """Creator OR org admin may edit/delete a workflow."""
    user_id = str(getattr(ctx.user, "user_id", "") or "")
    if workflow.get("created_by_user_id") == user_id:
        return True
    return _is_org_admin(ctx)


# ---------------------------------------------------------------------------
# Synthesize from run
# ---------------------------------------------------------------------------


@router.post("/synthesize-from-run/{run_id}", response_model=SynthesisResponse)
async def synthesize_from_run(
    run_id: str,
    ctx: Tenant,
    body: SynthesizeRequest | None = None,
) -> SynthesisResponse:
    """Build a workflow proposal from a recorded Automation Lab run.

    Returns the synthesis to the caller for review/edit. The caller is
    expected to POST the (possibly edited) shape back to POST /v1/learned-workflows
    to actually persist.
    """
    from learned_workflows_marketplace.agents.automation.workflow_synthesizer import synthesize_workflow_from_run
    from learned_workflows_marketplace.ports import get_automation_test_service

    run_payload = await get_automation_test_service().get_run(ctx, run_id)
    if not run_payload:
        raise HTTPException(status_code=404, detail="run not found")
    events = run_payload.get("events") or []
    synthesis = await synthesize_workflow_from_run(run_payload, events)

    return SynthesisResponse(
        name=synthesis.name,
        display_name=synthesis.display_name,
        description=synthesis.description,
        skill_prompt=synthesis.skill_prompt,
        portal=synthesis.portal,
        parameters=[p.model_dump() for p in synthesis.parameters],
        actions=[a.model_dump() for a in synthesis.actions],
        tags=synthesis.tags,
        confidence=synthesis.confidence,
        reasoning=synthesis.reasoning,
        source_run_id=run_id,
    )


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


@router.post("", response_model=WorkflowResponse)
async def create_workflow(body: CreateWorkflowRequest, ctx: Tenant) -> WorkflowResponse:
    name = normalize_workflow_name(body.name)
    if not name:
        raise HTTPException(status_code=400, detail="name is required and must contain at least one letter or digit")
    if body.scope not in VALID_SCOPES:
        raise HTTPException(status_code=400, detail=f"invalid scope; expected one of {sorted(VALID_SCOPES)}")
    if not body.actions:
        raise HTTPException(status_code=400, detail="actions list cannot be empty")

    user_id = str(getattr(ctx.user, "user_id", "") or "")
    if not user_id:
        raise HTTPException(status_code=401, detail="user identity missing from token")

    # Org-scoped workflows go through approval. User-scoped go straight to approved.
    visibility = "pending_approval" if body.scope == "org" else "approved"

    workflow = LearnedWorkflow(
        id=str(uuid4()),
        organization_id=str(ctx.organization_id),
        created_by_user_id=user_id,
        name=name,
        display_name=body.display_name or name.replace("_", " ").title(),
        description=body.description,
        skill_prompt=body.skill_prompt,
        portal=body.portal,
        parameters=body.parameters,
        actions=body.actions,
        tags=[t for t in body.tags if t and isinstance(t, str)][:24],
        scope=body.scope,
        visibility=visibility,
        enabled=True,
        source_run_id=body.source_run_id,
    )
    repo = get_learned_workflow_repository()
    try:
        saved = await repo.create(workflow)
    except Exception as exc:  # noqa: BLE001
        log.warning("learned_workflow_create_exception", error=str(exc))
        raise HTTPException(status_code=500, detail="create failed") from exc

    # Side effects: Qdrant embed + Neo4j edges (best-effort, non-blocking).
    try:
        from learned_workflows_marketplace.infrastructure.workflows.qdrant_index import embed_learned_workflow
        await embed_learned_workflow(saved)
    except Exception as exc:  # noqa: BLE001
        log.warning("learned_workflow_qdrant_index_failed", workflow_id=saved.id, error=str(exc))
    try:
        from learned_workflows_marketplace.infrastructure.workflows.neo4j_graph import upsert_workflow_edges
        await upsert_workflow_edges(saved)
    except Exception as exc:  # noqa: BLE001
        log.warning("learned_workflow_neo4j_upsert_failed", workflow_id=saved.id, error=str(exc))

    fetched = await repo.get(saved.id, ctx.organization_id)
    return WorkflowResponse(workflow=fetched or {"id": saved.id, "name": saved.name})


@router.get("", response_model=WorkflowListResponse)
async def list_workflows(
    ctx: Tenant,
    scope: str | None = None,
    visibility: str | None = None,
    portal: str | None = None,
    enabled: bool | None = None,
) -> WorkflowListResponse:
    if scope and scope not in VALID_SCOPES:
        raise HTTPException(status_code=400, detail=f"invalid scope filter")
    if visibility and visibility not in VALID_VISIBILITIES:
        raise HTTPException(status_code=400, detail=f"invalid visibility filter")
    user_id_str = str(getattr(ctx.user, "user_id", "") or "")
    if not user_id_str:
        raise HTTPException(status_code=401, detail="user identity missing from token")
    rows = await get_learned_workflow_repository().list_for_org(
        ctx.organization_id,
        UUID(user_id_str),
        scope=scope,
        visibility=visibility,
        portal=portal,
        enabled=enabled,
    )
    return WorkflowListResponse(workflows=rows, total=len(rows))


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(workflow_id: str, ctx: Tenant) -> WorkflowResponse:
    row = await get_learned_workflow_repository().get(workflow_id, ctx.organization_id)
    if not row:
        raise HTTPException(status_code=404, detail="workflow not found")
    return WorkflowResponse(workflow=row)


@router.patch("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: str,
    body: UpdateWorkflowRequest,
    ctx: Tenant,
) -> WorkflowResponse:
    repo = get_learned_workflow_repository()
    existing = await repo.get(workflow_id, ctx.organization_id)
    if not existing:
        raise HTTPException(status_code=404, detail="workflow not found")
    if not _can_edit(existing, ctx):
        raise HTTPException(status_code=403, detail="only the creator or an org admin can edit this workflow")

    patch: dict[str, Any] = {}
    payload = body.model_dump(exclude_none=True)
    if "scope" in payload and payload["scope"] not in VALID_SCOPES:
        raise HTTPException(status_code=400, detail="invalid scope")
    # If scope is being promoted from user → org, force re-approval.
    if payload.get("scope") == "org" and existing.get("scope") != "org":
        patch["visibility"] = "pending_approval"
    for key in ("display_name", "description", "skill_prompt", "parameters", "actions", "tags", "enabled", "scope"):
        if key in payload:
            patch[key] = payload[key]
    updated = await repo.update(workflow_id, ctx.organization_id, patch)
    if not updated:
        raise HTTPException(status_code=500, detail="update failed")
    # Re-embed / re-graph if any indexable field changed.
    if any(k in patch for k in ("display_name", "description", "skill_prompt", "tags", "portal")):
        try:
            from learned_workflows_marketplace.infrastructure.workflows.qdrant_index import embed_learned_workflow_dict
            await embed_learned_workflow_dict(updated)
        except Exception as exc:  # noqa: BLE001
            log.warning("learned_workflow_qdrant_reindex_failed", workflow_id=workflow_id, error=str(exc))
        try:
            from learned_workflows_marketplace.infrastructure.workflows.neo4j_graph import upsert_workflow_edges_dict
            await upsert_workflow_edges_dict(updated)
        except Exception as exc:  # noqa: BLE001
            log.warning("learned_workflow_neo4j_reupsert_failed", workflow_id=workflow_id, error=str(exc))
    return WorkflowResponse(workflow=updated)


@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: str, ctx: Tenant) -> dict[str, Any]:
    repo = get_learned_workflow_repository()
    existing = await repo.get(workflow_id, ctx.organization_id)
    if not existing:
        raise HTTPException(status_code=404, detail="workflow not found")
    if not _can_edit(existing, ctx):
        raise HTTPException(status_code=403, detail="only the creator or an org admin can delete this workflow")
    ok = await repo.soft_delete(workflow_id, ctx.organization_id)
    if not ok:
        raise HTTPException(status_code=500, detail="delete failed")
    try:
        from learned_workflows_marketplace.infrastructure.workflows.qdrant_index import remove_learned_workflow
        await remove_learned_workflow(workflow_id, ctx.organization_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("learned_workflow_qdrant_remove_failed", workflow_id=workflow_id, error=str(exc))
    return {"ok": True, "deleted": workflow_id}


@router.post("/{workflow_id}/enable", response_model=WorkflowResponse)
async def enable_workflow(workflow_id: str, ctx: Tenant) -> WorkflowResponse:
    return await _toggle_enabled(workflow_id, ctx, enabled=True)


@router.post("/{workflow_id}/disable", response_model=WorkflowResponse)
async def disable_workflow(workflow_id: str, ctx: Tenant) -> WorkflowResponse:
    return await _toggle_enabled(workflow_id, ctx, enabled=False)


async def _toggle_enabled(workflow_id: str, ctx: TenantContext, *, enabled: bool) -> WorkflowResponse:
    repo = get_learned_workflow_repository()
    existing = await repo.get(workflow_id, ctx.organization_id)
    if not existing:
        raise HTTPException(status_code=404, detail="workflow not found")
    if not _can_edit(existing, ctx):
        raise HTTPException(status_code=403, detail="only the creator or an org admin can toggle this workflow")
    updated = await repo.update(workflow_id, ctx.organization_id, {"enabled": enabled})
    if not updated:
        raise HTTPException(status_code=500, detail="update failed")
    return WorkflowResponse(workflow=updated)


# ---------------------------------------------------------------------------
# Approval workflow (org admins only)
# ---------------------------------------------------------------------------


@router.post("/{workflow_id}/approve", response_model=WorkflowResponse)
async def approve_workflow(workflow_id: str, ctx: Tenant) -> WorkflowResponse:
    if not _is_org_admin(ctx):
        raise HTTPException(status_code=403, detail="only org admins can approve workflows")
    repo = get_learned_workflow_repository()
    existing = await repo.get(workflow_id, ctx.organization_id)
    if not existing:
        raise HTTPException(status_code=404, detail="workflow not found")
    from datetime import UTC, datetime
    updated = await repo.update(workflow_id, ctx.organization_id, {
        "visibility": "approved",
        "approved_by_user_id": str(getattr(ctx.user, "user_id", "") or ""),
        "approved_at": datetime.now(UTC).isoformat(),
        "rejection_reason": "",
    })
    return WorkflowResponse(workflow=updated or existing)


@router.post("/{workflow_id}/reject", response_model=WorkflowResponse)
async def reject_workflow(workflow_id: str, body: RejectRequest, ctx: Tenant) -> WorkflowResponse:
    if not _is_org_admin(ctx):
        raise HTTPException(status_code=403, detail="only org admins can reject workflows")
    repo = get_learned_workflow_repository()
    existing = await repo.get(workflow_id, ctx.organization_id)
    if not existing:
        raise HTTPException(status_code=404, detail="workflow not found")
    updated = await repo.update(workflow_id, ctx.organization_id, {
        "visibility": "rejected",
        "rejection_reason": (body.reason or "")[:2000],
    })
    return WorkflowResponse(workflow=updated or existing)


# ---------------------------------------------------------------------------
# Invocation — expand params into a concrete action list
# ---------------------------------------------------------------------------


_PARAM_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


def _substitute_params(value: Any, params: dict[str, Any]) -> tuple[Any, set[str]]:
    """Return ``(substituted_value, missing_param_names)``."""
    if not isinstance(value, str):
        return value, set()
    missing: set[str] = set()

    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in params:
            return str(params[key])
        missing.add(key)
        return match.group(0)

    return _PARAM_RE.sub(_sub, value), missing


@router.post("/{workflow_id}/invoke", response_model=InvokeResponse)
async def invoke_workflow(workflow_id: str, body: InvokeRequest, ctx: Tenant) -> InvokeResponse:
    """Expand a workflow's stored actions with runtime parameters and
    return the concrete list. The extension/agent then executes them.
    """
    repo = get_learned_workflow_repository()
    workflow = await repo.get(workflow_id, ctx.organization_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="workflow not found")
    if not workflow.get("enabled"):
        raise HTTPException(status_code=409, detail="workflow is disabled")
    if workflow.get("visibility") not in ("approved",):
        if workflow.get("scope") == "user":
            # User-scoped workflows are always approved on create — defensive check.
            pass
        else:
            raise HTTPException(status_code=409, detail=f"workflow is {workflow.get('visibility')}; cannot invoke")

    params = dict(body.params or {})

    # Validate required parameters
    missing_required: list[str] = []
    for p in workflow.get("parameters", []) or []:
        if not isinstance(p, dict):
            continue
        if p.get("required", True) and p.get("name") and p["name"] not in params:
            missing_required.append(p["name"])

    actions = workflow.get("actions") or []
    expanded: list[dict[str, Any]] = []
    missing_in_body: set[str] = set()
    for action in actions:
        if not isinstance(action, dict):
            continue
        new_target, m1 = _substitute_params(action.get("target", ""), params)
        new_value, m2 = _substitute_params(action.get("value", ""), params)
        missing_in_body |= m1 | m2
        expanded.append({
            "action_type": action.get("action_type"),
            "target": new_target,
            "value": new_value,
            "reasoning": action.get("reasoning", ""),
            "frame_url": action.get("frame_url", ""),
        })

    missing = sorted(set(missing_required) | missing_in_body)

    user_id_str = str(getattr(ctx.user, "user_id", "") or "")
    user_uuid = UUID(user_id_str) if user_id_str else UUID("00000000-0000-0000-0000-000000000000")
    try:
        await repo.log_invocation(
            ctx.organization_id,
            workflow_id,
            user_uuid,
            params=params,
            success=not bool(missing),
            actions_executed=0 if missing else len(expanded),
            error="missing parameters: " + ", ".join(missing) if missing else "",
            metadata={"source": "api_invoke"},
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("learned_workflow_invoke_log_failed", workflow_id=workflow_id, error=str(exc))

    # Write to episodic memory so the planner remembers next time
    # that THIS user successfully ran THIS workflow for similar pages
    # / similar instructions. Best-effort — mem0 outage must not block.
    if user_id_str and not missing:
        try:
            from agentic_browser_lab.memory.memory import get_automation_memory
            mem = get_automation_memory()
            await mem.remember_workflow_usage(
                org_id=ctx.organization_id,
                user_id=user_id_str,
                workflow_id=workflow_id,
                workflow_name=workflow.get("name") or "",
                page_url="",  # invoke endpoint doesn't know the page; populated by caller in metadata
                param_names=list(params.keys()),
                outcome="invoked",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("automation_memory_workflow_usage_skipped", error=str(exc))

    return InvokeResponse(
        workflow_id=workflow_id,
        workflow_name=workflow.get("name") or "",
        actions=expanded,
        missing_params=missing,
    )


class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    score_threshold: float = 0.55


class SearchResponse(BaseModel):
    matches: list[dict[str, Any]]


# ─────────────────────────────────────────────────────────────────────
# Marketplace — cross-org publishing + install (Item 5)
# ─────────────────────────────────────────────────────────────────────


class MarketplacePublishResponse(BaseModel):
    workflow_id: str
    marketplace_published: bool


@router.post("/{workflow_id}/marketplace/publish", response_model=MarketplacePublishResponse)
async def publish_to_marketplace(workflow_id: str, ctx: Tenant) -> MarketplacePublishResponse:
    """Opt-in: publish an APPROVED workflow to the cross-org marketplace.
    The row stays in this org; other orgs see a read-only listing they
    can install (fork) into their own catalog."""
    repo = get_learned_workflow_repository()
    workflow = await repo.get(workflow_id, ctx.organization_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="workflow not found")
    if workflow.get("visibility") != "approved":
        raise HTTPException(status_code=409, detail="only approved workflows can be published")
    if not workflow.get("enabled"):
        raise HTTPException(status_code=409, detail="workflow is disabled")
    updated = await repo.set_marketplace_published(workflow_id, ctx.organization_id, True)
    return MarketplacePublishResponse(
        workflow_id=workflow_id,
        marketplace_published=bool(updated and updated.get("marketplace_published")),
    )


@router.post("/{workflow_id}/marketplace/unpublish", response_model=MarketplacePublishResponse)
async def unpublish_from_marketplace(workflow_id: str, ctx: Tenant) -> MarketplacePublishResponse:
    """Hide this workflow from the marketplace. Existing installs in
    other orgs are unaffected — they own their fork."""
    repo = get_learned_workflow_repository()
    workflow = await repo.get(workflow_id, ctx.organization_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="workflow not found")
    updated = await repo.set_marketplace_published(workflow_id, ctx.organization_id, False)
    return MarketplacePublishResponse(
        workflow_id=workflow_id,
        marketplace_published=bool(updated and updated.get("marketplace_published")),
    )


class MarketplaceListingItem(BaseModel):
    id: str
    name: str
    display_name: str
    description: str
    portal: str = ""
    tags: list[str] = []
    parameters: list[dict[str, Any]] = []
    action_count: int = 0
    marketplace_install_count: int = 0
    source_organization_id: str = ""


class MarketplaceListResponse(BaseModel):
    listings: list[MarketplaceListingItem]


@router.get("/marketplace/list", response_model=MarketplaceListResponse)
async def list_marketplace(
    ctx: Tenant,
    portal: str | None = None,
    search: str | None = None,
    limit: int = 50,
) -> MarketplaceListResponse:
    """Cross-org read of every workflow flagged marketplace_published=true.
    Caller's org_id is irrelevant for visibility here — that's the point."""
    repo = get_learned_workflow_repository()
    rows = await repo.list_marketplace(portal=portal, search=search, limit=limit)
    listings = [
        MarketplaceListingItem(
            id=str(r.get("id")),
            name=r.get("name") or "",
            display_name=r.get("display_name") or r.get("name") or "",
            description=(r.get("description") or "")[:480],
            portal=r.get("portal") or "",
            tags=list(r.get("tags") or []),
            parameters=list(r.get("parameters") or []),
            action_count=len(r.get("actions") or []),
            marketplace_install_count=int(r.get("marketplace_install_count") or 0),
            source_organization_id=str(r.get("organization_id") or ""),
        )
        for r in rows
    ]
    return MarketplaceListResponse(listings=listings)


class MarketplaceInstallResponse(BaseModel):
    installed_workflow_id: str
    source_workflow_id: str
    name: str
    display_name: str


@router.post("/marketplace/install/{source_workflow_id}", response_model=MarketplaceInstallResponse)
async def install_from_marketplace(source_workflow_id: str, ctx: Tenant) -> MarketplaceInstallResponse:
    """Fork a marketplace-published workflow into the caller's org.
    The new row gets a fresh id, new organization_id (the caller's),
    new created_by_user_id, scope=org, visibility=approved, and a
    pointer back to the source via marketplace_source_workflow_id."""
    repo = get_learned_workflow_repository()
    user_id_str = str(getattr(ctx.user, "user_id", "") or "")
    if not user_id_str:
        raise HTTPException(status_code=401, detail="user not resolved")
    installed = await repo.install_from_marketplace(
        source_workflow_id=source_workflow_id,
        target_organization_id=ctx.organization_id,
        target_user_id=UUID(user_id_str),
    )
    if not installed:
        raise HTTPException(status_code=404, detail="marketplace workflow not found or not published")
    # Best-effort: index the new copy in Qdrant for this org so search works.
    try:
        from learned_workflows_marketplace.infrastructure.workflows.qdrant_index import embed_learned_workflow_dict
        await embed_learned_workflow_dict(installed)
    except Exception as exc:  # noqa: BLE001
        log.warning("marketplace_install_qdrant_index_skipped", error=str(exc))
    return MarketplaceInstallResponse(
        installed_workflow_id=installed.get("id"),
        source_workflow_id=source_workflow_id,
        name=installed.get("name") or "",
        display_name=installed.get("display_name") or "",
    )


# ─────────────────────────────────────────────────────────────────────
# Versioning + Stats (Item 2)
# ─────────────────────────────────────────────────────────────────────


class VersionListResponse(BaseModel):
    workflow_id: str
    versions: list[dict[str, Any]]


@router.get("/{workflow_id}/versions", response_model=VersionListResponse)
async def list_workflow_versions(workflow_id: str, ctx: Tenant) -> VersionListResponse:
    """Return every prior snapshot of this workflow (newest first).
    Each entry is a full row from learned_workflow_versions: previous
    name/display_name/description/skill/parameters/actions/tags +
    change_summary + created_at + created_by_user_id."""
    repo = get_learned_workflow_repository()
    # Visibility check: caller must be able to see the workflow.
    workflow = await repo.get(workflow_id, ctx.organization_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="workflow not found")
    versions = await repo.list_versions(workflow_id, ctx.organization_id)
    return VersionListResponse(workflow_id=workflow_id, versions=versions or [])


class StatsResponse(BaseModel):
    workflow_id: str
    stats: dict[str, Any] | None


@router.get("/{workflow_id}/stats", response_model=StatsResponse)
async def get_workflow_stats(workflow_id: str, ctx: Tenant) -> StatsResponse:
    """use_count + success_rate + last_used_at + unique_users for one workflow."""
    repo = get_learned_workflow_repository()
    workflow = await repo.get(workflow_id, ctx.organization_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="workflow not found")
    stats = await repo.get_stats(workflow_id, ctx.organization_id)
    return StatsResponse(workflow_id=workflow_id, stats=stats)


class StatsListResponse(BaseModel):
    stats: list[dict[str, Any]]


@router.get("/stats", response_model=StatsListResponse)
async def list_org_workflow_stats(
    ctx: Tenant,
    scope: str | None = None,
    visibility: str | None = None,
) -> StatsListResponse:
    """All-workflow stats for the org, ordered by use_count desc.
    Useful for the portal admin dashboard."""
    repo = get_learned_workflow_repository()
    out = await repo.list_stats_for_org(ctx.organization_id, scope=scope, visibility=visibility)
    return StatsListResponse(stats=out)


@router.post("/{workflow_id}/rollback/{version_number}", response_model=WorkflowResponse)
async def rollback_workflow_to_version(
    workflow_id: str, version_number: int, ctx: Tenant,
) -> WorkflowResponse:
    """Revert the workflow to a previous snapshot. The current state
    gets archived as a new version first, then the chosen version is
    restored, bumping current_version."""
    repo = get_learned_workflow_repository()
    workflow = await repo.get(workflow_id, ctx.organization_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="workflow not found")
    versions = await repo.list_versions(workflow_id, ctx.organization_id)
    target = next((v for v in versions if int(v.get("version_number") or 0) == version_number), None)
    if not target:
        raise HTTPException(status_code=404, detail=f"version {version_number} not found")
    patch = {
        "name": target.get("name"),
        "display_name": target.get("display_name"),
        "description": target.get("description"),
        "skill_prompt": target.get("skill_prompt"),
        "parameters": target.get("parameters"),
        "actions": target.get("actions"),
        "tags": target.get("tags"),
        "portal": target.get("portal"),
        "_change_summary": f"rollback to v{version_number}",
    }
    updated = await repo.update(workflow_id, ctx.organization_id, patch)
    if not updated:
        raise HTTPException(status_code=502, detail="rollback failed")
    return _to_response(updated)


@router.post("/search", response_model=SearchResponse)
async def search_workflows(body: SearchRequest, ctx: Tenant) -> SearchResponse:
    """Semantic search via Qdrant. Returns ranked workflows for the given
    free-text query within the caller's org."""
    if not body.query.strip():
        return SearchResponse(matches=[])
    try:
        from learned_workflows_marketplace.infrastructure.workflows.qdrant_index import search_learned_workflows
    except Exception as exc:  # noqa: BLE001
        log.warning("learned_workflow_search_module_missing", error=str(exc))
        return SearchResponse(matches=[])
    matches = await search_learned_workflows(
        ctx.organization_id,
        body.query.strip(),
        limit=max(1, min(20, body.limit)),
        score_threshold=max(0.0, min(1.0, body.score_threshold)),
    )
    return SearchResponse(matches=matches)


__all__ = ["router"]
