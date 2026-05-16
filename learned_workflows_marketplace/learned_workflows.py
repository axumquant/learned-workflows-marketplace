"""Learned workflows — reusable browser-automation tools synthesized from
Automation Lab runs.

Pipeline:

  1. User runs an agentic instruction loop in the Automation Lab overlay.
     Every action (type / click / press_key / navigate) is logged as an
     ``automation_test_events`` row.
  2. User clicks "Save as Workflow" in the overlay.
  3. ``WorkflowSynthesizer`` reads the run + events, calls Ollama Cloud
     (deepseek-v4-pro) to generate a tool name, display_name, description,
     skill_prompt, parameter list, and tag set. Parameters are auto-flagged
     from typed values that match seed_data keys or look like user input.
  4. User reviews/edits in the modal, hits Save.
  5. ``LearnedWorkflowService.create_workflow`` persists to Postgres,
     embeds via Qdrant for semantic search, and writes Neo4j relationship
     edges (Workflow)-[:RUNS_ON]->(Portal), [:HAS_TAG]->(Tag), [:CREATED_BY]->(User).
  6. The workflow appears in:
       - Extension Settings → Custom MCPs catalog (search, enable, invoke)
       - Portal → Workflows page (full management + org approval)
       - The autopilot agent's available toolset

Sharing scopes:
  - ``user``: visible only to the creator. Always ``approved``.
  - ``org``: visible to everyone in the org once an admin approves it.
    Starts at ``pending_approval`` until an admin calls ``/approve``.

All persistence is org-scoped via RLS — see migration 027.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import httpx
import structlog

from learned_workflows_marketplace.config import get_settings

log = structlog.get_logger(__name__)

VALID_SCOPES = {"user", "org"}
VALID_VISIBILITIES = {"private", "pending_approval", "approved", "rejected"}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WorkflowParameter:
    """One runtime parameter for a learned workflow.

    e.g. ``{"name": "zip", "type": "string", "pattern": r"^\\d{5}$",
            "description": "5-digit ZIP code", "required": true}``
    """

    name: str
    type: str = "string"  # string | number | boolean
    description: str = ""
    pattern: str = ""  # optional regex for validation
    required: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WorkflowAction:
    """One step of a learned workflow.

    ``value`` may contain ``{{param_name}}`` placeholders that get
    substituted from runtime parameters at invocation time.
    """

    action_type: str  # click | type | press_key | navigate | wait
    target: str = ""  # CSS selector OR "x,y" coords
    value: str = ""
    reasoning: str = ""
    frame_url: str = ""  # if action targets a subframe

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LearnedWorkflow:
    id: str
    organization_id: str
    created_by_user_id: str
    name: str  # snake_case, unique per org
    display_name: str
    description: str
    skill_prompt: str
    portal: str
    parameters: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    scope: str = "user"
    visibility: str = "approved"
    approved_by_user_id: str | None = None
    approved_at: str | None = None
    rejection_reason: str = ""
    enabled: bool = True
    source_run_id: str | None = None
    use_count: int = 0
    last_used_at: str | None = None
    qdrant_point_id: str = ""
    created_at: str = ""
    updated_at: str = ""


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class LearnedWorkflowRepository(ABC):
    @abstractmethod
    async def create(self, workflow: LearnedWorkflow) -> LearnedWorkflow:
        raise NotImplementedError

    @abstractmethod
    async def get(self, workflow_id: str, organization_id: UUID) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    async def list_for_org(
        self,
        organization_id: UUID,
        user_id: UUID,
        *,
        scope: str | None = None,
        visibility: str | None = None,
        portal: str | None = None,
        enabled: bool | None = None,
        include_user_scope: bool = True,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def update(
        self,
        workflow_id: str,
        organization_id: UUID,
        patch: dict[str, Any],
    ) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    async def soft_delete(self, workflow_id: str, organization_id: UUID) -> bool:
        raise NotImplementedError

    # Marketplace — publish/unpublish a workflow, list cross-org
    # published workflows, install a fork into another org. Defaults
    # are no-ops so the in-memory tests keep working.
    async def set_marketplace_published(
        self, workflow_id: str, organization_id: UUID, published: bool,
    ) -> dict[str, Any] | None:
        return await self.update(
            workflow_id, organization_id,
            {"marketplace_published": published},
        )

    async def list_marketplace(
        self, *, portal: str | None = None, search: str | None = None, limit: int = 50,
    ) -> list[dict[str, Any]]:
        return []

    async def install_from_marketplace(
        self,
        *,
        source_workflow_id: str,
        target_organization_id: UUID,
        target_user_id: UUID,
    ) -> dict[str, Any] | None:
        return None

    # Versioning + stats — both default to no-op on the in-memory
    # implementation so callers can use them unconditionally.
    async def list_versions(
        self, workflow_id: str, organization_id: UUID,
    ) -> list[dict[str, Any]]:
        return []

    async def get_stats(
        self, workflow_id: str, organization_id: UUID,
    ) -> dict[str, Any] | None:
        return None

    async def list_stats_for_org(
        self,
        organization_id: UUID,
        *,
        scope: str | None = None,
        visibility: str | None = None,
    ) -> list[dict[str, Any]]:
        return []

    @abstractmethod
    async def log_invocation(
        self,
        organization_id: UUID,
        workflow_id: str,
        invoked_by_user_id: UUID,
        params: dict[str, Any],
        success: bool,
        actions_executed: int,
        error: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError


class InMemoryLearnedWorkflowRepository(LearnedWorkflowRepository):
    """In-memory fallback for tests and offline dev."""

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}
        self._invocations: list[dict[str, Any]] = []
        self._versions: list[dict[str, Any]] = []  # version snapshots (item-2)

    async def create(self, workflow: LearnedWorkflow) -> LearnedWorkflow:
        row = asdict(workflow)
        row.setdefault("created_at", _now_iso())
        row.setdefault("updated_at", row["created_at"])
        row["deleted_at"] = None
        self._rows[workflow.id] = row
        return workflow

    async def get(self, workflow_id: str, organization_id: UUID) -> dict[str, Any] | None:
        row = self._rows.get(workflow_id)
        if not row or row.get("organization_id") != str(organization_id) or row.get("deleted_at"):
            return None
        return dict(row)

    async def list_for_org(
        self,
        organization_id: UUID,
        user_id: UUID,
        *,
        scope: str | None = None,
        visibility: str | None = None,
        portal: str | None = None,
        enabled: bool | None = None,
        include_user_scope: bool = True,
    ) -> list[dict[str, Any]]:
        org = str(organization_id)
        usr = str(user_id)
        out: list[dict[str, Any]] = []
        for row in self._rows.values():
            if row.get("deleted_at"):
                continue
            if row.get("organization_id") != org:
                continue
            row_scope = row.get("scope")
            # User-scoped rows are only visible to their creator.
            if row_scope == "user" and row.get("created_by_user_id") != usr:
                continue
            if scope and row_scope != scope:
                continue
            if visibility and row.get("visibility") != visibility:
                continue
            if portal and row.get("portal") != portal:
                continue
            if enabled is not None and bool(row.get("enabled")) != enabled:
                continue
            if not include_user_scope and row_scope == "user":
                continue
            out.append(dict(row))
        out.sort(key=lambda r: r.get("updated_at") or "", reverse=True)
        return out

    def _snapshot(self, row: dict[str, Any], change_summary: str = "") -> None:
        """Append the row's current state into _versions before mutating it."""
        current_version = int(row.get("current_version") or 1)
        self._versions.append({
            "id": str(uuid4()),
            "workflow_id": row.get("id"),
            "organization_id": row.get("organization_id"),
            "version_number": current_version,
            "name": row.get("name") or "",
            "display_name": row.get("display_name") or "",
            "description": row.get("description") or "",
            "skill_prompt": row.get("skill_prompt") or "",
            "parameters": list(row.get("parameters") or []),
            "actions": list(row.get("actions") or []),
            "tags": list(row.get("tags") or []),
            "portal": row.get("portal") or "",
            "scope": row.get("scope") or "user",
            "visibility": row.get("visibility") or "approved",
            "change_summary": change_summary[:240],
            "created_by_user_id": row.get("created_by_user_id"),
            "created_at": _now_iso(),
        })

    async def update(
        self,
        workflow_id: str,
        organization_id: UUID,
        patch: dict[str, Any],
    ) -> dict[str, Any] | None:
        row = self._rows.get(workflow_id)
        if not row or row.get("organization_id") != str(organization_id) or row.get("deleted_at"):
            return None
        # Snapshot the OLD state before applying the patch — every
        # update creates a new version row. Skip pure metadata bumps
        # like last_used_at to avoid noise.
        SUBSTANTIVE_KEYS = {
            "name", "display_name", "description", "skill_prompt",
            "parameters", "actions", "tags", "portal", "scope", "visibility",
        }
        if any(k in patch and patch[k] is not None for k in SUBSTANTIVE_KEYS):
            change_summary = (patch.pop("_change_summary", "") if isinstance(patch, dict) else "") or "edit"
            self._snapshot(row, change_summary)
            row["current_version"] = int(row.get("current_version") or 1) + 1
        row.update({k: v for k, v in patch.items() if v is not None})
        row["updated_at"] = _now_iso()
        return dict(row)

    async def list_versions(self, workflow_id: str, organization_id: UUID) -> list[dict[str, Any]]:
        return [
            dict(v) for v in self._versions
            if v.get("workflow_id") == workflow_id and v.get("organization_id") == str(organization_id)
        ]

    async def get_stats(self, workflow_id: str, organization_id: UUID) -> dict[str, Any] | None:
        row = self._rows.get(workflow_id)
        if not row or row.get("organization_id") != str(organization_id):
            return None
        invocations = [i for i in self._invocations if i.get("workflow_id") == workflow_id]
        use_count = len(invocations)
        success_count = sum(1 for i in invocations if i.get("success"))
        failure_count = use_count - success_count
        unique_users = len({i.get("invoked_by_user_id") for i in invocations if i.get("invoked_by_user_id")})
        success_rate = round(success_count / use_count, 4) if use_count else 0.0
        last_used_at = max((i.get("created_at") for i in invocations), default=None)
        return {
            "workflow_id": workflow_id,
            "organization_id": str(organization_id),
            "name": row.get("name"),
            "display_name": row.get("display_name"),
            "scope": row.get("scope"),
            "visibility": row.get("visibility"),
            "enabled": row.get("enabled"),
            "current_version": int(row.get("current_version") or 1),
            "use_count": use_count,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_rate,
            "last_used_at": last_used_at,
            "unique_users": unique_users,
        }

    async def list_stats_for_org(
        self,
        organization_id: UUID,
        *,
        scope: str | None = None,
        visibility: str | None = None,
    ) -> list[dict[str, Any]]:
        out = []
        for wf_id, row in self._rows.items():
            if row.get("organization_id") != str(organization_id):
                continue
            if row.get("deleted_at"):
                continue
            if scope and row.get("scope") != scope:
                continue
            if visibility and row.get("visibility") != visibility:
                continue
            stats = await self.get_stats(wf_id, organization_id)
            if stats:
                out.append(stats)
        return out

    async def list_marketplace(
        self, *, portal: str | None = None, search: str | None = None, limit: int = 50,
    ) -> list[dict[str, Any]]:
        out = []
        q = (search or "").strip().lower()
        for row in self._rows.values():
            if not row.get("marketplace_published"):
                continue
            if row.get("deleted_at") or not row.get("enabled"):
                continue
            if row.get("visibility") != "approved":
                continue
            if portal and row.get("portal") != portal:
                continue
            if q:
                hay = " ".join([
                    row.get("name") or "", row.get("display_name") or "",
                    row.get("description") or "", " ".join(row.get("tags") or []),
                ]).lower()
                if q not in hay:
                    continue
            out.append(dict(row))
        return out[:limit]

    async def install_from_marketplace(
        self,
        *,
        source_workflow_id: str,
        target_organization_id: UUID,
        target_user_id: UUID,
    ) -> dict[str, Any] | None:
        src = self._rows.get(source_workflow_id)
        if not src or not src.get("marketplace_published") or src.get("deleted_at"):
            return None
        import copy
        EXCLUDE = {
            "id", "organization_id", "created_by_user_id",
            "marketplace_published", "marketplace_install_count",
            "marketplace_source_workflow_id",
            "use_count", "last_used_at", "current_version",
            "scope", "visibility", "created_at", "updated_at",
        }
        new_id = str(uuid4())
        # Deep-copy every cloned value so future edits on the installed
        # row never bleed into the publisher's row (the in-memory
        # implementation shares dict references across orgs without this).
        cloned = {k: copy.deepcopy(v) for k, v in src.items() if k not in EXCLUDE}
        new_row = {
            **cloned,
            "id": new_id,
            "organization_id": str(target_organization_id),
            "created_by_user_id": str(target_user_id),
            "marketplace_published": False,
            "marketplace_install_count": 0,
            "marketplace_source_workflow_id": source_workflow_id,
            "use_count": 0,
            "last_used_at": None,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "current_version": 1,
            "scope": "org",
            "visibility": "approved",
        }
        self._rows[new_id] = new_row
        # Bump install count on source
        src["marketplace_install_count"] = int(src.get("marketplace_install_count") or 0) + 1
        return dict(new_row)

    async def soft_delete(self, workflow_id: str, organization_id: UUID) -> bool:
        row = self._rows.get(workflow_id)
        if not row or row.get("organization_id") != str(organization_id) or row.get("deleted_at"):
            return False
        row["deleted_at"] = _now_iso()
        row["updated_at"] = row["deleted_at"]
        return True

    async def log_invocation(
        self,
        organization_id: UUID,
        workflow_id: str,
        invoked_by_user_id: UUID,
        params: dict[str, Any],
        success: bool,
        actions_executed: int,
        error: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._invocations.append({
            "id": str(uuid4()),
            "organization_id": str(organization_id),
            "workflow_id": workflow_id,
            "invoked_by_user_id": str(invoked_by_user_id),
            "params": params,
            "success": success,
            "actions_executed": actions_executed,
            "error": error,
            "metadata": metadata or {},
            "created_at": _now_iso(),
        })
        row = self._rows.get(workflow_id)
        if row:
            row["use_count"] = int(row.get("use_count") or 0) + 1
            row["last_used_at"] = _now_iso()


class SupabaseLearnedWorkflowRepository(LearnedWorkflowRepository):
    """REST-based Supabase repository. Falls back to in-memory on failure."""

    def __init__(self) -> None:
        self._memory = InMemoryLearnedWorkflowRepository()

    def _headers(self, prefer: str | None = None) -> dict[str, str]:
        settings = get_settings()
        h = {
            "apikey": settings.supabase_service_key,
            "Authorization": f"Bearer {settings.supabase_service_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if prefer:
            h["Prefer"] = prefer
        return h

    def _enabled(self) -> bool:
        s = get_settings()
        return bool(s.supabase_url and s.supabase_service_key)

    def _base(self) -> str:
        return f"{get_settings().supabase_url}/rest/v1/learned_workflows"

    async def create(self, workflow: LearnedWorkflow) -> LearnedWorkflow:
        if not self._enabled():
            return await self._memory.create(workflow)
        payload = {
            "id": workflow.id,
            "organization_id": workflow.organization_id,
            "created_by_user_id": workflow.created_by_user_id,
            "name": workflow.name,
            "display_name": workflow.display_name,
            "description": workflow.description,
            "skill_prompt": workflow.skill_prompt,
            "portal": workflow.portal,
            "parameters": workflow.parameters,
            "actions": workflow.actions,
            "tags": workflow.tags,
            "scope": workflow.scope,
            "visibility": workflow.visibility,
            "enabled": workflow.enabled,
            "source_run_id": workflow.source_run_id,
        }
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.post(self._base(), headers=self._headers("return=minimal"), json=payload)
        if r.status_code >= 400:
            log.warning("learned_workflow_create_failed", status=r.status_code, body=r.text[:200])
            return await self._memory.create(workflow)
        return workflow

    async def get(self, workflow_id: str, organization_id: UUID) -> dict[str, Any] | None:
        if not self._enabled():
            return await self._memory.get(workflow_id, organization_id)
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(
                self._base(),
                headers=self._headers(),
                params={
                    "id": f"eq.{workflow_id}",
                    "organization_id": f"eq.{organization_id}",
                    "deleted_at": "is.null",
                    "select": "*",
                    "limit": "1",
                },
            )
        if r.status_code >= 400:
            log.warning("learned_workflow_get_failed", status=r.status_code, body=r.text[:200])
            return None
        rows = r.json()
        return rows[0] if rows else None

    async def list_for_org(
        self,
        organization_id: UUID,
        user_id: UUID,
        *,
        scope: str | None = None,
        visibility: str | None = None,
        portal: str | None = None,
        enabled: bool | None = None,
        include_user_scope: bool = True,
    ) -> list[dict[str, Any]]:
        if not self._enabled():
            return await self._memory.list_for_org(
                organization_id, user_id, scope=scope, visibility=visibility,
                portal=portal, enabled=enabled, include_user_scope=include_user_scope,
            )
        params: dict[str, str] = {
            "organization_id": f"eq.{organization_id}",
            "deleted_at": "is.null",
            "select": "*",
            "order": "updated_at.desc",
        }
        if scope:
            params["scope"] = f"eq.{scope}"
        if visibility:
            params["visibility"] = f"eq.{visibility}"
        if portal:
            params["portal"] = f"eq.{portal}"
        if enabled is not None:
            params["enabled"] = f"eq.{str(enabled).lower()}"
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(self._base(), headers=self._headers(), params=params)
        if r.status_code >= 400:
            log.warning("learned_workflow_list_failed", status=r.status_code, body=r.text[:200])
            return []
        rows = r.json() or []
        usr = str(user_id)
        out = []
        for row in rows:
            row_scope = row.get("scope")
            if row_scope == "user" and row.get("created_by_user_id") != usr:
                continue
            if not include_user_scope and row_scope == "user":
                continue
            out.append(row)
        return out

    async def update(
        self,
        workflow_id: str,
        organization_id: UUID,
        patch: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self._enabled():
            return await self._memory.update(workflow_id, organization_id, patch)
        # If the patch touches substantive fields, snapshot the row
        # into learned_workflow_versions and bump current_version.
        SUBSTANTIVE_KEYS = {"name", "display_name", "description", "skill_prompt",
                            "parameters", "actions", "tags", "portal", "scope", "visibility"}
        should_snapshot = any(k in patch and patch[k] is not None for k in SUBSTANTIVE_KEYS)
        # Only consume `_change_summary` when we're actually going to
        # snapshot — avoids silently dropping it on non-snapshot updates.
        change_summary = ""
        if should_snapshot and isinstance(patch, dict) and "_change_summary" in patch:
            change_summary = str(patch.pop("_change_summary") or "")[:240]
        elif isinstance(patch, dict) and "_change_summary" in patch:
            # Non-snapshot update: pop so PostgREST doesn't error on unknown column,
            # but don't consume — log instead so callers know it was ignored.
            patch.pop("_change_summary", None)
        if should_snapshot:
            current = await self.get(workflow_id, organization_id)
            if current:
                version_payload = {
                    "workflow_id": current.get("id"),
                    "organization_id": str(organization_id),
                    "version_number": int(current.get("current_version") or 1),
                    "name": current.get("name") or "",
                    "display_name": current.get("display_name") or "",
                    "description": current.get("description") or "",
                    "skill_prompt": current.get("skill_prompt") or "",
                    "parameters": current.get("parameters") or [],
                    "actions": current.get("actions") or [],
                    "tags": current.get("tags") or [],
                    "portal": current.get("portal") or "",
                    "scope": current.get("scope") or "user",
                    "visibility": current.get("visibility") or "approved",
                    "change_summary": change_summary or "edit",
                    "created_by_user_id": current.get("created_by_user_id"),
                }
                try:
                    async with httpx.AsyncClient(timeout=8.0) as c:
                        await c.post(
                            f"{get_settings().supabase_url}/rest/v1/learned_workflow_versions",
                            headers=self._headers("return=minimal"),
                            json=version_payload,
                        )
                except Exception as exc:  # noqa: BLE001
                    log.warning("learned_workflow_version_snapshot_failed", error=str(exc))
            # Bump current_version on the patch
            new_version = int((current or {}).get("current_version") or 1) + 1
            patch["current_version"] = new_version
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.patch(
                self._base(),
                headers=self._headers("return=representation"),
                params={"id": f"eq.{workflow_id}", "organization_id": f"eq.{organization_id}"},
                json=patch,
            )
        if r.status_code >= 400:
            log.warning("learned_workflow_update_failed", status=r.status_code, body=r.text[:200])
            return None
        rows = r.json()
        return rows[0] if rows else None

    async def list_versions(
        self, workflow_id: str, organization_id: UUID,
    ) -> list[dict[str, Any]]:
        if not self._enabled():
            return await self._memory.list_versions(workflow_id, organization_id)
        params = {
            "workflow_id": f"eq.{workflow_id}",
            "organization_id": f"eq.{organization_id}",
            "order": "version_number.desc",
        }
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(
                f"{get_settings().supabase_url}/rest/v1/learned_workflow_versions",
                headers=self._headers(),
                params=params,
            )
        if r.status_code >= 400:
            log.warning("learned_workflow_versions_list_failed", status=r.status_code)
            return []
        return r.json() or []

    async def get_stats(
        self, workflow_id: str, organization_id: UUID,
    ) -> dict[str, Any] | None:
        if not self._enabled():
            return await self._memory.get_stats(workflow_id, organization_id)
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(
                f"{get_settings().supabase_url}/rest/v1/learned_workflow_stats",
                headers=self._headers(),
                params={
                    "workflow_id": f"eq.{workflow_id}",
                    "organization_id": f"eq.{organization_id}",
                },
            )
        if r.status_code >= 400:
            log.warning("learned_workflow_stats_get_failed", status=r.status_code)
            return None
        rows = r.json() or []
        return rows[0] if rows else None

    async def list_stats_for_org(
        self,
        organization_id: UUID,
        *,
        scope: str | None = None,
        visibility: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self._enabled():
            return await self._memory.list_stats_for_org(organization_id, scope=scope, visibility=visibility)
        params: dict[str, str] = {"organization_id": f"eq.{organization_id}", "order": "use_count.desc"}
        if scope:
            params["scope"] = f"eq.{scope}"
        if visibility:
            params["visibility"] = f"eq.{visibility}"
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(
                f"{get_settings().supabase_url}/rest/v1/learned_workflow_stats",
                headers=self._headers(),
                params=params,
            )
        if r.status_code >= 400:
            log.warning("learned_workflow_stats_list_failed", status=r.status_code)
            return []
        return r.json() or []

    async def list_marketplace(
        self, *, portal: str | None = None, search: str | None = None, limit: int = 50,
    ) -> list[dict[str, Any]]:
        if not self._enabled():
            return await self._memory.list_marketplace(portal=portal, search=search, limit=limit)
        params: dict[str, str] = {
            "marketplace_published": "eq.true",
            "enabled": "eq.true",
            "visibility": "eq.approved",
            "deleted_at": "is.null",
            "order": "marketplace_install_count.desc",
            "limit": str(min(max(int(limit or 50), 1), 200)),
        }
        if portal:
            params["portal"] = f"eq.{portal}"
        if search:
            # Postgres full-text via ilike across name/display_name/description
            params["or"] = (
                f"(name.ilike.*{search}*,display_name.ilike.*{search}*,description.ilike.*{search}*)"
            )
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(self._base(), headers=self._headers(), params=params)
        if r.status_code >= 400:
            log.warning("learned_workflow_marketplace_list_failed", status=r.status_code, body=r.text[:240])
            return []
        return r.json() or []

    async def install_from_marketplace(
        self,
        *,
        source_workflow_id: str,
        target_organization_id: UUID,
        target_user_id: UUID,
    ) -> dict[str, Any] | None:
        if not self._enabled():
            return await self._memory.install_from_marketplace(
                source_workflow_id=source_workflow_id,
                target_organization_id=target_organization_id,
                target_user_id=target_user_id,
            )
        # Fetch the source workflow (cross-org read — only allowed when
        # marketplace_published=true).
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(
                self._base(), headers=self._headers(),
                params={"id": f"eq.{source_workflow_id}", "marketplace_published": "eq.true"},
            )
        if r.status_code >= 400 or not (r.json() or []):
            log.warning("learned_workflow_marketplace_source_not_found",
                        source_id=source_workflow_id, status=r.status_code)
            return None
        src = r.json()[0]
        # Strip the source's identity fields; clone everything else.
        new_id = str(uuid4())
        new_row = {
            "id": new_id,
            "organization_id": str(target_organization_id),
            "created_by_user_id": str(target_user_id),
            "name": src.get("name") or "",
            "display_name": src.get("display_name") or "",
            "description": src.get("description") or "",
            "skill_prompt": src.get("skill_prompt") or "",
            "parameters": src.get("parameters") or [],
            "actions": src.get("actions") or [],
            "tags": src.get("tags") or [],
            "portal": src.get("portal") or "",
            "scope": "org",
            "visibility": "approved",  # installed clones land approved by default
            "enabled": True,
            "marketplace_published": False,
            "marketplace_source_workflow_id": source_workflow_id,
            "marketplace_install_count": 0,
            "current_version": 1,
        }
        async with httpx.AsyncClient(timeout=10.0) as c:
            ins = await c.post(self._base(), headers=self._headers("return=representation"), json=new_row)
            if ins.status_code >= 400:
                log.warning("learned_workflow_marketplace_install_failed",
                            status=ins.status_code, body=ins.text[:240])
                return None
            installed = ins.json()[0]
            # Bump install count on the source workflow.
            try:
                await c.patch(
                    self._base(), headers=self._headers("return=minimal"),
                    params={"id": f"eq.{source_workflow_id}"},
                    json={"marketplace_install_count": int(src.get("marketplace_install_count") or 0) + 1},
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("learned_workflow_marketplace_install_count_bump_failed", error=str(exc))
        return installed

    async def soft_delete(self, workflow_id: str, organization_id: UUID) -> bool:
        result = await self.update(workflow_id, organization_id, {"deleted_at": _now_iso()})
        return result is not None

    async def log_invocation(
        self,
        organization_id: UUID,
        workflow_id: str,
        invoked_by_user_id: UUID,
        params: dict[str, Any],
        success: bool,
        actions_executed: int,
        error: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self._enabled():
            return await self._memory.log_invocation(
                organization_id, workflow_id, invoked_by_user_id, params,
                success, actions_executed, error, metadata,
            )
        settings = get_settings()
        payload = {
            "organization_id": str(organization_id),
            "workflow_id": workflow_id,
            "invoked_by_user_id": str(invoked_by_user_id),
            "params": params,
            "success": success,
            "actions_executed": actions_executed,
            "error": error[:2000] if error else "",
            "metadata": metadata or {},
        }
        try:
            async with httpx.AsyncClient(timeout=8.0) as c:
                await c.post(
                    f"{settings.supabase_url}/rest/v1/learned_workflow_invocations",
                    headers=self._headers("return=minimal"),
                    json=payload,
                )
                # Bump use_count + last_used_at on the workflow row.
                await c.patch(
                    self._base(),
                    headers=self._headers("return=minimal"),
                    params={
                        "id": f"eq.{workflow_id}",
                        "organization_id": f"eq.{organization_id}",
                    },
                    json={"last_used_at": _now_iso()},
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("learned_workflow_invocation_log_failed", error=str(exc))


# ---------------------------------------------------------------------------
# Service singleton
# ---------------------------------------------------------------------------


_repo: LearnedWorkflowRepository | None = None


def get_learned_workflow_repository() -> LearnedWorkflowRepository:
    global _repo
    if _repo is None:
        s = get_settings()
        if s.supabase_url and s.supabase_service_key:
            _repo = SupabaseLearnedWorkflowRepository()
        else:
            _repo = InMemoryLearnedWorkflowRepository()
    return _repo


# ---------------------------------------------------------------------------
# Name normalization helper used by API + synthesizer
# ---------------------------------------------------------------------------

_SNAKE_RE = re.compile(r"[^a-z0-9_]+")


def normalize_workflow_name(value: str) -> str:
    """Normalize a candidate name to snake_case, machine-callable form."""
    if not value:
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"[^\w\s-]+", "", s)
    s = re.sub(r"[\s\-]+", "_", s)
    s = _SNAKE_RE.sub("", s)
    return s.strip("_")[:80]


def validate_scope(scope: str) -> str:
    if scope not in VALID_SCOPES:
        raise ValueError(f"invalid scope: {scope}")
    return scope


def validate_visibility(visibility: str) -> str:
    if visibility not in VALID_VISIBILITIES:
        raise ValueError(f"invalid visibility: {visibility}")
    return visibility


__all__ = [
    "LearnedWorkflow",
    "LearnedWorkflowRepository",
    "InMemoryLearnedWorkflowRepository",
    "SupabaseLearnedWorkflowRepository",
    "WorkflowAction",
    "WorkflowParameter",
    "get_learned_workflow_repository",
    "normalize_workflow_name",
    "validate_scope",
    "validate_visibility",
    "VALID_SCOPES",
    "VALID_VISIBILITIES",
]
