"""MCP server exposing saved workflows as stateless tools.

Two tools, mounted as a Starlette ASGI app at ``/mcp/v1/`` on the host
FastAPI application:

  ``list_workflows(portal=None, enabled_only=True)``
    Returns the calling tenant's saved workflow catalog as JSON rows.
    Use this before ``invoke_workflow`` to see what's available.

  ``invoke_workflow(workflow_name, params={})``
    Looks up the workflow by name in the catalog, substitutes
    ``{{param}}`` placeholders in the recorded actions, and returns
    the expanded action list. The CALLER is responsible for actually
    executing the actions (via a browser extension, Playwright, etc.).
    This server does NOT drive the browser.

The MCP tools are stateless: no run-time LLM calls, no browser, no
session. They're a queryable view onto the workflow catalog that any
MCP client (the Sales Coach autopilot, Claude Desktop, a third-party
agentic system) can consume.

Tenant binding
--------------
A Starlette middleware wraps every MCP request and resolves a
:class:`TenantContext` from the incoming headers before the tool
handler runs. Hosts with their own auth (e.g. JWT-based) register a
resolver via :func:`set_mcp_tenant_resolver` so MCP requests go
through the same verification path as the rest of the host's API.
"""

from __future__ import annotations

import contextvars
import re
from typing import Any, Awaitable, Callable
from uuid import UUID

import structlog
from fastapi import HTTPException
from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from learned_workflows_marketplace.auth import CurrentUser, TenantContext
from learned_workflows_marketplace.learned_workflows import (
    get_learned_workflow_repository,
)

log = structlog.get_logger(__name__)


# ── Tenant binding ──────────────────────────────────────────────────────

# Per-request tenant context. Set by TenantBindingMiddleware before the
# MCP server dispatches a call to a tool handler; read by the handlers
# via _require_tenant().
_current_tenant: contextvars.ContextVar[TenantContext | None] = contextvars.ContextVar(
    "lwm_mcp_tenant", default=None,
)

_tenant_resolver: Callable[[Request], Awaitable[TenantContext]] | None = None


def set_mcp_tenant_resolver(fn: Callable[[Request], Awaitable[TenantContext]]) -> None:
    """Register the host's request-to-tenant resolver for MCP requests.

    Default behaviour (no resolver registered) reads ``X-Organization-Id``,
    ``X-User-Id``, and ``X-User-Role`` headers directly. Hosts with JWT
    auth or richer tenant lookup register their own resolver so the
    MCP request flow matches the rest of the application's auth.
    """
    global _tenant_resolver
    _tenant_resolver = fn


async def _resolve_tenant_default(request: Request) -> TenantContext:
    org_id_str = (request.headers.get("X-Organization-Id") or "").strip()
    if not org_id_str:
        raise HTTPException(status_code=401, detail="missing X-Organization-Id header")
    try:
        org_id = UUID(org_id_str)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="X-Organization-Id is not a valid UUID") from exc
    user_id = (request.headers.get("X-User-Id") or "").strip()
    user_role = (request.headers.get("X-User-Role") or "").strip()
    return TenantContext(
        organization_id=org_id,
        user=CurrentUser(user_id=user_id, role=user_role),
    )


async def _resolve_tenant(request: Request) -> TenantContext:
    if _tenant_resolver is not None:
        return await _tenant_resolver(request)
    return await _resolve_tenant_default(request)


def _require_tenant() -> TenantContext:
    ctx = _current_tenant.get()
    if ctx is None:
        # Defensive — the middleware should have bound a tenant before
        # the tool handler ran. Reaching this branch means the host
        # mounted the raw FastMCP app without the middleware.
        raise RuntimeError(
            "no tenant bound on this MCP request — host must mount "
            "build_mcp_asgi_app() (which wraps the FastMCP app in "
            "TenantBindingMiddleware), not the bare FastMCP app.",
        )
    return ctx


class TenantBindingMiddleware(BaseHTTPMiddleware):
    """Resolve a TenantContext per MCP request and bind it into a
    ContextVar before downstream handlers run.

    On auth failure returns the HTTPException as a JSON response with
    the appropriate status code, never reaching the tool handlers.
    """

    async def dispatch(self, request: Request, call_next):
        try:
            ctx = await _resolve_tenant(request)
        except HTTPException as exc:
            return JSONResponse(
                {"error": {"code": exc.status_code, "message": exc.detail}},
                status_code=exc.status_code,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("mcp_tenant_resolve_failed", error=str(exc))
            return JSONResponse(
                {"error": {"code": 500, "message": "tenant resolution failed"}},
                status_code=500,
            )
        token = _current_tenant.set(ctx)
        try:
            return await call_next(request)
        finally:
            _current_tenant.reset(token)


# ── Parameter expansion ─────────────────────────────────────────────────

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


def _expand_placeholders(s: str, params: dict[str, str]) -> str:
    """Substitute ``{{param_name}}`` placeholders. Missing params left as-is."""
    if not s:
        return s
    return _PLACEHOLDER_RE.sub(
        lambda m: params.get(m.group(1), m.group(0)),
        s,
    )


# ── Tools ───────────────────────────────────────────────────────────────

mcp = FastMCP("learned-workflows")


@mcp.tool()
async def list_workflows(
    portal: str | None = None,
    enabled_only: bool = True,
) -> list[dict[str, Any]]:
    """List saved workflows in the current tenant's catalog.

    Use this before calling ``invoke_workflow`` to see what's available.
    Each returned row carries the workflow's machine name (for the
    invoke call), display name + description (for showing the user),
    portal hint, and the declared parameter list so the caller can
    prompt for / supply the right inputs.

    Args:
        portal: Optional filter by recorded portal short name
            (e.g. ``"sunfire"``, ``"uhc"``). When omitted, returns
            workflows across all portals.
        enabled_only: Skip disabled workflows (default ``True``).
            Set to ``False`` to include disabled rows for admin /
            inspection purposes.
    """
    ctx = _require_tenant()
    repo = get_learned_workflow_repository()
    # The repo signature requires a user_id even when listing org
    # scope. We pass the resolved user_id from the tenant context —
    # falls back to a zero UUID so org-wide visibility still works
    # when the caller didn't provide a user header.
    try:
        user_id = UUID(str(getattr(ctx.user, "user_id", "") or "")) if getattr(ctx.user, "user_id", "") else UUID(int=0)
    except (ValueError, TypeError):
        user_id = UUID(int=0)
    rows = await repo.list_for_org(
        organization_id=ctx.organization_id,
        user_id=user_id,
        scope=None,
        visibility="approved",
        portal=portal,
        enabled=True if enabled_only else None,
        include_user_scope=True,
    )
    out: list[dict[str, Any]] = []
    for w in rows:
        out.append({
            "name": w.get("name"),
            "display_name": w.get("display_name"),
            "description": w.get("description"),
            "portal": w.get("portal"),
            "parameters": w.get("parameters") or [],
            "tags": w.get("tags") or [],
            "scope": w.get("scope"),
            "use_count": w.get("use_count") or 0,
        })
    return out


@mcp.tool()
async def invoke_workflow(
    workflow_name: str,
    params: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Get the expanded action list for a saved workflow.

    Looks up ``workflow_name`` in the calling tenant's catalog,
    substitutes ``{{placeholder}}`` references in the recorded actions
    using the supplied ``params``, and returns the fully-expanded
    action list. The CALLER is responsible for actually executing the
    actions in a browser — this tool does NOT drive a browser.

    Returns a dict with::

        {
          "workflow_name": "customer_lookup",
          "workflow_display_name": "Customer Lookup",
          "workflow_id": "...",
          "portal": "sunfire",
          "actions": [
            {"action_type": "click", "target": "#consent", "value": "", ...},
            {"action_type": "type", "target": "#zip", "value": "90210", ...},
            ...
          ],
          "missing_params": ["dob"],   # required params the caller didn't supply
          "declared_parameters": ["zip", "medicare_number", "dob"],
        }

    If ``missing_params`` is non-empty the caller should re-invoke
    with the missing values rather than executing the actions —
    placeholders for missing params are left as ``{{name}}`` in the
    output, which would fail at execution time.

    Args:
        workflow_name: The workflow's snake_case machine name
            (from ``list_workflows``).
        params: Map of parameter name → value to substitute.
    """
    ctx = _require_tenant()
    params = params or {}
    repo = get_learned_workflow_repository()
    try:
        user_id = UUID(str(getattr(ctx.user, "user_id", "") or "")) if getattr(ctx.user, "user_id", "") else UUID(int=0)
    except (ValueError, TypeError):
        user_id = UUID(int=0)
    rows = await repo.list_for_org(
        organization_id=ctx.organization_id,
        user_id=user_id,
        scope=None,
        visibility="approved",
        portal=None,
        enabled=True,
        include_user_scope=True,
    )
    match = next((w for w in rows if w.get("name") == workflow_name), None)
    if match is None:
        raise ValueError(
            f"workflow '{workflow_name}' not found in this tenant's catalog. "
            f"Call list_workflows() to see available names."
        )

    declared = list(match.get("parameters") or [])
    declared_names = [p.get("name") for p in declared if p.get("name")]
    required_names = [
        p.get("name") for p in declared
        if p.get("name") and p.get("required") is not False
    ]
    missing = [n for n in required_names if not params.get(n)]

    expanded_actions: list[dict[str, Any]] = []
    for a in (match.get("actions") or []):
        expanded_actions.append({
            **a,
            "value": _expand_placeholders(str(a.get("value", "")), params),
            "target": _expand_placeholders(str(a.get("target", "")), params),
        })

    return {
        "workflow_name": match.get("name"),
        "workflow_display_name": match.get("display_name"),
        "workflow_id": match.get("id"),
        "portal": match.get("portal"),
        "actions": expanded_actions,
        "missing_params": missing,
        "declared_parameters": declared_names,
    }


def build_mcp_asgi_app():
    """Build the MCP ASGI app with tenant binding applied.

    The host mounts this at ``/mcp/v1/`` via :meth:`FastAPI.mount`.
    The wrapping middleware ensures every request resolves a
    :class:`TenantContext` before any tool handler runs.
    """
    app = mcp.streamable_http_app()
    app.add_middleware(TenantBindingMiddleware)
    return app


__all__ = [
    "build_mcp_asgi_app",
    "mcp",
    "set_mcp_tenant_resolver",
]
