"""TenantContext + dependency resolver for FastAPI routes.

Every marketplace endpoint scopes its writes/reads to ``ctx.organization_id``
to enforce multi-tenant isolation. The default resolver here reads the
org_id from request headers — host projects that already verify JWTs
should override the dependency via FastAPI's ``app.dependency_overrides``:

    from learned_workflows_marketplace.auth import get_tenant_context
    from my_app.auth import resolve_tenant_from_jwt

    app.dependency_overrides[get_tenant_context] = resolve_tenant_from_jwt
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from fastapi import Header, HTTPException


@dataclass(slots=True)
class CurrentUser:
    """Minimal authenticated-user descriptor.

    Host projects with richer User models (with email, role, etc.) can
    supply an instance with extra attributes — the marketplace only reads
    ``user_id`` and ``role``.
    """

    user_id: str = ""
    role: str = ""
    email: str = ""


@dataclass(slots=True)
class TenantContext:
    """Per-request tenant state.

    Marketplace endpoints read ``ctx.organization_id`` and ``ctx.user.user_id``.
    """

    organization_id: UUID
    user: CurrentUser = field(default_factory=CurrentUser)
    org_name: str = ""
    is_active: bool = True

    def require_active(self) -> None:
        if not self.is_active:
            raise HTTPException(status_code=402, detail="subscription inactive")


async def get_tenant_context(
    x_organization_id: str = Header(default="", alias="X-Organization-Id"),
    x_user_id: str = Header(default="", alias="X-User-Id"),
    x_user_role: str = Header(default="", alias="X-User-Role"),
) -> TenantContext:
    """Default tenant resolver — header-based.

    Reads ``X-Organization-Id``, ``X-User-Id``, ``X-User-Role`` from the
    request. Returns 401 if no org_id is present. Host projects with
    JWT auth should override via ``app.dependency_overrides``.
    """
    org_id_str = (x_organization_id or "").strip()
    if not org_id_str:
        raise HTTPException(status_code=401, detail="missing X-Organization-Id header")
    try:
        org_id = UUID(org_id_str)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="X-Organization-Id is not a valid UUID") from exc
    return TenantContext(
        organization_id=org_id,
        user=CurrentUser(user_id=(x_user_id or "").strip(), role=(x_user_role or "").strip()),
    )


__all__ = ["CurrentUser", "TenantContext", "get_tenant_context"]
