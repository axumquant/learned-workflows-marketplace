"""MCP (Model Context Protocol) server exposing the workflow catalog.

Host integration::

    from fastapi import FastAPI
    from learned_workflows_marketplace.mcp import build_mcp_asgi_app, set_mcp_tenant_resolver

    app = FastAPI()
    # ... rest of host setup ...

    # Optional: route MCP auth through the host's own resolver
    async def my_resolver(request):
        # decode JWT etc. → return TenantContext
        ...
    set_mcp_tenant_resolver(my_resolver)

    # Mount the MCP server
    app.mount("/mcp/v1", build_mcp_asgi_app())
"""

from .server import (  # noqa: F401
    build_mcp_asgi_app,
    mcp,
    set_mcp_tenant_resolver,
)

__all__ = ["build_mcp_asgi_app", "mcp", "set_mcp_tenant_resolver"]
