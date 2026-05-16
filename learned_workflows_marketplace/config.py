"""Marketplace runtime settings.

Read from environment variables so the package is pip-installable without
forcing every consumer to vendor a sales-coach-style settings module.

Host projects with their own settings object can replace the default by
calling :func:`set_settings_factory` at startup.

    from learned_workflows_marketplace.config import set_settings_factory
    set_settings_factory(lambda: my_host_settings_object)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable


@dataclass(slots=True)
class MarketplaceSettings:
    """Everything the marketplace needs to talk to its three stores.

    Field names match what the existing repo / Qdrant / Neo4j adapters
    already read from a settings object (``settings.supabase_url``,
    ``settings.neo4j_uri``, etc.) so swapping a host-supplied settings
    object is drop-in compatible.
    """

    # Postgres / Supabase
    supabase_url: str = ""
    supabase_service_key: str = ""
    supabase_anon_key: str = ""

    # Qdrant
    qdrant_host: str = ""
    qdrant_port: int = 6333
    qdrant_api_key: str = ""
    qdrant_https: bool = True
    qdrant_collection_prefix: str = "coach"

    # Neo4j
    neo4j_uri: str = ""
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"


def _settings_from_env() -> MarketplaceSettings:
    return MarketplaceSettings(
        supabase_url=os.environ.get("SUPABASE_URL", "").strip(),
        supabase_service_key=os.environ.get("SUPABASE_SERVICE_KEY", "").strip(),
        supabase_anon_key=os.environ.get("SUPABASE_ANON_KEY", "").strip(),
        qdrant_host=os.environ.get("QDRANT_HOST", "").strip(),
        qdrant_port=int(os.environ.get("QDRANT_PORT", "6333") or 6333),
        qdrant_api_key=os.environ.get("QDRANT_API_KEY", "").strip(),
        qdrant_https=os.environ.get("QDRANT_HTTPS", "true").lower() not in {"0", "false", "no"},
        qdrant_collection_prefix=os.environ.get("QDRANT_COLLECTION_PREFIX", "coach").strip() or "coach",
        neo4j_uri=os.environ.get("NEO4J_URI", "").strip(),
        neo4j_user=os.environ.get("NEO4J_USER", "neo4j").strip() or "neo4j",
        neo4j_password=os.environ.get("NEO4J_PASSWORD", "").strip(),
        neo4j_database=os.environ.get("NEO4J_DATABASE", "neo4j").strip() or "neo4j",
    )


_factory: Callable[[], object] = _settings_from_env


def set_settings_factory(factory: Callable[[], object]) -> None:
    """Replace the default env-based factory.

    Useful when the host project already has its own pydantic-settings
    object — pass a zero-arg callable returning it. As long as the
    returned object exposes the same attribute names as
    :class:`MarketplaceSettings`, every adapter in this package will
    keep working.
    """
    global _factory
    _factory = factory


def get_settings():
    """Return the active settings object (env-based by default)."""
    return _factory()


__all__ = [
    "MarketplaceSettings",
    "get_settings",
    "set_settings_factory",
]
