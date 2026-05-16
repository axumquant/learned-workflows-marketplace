"""Thin async wrapper around qdrant-client used by the workflow index.

Reads connection details from :func:`learned_workflows_marketplace.config.get_settings`
so the host can swap settings without touching this module. The
collection name for each organization is ``{prefix}_{org_uuid_hex}`` —
default prefix ``coach`` (override via ``QDRANT_COLLECTION_PREFIX``).

All functions are best-effort: if Qdrant credentials are missing or
the cluster is unreachable, we log a warning and return None / False /
[] instead of raising. The marketplace endpoints handle Qdrant being
unavailable gracefully.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import structlog

from learned_workflows_marketplace.config import get_settings

log = structlog.get_logger(__name__)


_client_cache: Any = None
_VECTOR_SIZE = 1024  # qwen3-embed default; can be overridden by host


def collection_for(organization_id: UUID) -> str:
    """Return the per-org collection name."""
    prefix = getattr(get_settings(), "qdrant_collection_prefix", "coach") or "coach"
    return f"{prefix}_{organization_id.hex}"


def get_client() -> Any:
    """Return a cached AsyncQdrantClient or None when not configured."""
    global _client_cache
    if _client_cache is not None:
        return _client_cache
    settings = get_settings()
    host = (getattr(settings, "qdrant_host", "") or "").strip()
    if not host:
        return None
    try:
        from qdrant_client import AsyncQdrantClient  # type: ignore[import-not-found]
    except ImportError as exc:
        log.warning("qdrant_client_not_installed", error=str(exc))
        return None
    try:
        _client_cache = AsyncQdrantClient(
            host=host,
            port=int(getattr(settings, "qdrant_port", 6333) or 6333),
            api_key=(getattr(settings, "qdrant_api_key", "") or "") or None,
            https=bool(getattr(settings, "qdrant_https", True)),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("qdrant_client_connect_failed", error=str(exc))
        return None
    return _client_cache


async def ensure_collection(organization_id: UUID, vector_size: int = _VECTOR_SIZE) -> bool:
    """Create the org's collection if it doesn't exist. Idempotent."""
    client = get_client()
    if client is None:
        return False
    name = collection_for(organization_id)
    try:
        from qdrant_client.http import models as qm  # type: ignore[import-not-found]
        collections = await client.get_collections()
        if any(c.name == name for c in (collections.collections or [])):
            return True
        await client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
        )
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("qdrant_ensure_collection_failed", org=str(organization_id), error=str(exc))
        return False


async def upsert_documents(
    organization_id: UUID,
    points: list[tuple[str, list[float], dict[str, Any]]],
) -> bool:
    """Upsert ``[(point_id, vector, payload), ...]`` into the org's collection."""
    if not points:
        return False
    client = get_client()
    if client is None:
        return False
    try:
        from qdrant_client.http import models as qm  # type: ignore[import-not-found]
        await client.upsert(
            collection_name=collection_for(organization_id),
            points=[
                qm.PointStruct(id=pid, vector=vec, payload=payload)
                for pid, vec, payload in points
            ],
        )
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("qdrant_upsert_failed", org=str(organization_id), error=str(exc))
        return False


async def search(
    organization_id: UUID,
    query_vector: list[float],
    *,
    limit: int = 5,
    score_threshold: float = 0.55,
) -> list[Any]:
    """Top-K search. Returns Qdrant ScoredPoint objects (with .payload, .score)."""
    client = get_client()
    if client is None or not query_vector:
        return []
    try:
        result = await client.search(
            collection_name=collection_for(organization_id),
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )
        return list(result or [])
    except Exception as exc:  # noqa: BLE001
        log.warning("qdrant_search_failed", org=str(organization_id), error=str(exc))
        return []


__all__ = [
    "collection_for",
    "get_client",
    "ensure_collection",
    "upsert_documents",
    "search",
]
