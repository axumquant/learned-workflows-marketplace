"""Qdrant adapter for learned_workflows — embed + semantic search.

Each workflow is embedded once on create (and re-embedded on edits to
indexable fields). The vector is stored in a per-org Qdrant collection
(``coach_{org_uuid_hex}`` via :func:`app.infrastructure.qdrant.client.collection_for`)
with payload metadata so search results carry the workflow id, name,
portal, tags, and visibility.

The "indexable text" for a workflow is the concatenation of:

  - display_name
  - description
  - skill_prompt
  - tags
  - portal

That's what an autopilot agent's instruction will be semantically
compared against when ranking which tool to suggest.

All functions are best-effort: if Qdrant credentials are missing or
the cluster is unreachable, we log a warning and silently no-op.
"""

from __future__ import annotations

import uuid
from typing import Any
from uuid import UUID

import structlog

from learned_workflows_marketplace.learned_workflows import LearnedWorkflow

log = structlog.get_logger(__name__)


_WORKFLOW_VECTOR_NAMESPACE = uuid.UUID("3b6e5c8e-1d6d-4e92-9c41-7f1a2d3b4c50")


def _qdrant_point_id(workflow_id: str) -> str:
    """Deterministic UUID5 from the workflow id so re-embeds overwrite
    the same point instead of accumulating duplicates."""
    return str(uuid.uuid5(_WORKFLOW_VECTOR_NAMESPACE, str(workflow_id)))


def _indexable_text(payload: dict[str, Any]) -> str:
    parts = [
        str(payload.get("display_name") or ""),
        str(payload.get("description") or ""),
        str(payload.get("skill_prompt") or ""),
        " ".join(str(t) for t in (payload.get("tags") or []) if t),
        str(payload.get("portal") or ""),
    ]
    return " | ".join(p for p in parts if p).strip()


async def _embed(text: str) -> list[float] | None:
    if not text.strip():
        return None
    from learned_workflows_marketplace.ports import embed_query
    try:
        emb = await embed_query(text)
    except Exception as exc:  # noqa: BLE001
        log.warning("workflow_qdrant_embed_failed", error=str(exc))
        return None
    return emb or None


def _payload_for(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip the payload to indexable surface — never store the actions
    array (it can be large and isn't searched). The action_count is
    enough to filter "no-op" workflows."""
    return {
        "workflow_id": payload.get("id"),
        "name": payload.get("name"),
        "display_name": payload.get("display_name"),
        "description": (payload.get("description") or "")[:500],
        "portal": payload.get("portal"),
        "tags": list(payload.get("tags") or []),
        "scope": payload.get("scope"),
        "visibility": payload.get("visibility"),
        "enabled": bool(payload.get("enabled")),
        "action_count": len(payload.get("actions") or []),
        "param_count": len(payload.get("parameters") or []),
    }


async def embed_learned_workflow(workflow: LearnedWorkflow) -> bool:
    return await embed_learned_workflow_dict({
        "id": workflow.id,
        "name": workflow.name,
        "display_name": workflow.display_name,
        "description": workflow.description,
        "skill_prompt": workflow.skill_prompt,
        "portal": workflow.portal,
        "tags": workflow.tags,
        "scope": workflow.scope,
        "visibility": workflow.visibility,
        "enabled": workflow.enabled,
        "actions": workflow.actions,
        "parameters": workflow.parameters,
        "organization_id": workflow.organization_id,
    })


async def embed_learned_workflow_dict(payload: dict[str, Any]) -> bool:
    """Embed a workflow into the org's Qdrant collection.

    Returns True on success, False if anything went wrong. Never raises —
    Qdrant being down should never block the user from saving.
    """
    org_id_raw = payload.get("organization_id")
    workflow_id = payload.get("id")
    if not org_id_raw or not workflow_id:
        return False
    try:
        org_uuid = UUID(str(org_id_raw))
    except Exception as exc:  # noqa: BLE001
        log.warning("workflow_qdrant_bad_org_id", org_id=str(org_id_raw), error=str(exc))
        return False

    text = _indexable_text(payload)
    embedding = await _embed(text)
    if not embedding:
        log.debug("workflow_qdrant_skip_no_embedding", workflow_id=workflow_id)
        return False

    try:
        from learned_workflows_marketplace.infrastructure.qdrant_client import ensure_collection, upsert_documents
    except Exception as exc:  # noqa: BLE001
        log.warning("workflow_qdrant_client_unavailable", error=str(exc))
        return False

    try:
        await ensure_collection(org_uuid)
    except Exception as exc:  # noqa: BLE001
        log.warning("workflow_qdrant_ensure_collection_failed", error=str(exc))
        return False

    point_id = _qdrant_point_id(str(workflow_id))
    payload_for_store = _payload_for(payload)
    try:
        await upsert_documents(org_uuid, [(point_id, embedding, payload_for_store)])
    except Exception as exc:  # noqa: BLE001
        log.warning("workflow_qdrant_upsert_failed", workflow_id=workflow_id, error=str(exc))
        return False
    log.info("workflow_qdrant_indexed", workflow_id=workflow_id, point_id=point_id)
    return True


async def remove_learned_workflow(workflow_id: str, organization_id: UUID) -> bool:
    """Delete the Qdrant point for a workflow when it's soft-deleted."""
    try:
        from learned_workflows_marketplace.infrastructure.qdrant_client import collection_for, get_client
    except Exception as exc:  # noqa: BLE001
        log.warning("workflow_qdrant_remove_client_unavailable", error=str(exc))
        return False
    client = get_client()
    if client is None:
        return False
    point_id = _qdrant_point_id(str(workflow_id))
    try:
        from qdrant_client.http import models as qm  # type: ignore
        await client.delete(
            collection_name=collection_for(organization_id),
            points_selector=qm.PointIdsList(points=[point_id]),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("workflow_qdrant_delete_failed", workflow_id=workflow_id, error=str(exc))
        return False
    return True


async def search_learned_workflows(
    organization_id: UUID,
    query: str,
    *,
    limit: int = 5,
    score_threshold: float = 0.55,
) -> list[dict[str, Any]]:
    """Top-K nearest workflows in the org's collection for a free-text query."""
    if not query.strip():
        return []
    embedding = await _embed(query)
    if not embedding:
        return []
    try:
        from learned_workflows_marketplace.infrastructure.qdrant_client import search as qdrant_search
    except Exception as exc:  # noqa: BLE001
        log.warning("workflow_qdrant_search_module_missing", error=str(exc))
        return []
    try:
        points = await qdrant_search(
            organization_id,
            embedding,
            limit=limit,
            score_threshold=score_threshold,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("workflow_qdrant_search_failed", error=str(exc))
        return []
    out: list[dict[str, Any]] = []
    for p in points or []:
        pl = getattr(p, "payload", None) or {}
        if pl.get("enabled") is False:
            continue
        out.append({
            **pl,
            "score": float(getattr(p, "score", 0.0) or 0.0),
        })
    return out


__all__ = [
    "embed_learned_workflow",
    "embed_learned_workflow_dict",
    "remove_learned_workflow",
    "search_learned_workflows",
]
