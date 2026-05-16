"""Neo4j adapter for learned_workflows — relationship graph edges.

Each saved workflow becomes a ``Workflow`` node tied to:

  - the ``Organization`` (every node carries ``organization_id`` for
    tenant scoping)
  - the ``User`` who created it
  - the ``Portal`` it operates on (sunfire, enrollhere, etc.)
  - one ``Tag`` node per tag

Edges:

  (User)-[:CREATED]->(Workflow)
  (Workflow)-[:RUNS_ON]->(Portal)
  (Workflow)-[:HAS_TAG]->(Tag)
  (Workflow)-[:DERIVED_FROM]->(AutomationRun)   (if source_run_id present)

The graph powers later analytics queries — "what tools live on
SunFire", "which user has the most adopted tools", "which tags
co-occur most often" — without requiring a separate analytics db.

If Neo4j credentials are missing or the driver is unreachable the
function logs a warning and returns False. Never blocks a save.
"""

from __future__ import annotations

from typing import Any

import structlog

from app.config import get_settings
from app.learned_workflows import LearnedWorkflow

log = structlog.get_logger(__name__)


_driver = None
_driver_signature: tuple[str, str, str] | None = None


def _get_driver():
    """Lazy-init the global Neo4j async driver.

    Returns ``None`` when credentials are missing or the driver package
    can't be imported (e.g. neo4j not installed in this environment).
    """
    global _driver, _driver_signature
    settings = get_settings()
    uri = (settings.neo4j_uri or "").strip()
    user = (settings.neo4j_user or "neo4j").strip()
    password = (settings.neo4j_password or "").strip()
    if not uri or not password:
        return None
    signature = (uri, user, password)
    if _driver is not None and _driver_signature == signature:
        return _driver
    try:
        from neo4j import AsyncGraphDatabase  # type: ignore
    except Exception as exc:  # noqa: BLE001
        log.debug("workflow_neo4j_driver_unavailable", error=str(exc))
        return None
    try:
        _driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        _driver_signature = signature
    except Exception as exc:  # noqa: BLE001
        log.warning("workflow_neo4j_driver_init_failed", error=str(exc))
        return None
    return _driver


_UPSERT_CYPHER = """
MERGE (org:Organization {id: $org_id})
MERGE (u:User {id: $user_id})
SET u.organization_id = $org_id
MERGE (wf:Workflow {id: $workflow_id})
SET wf.organization_id = $org_id,
    wf.name = $name,
    wf.display_name = $display_name,
    wf.portal = $portal,
    wf.scope = $scope,
    wf.visibility = $visibility,
    wf.enabled = $enabled,
    wf.updated_at = timestamp()
MERGE (u)-[:CREATED]->(wf)
MERGE (wf)-[:OWNED_BY]->(org)
WITH wf
OPTIONAL MATCH (wf)-[r:HAS_TAG]->(:Tag) DELETE r
WITH wf
UNWIND $tags AS tag_name
  MERGE (t:Tag {name: tag_name})
  MERGE (wf)-[:HAS_TAG]->(t)
WITH wf
FOREACH (portal_name IN CASE WHEN $portal <> '' THEN [$portal] ELSE [] END |
  MERGE (p:Portal {name: portal_name})
  MERGE (wf)-[:RUNS_ON]->(p)
)
FOREACH (run_id IN CASE WHEN $source_run_id IS NULL THEN [] ELSE [$source_run_id] END |
  MERGE (r:AutomationRun {id: run_id})
  MERGE (wf)-[:DERIVED_FROM]->(r)
)
RETURN wf.id AS id
"""


async def upsert_workflow_edges(workflow: LearnedWorkflow) -> bool:
    return await upsert_workflow_edges_dict({
        "id": workflow.id,
        "organization_id": workflow.organization_id,
        "created_by_user_id": workflow.created_by_user_id,
        "name": workflow.name,
        "display_name": workflow.display_name,
        "portal": workflow.portal,
        "tags": workflow.tags,
        "scope": workflow.scope,
        "visibility": workflow.visibility,
        "enabled": workflow.enabled,
        "source_run_id": workflow.source_run_id,
    })


async def upsert_workflow_edges_dict(payload: dict[str, Any]) -> bool:
    """Write Workflow + edges to Neo4j. Idempotent — safe to call multiple times.

    Returns True on success, False on any failure. Never raises.
    """
    driver = _get_driver()
    if driver is None:
        return False
    workflow_id = payload.get("id")
    org_id = payload.get("organization_id")
    user_id = payload.get("created_by_user_id")
    if not workflow_id or not org_id or not user_id:
        return False
    tags = [str(t) for t in (payload.get("tags") or []) if t]
    params = {
        "workflow_id": str(workflow_id),
        "org_id": str(org_id),
        "user_id": str(user_id),
        "name": str(payload.get("name") or ""),
        "display_name": str(payload.get("display_name") or ""),
        "portal": str(payload.get("portal") or ""),
        "scope": str(payload.get("scope") or "user"),
        "visibility": str(payload.get("visibility") or "approved"),
        "enabled": bool(payload.get("enabled")),
        "tags": tags,
        "source_run_id": str(payload["source_run_id"]) if payload.get("source_run_id") else None,
    }
    try:
        await driver.execute_query(_UPSERT_CYPHER, params)
    except Exception as exc:  # noqa: BLE001
        log.warning("workflow_neo4j_upsert_failed", workflow_id=workflow_id, error=str(exc))
        return False
    log.info("workflow_neo4j_upserted", workflow_id=workflow_id, tags=len(tags), portal=params["portal"])
    return True


async def remove_workflow_node(workflow_id: str) -> bool:
    """Detach-delete a Workflow node and all its inbound/outbound edges."""
    driver = _get_driver()
    if driver is None:
        return False
    try:
        await driver.execute_query(
            "MATCH (wf:Workflow {id: $workflow_id}) DETACH DELETE wf",
            {"workflow_id": str(workflow_id)},
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("workflow_neo4j_delete_failed", workflow_id=workflow_id, error=str(exc))
        return False
    return True


__all__ = [
    "upsert_workflow_edges",
    "upsert_workflow_edges_dict",
    "remove_workflow_node",
]
