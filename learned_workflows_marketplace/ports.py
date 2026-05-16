"""Ports — host-injected integrations.

The marketplace depends on two pieces of host-side functionality that
cannot reasonably be defaulted:

  1. A way to load a recorded automation run's payload (used by
     ``POST /v1/learned-workflows/synthesize-from-run/{run_id}``).
  2. A function that returns a vector embedding for free text (used
     by Qdrant semantic search).

Both default to a null implementation. If the host doesn't register
one, the corresponding endpoint returns HTTP 503 with a clear message;
search returns an empty embedding (so search returns no results but
nothing crashes).

Host wiring example::

    from learned_workflows_marketplace.ports import (
        set_automation_test_service,
        set_embed_function,
    )

    set_automation_test_service(MyHostAutomationTestService())
    set_embed_function(my_host_embed_query)
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol

from fastapi import HTTPException


class AutomationTestServicePort(Protocol):
    """Loads recorded automation-run payloads for the synthesize endpoint."""

    async def get_run(self, ctx: Any, run_id: str) -> dict | None: ...


_automation_test_service: AutomationTestServicePort | None = None
_embed_fn: Callable[[str], Awaitable[list[float]]] | None = None


def set_automation_test_service(service: AutomationTestServicePort) -> None:
    """Register the host's automation-test-run loader.

    Must be called once at host startup if the host wants the
    ``synthesize-from-run`` endpoint to work.
    """
    global _automation_test_service
    _automation_test_service = service


def get_automation_test_service() -> AutomationTestServicePort:
    """Return the host-registered service or raise 503."""
    if _automation_test_service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "automation test service not configured — host must call "
                "learned_workflows_marketplace.ports.set_automation_test_service() "
                "at startup to enable POST /v1/learned-workflows/synthesize-from-run"
            ),
        )
    return _automation_test_service


def set_embed_function(fn: Callable[[str], Awaitable[list[float]]]) -> None:
    """Register the host's free-text → vector embedding callable.

    Called by the Qdrant adapter every time a workflow is embedded or
    a semantic search runs. If not registered, Qdrant indexing skips
    silently (no embedding → no upsert) and search returns [].
    """
    global _embed_fn
    _embed_fn = fn


async def embed_query(text: str) -> list[float]:
    """Embed free text using the host-registered function.

    Returns an empty list if no host function has been registered or
    the registered function returns nothing.
    """
    if _embed_fn is None or not text.strip():
        return []
    try:
        return await _embed_fn(text) or []
    except Exception:  # noqa: BLE001
        return []


__all__ = [
    "AutomationTestServicePort",
    "set_automation_test_service",
    "get_automation_test_service",
    "set_embed_function",
    "embed_query",
]
