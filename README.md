# Learned Workflows Marketplace

> A workflow **recording, versioning, and cross-organization sharing platform** for browser-automation agents. Think "GitHub for AI workflows" — but with multi-tenant isolation, parameter substitution, semantic search, and an opt-in marketplace.

## What problem it solves

Browser automation agents (Browser-Use, Skyvern, Anthropic Computer Use, custom Pydantic AI teams) all share the same gap: **every user reinvents the same workflow**. If 50 agencies all need to look up Medicare customers on the SunFire portal, each agent figures it out from scratch — slow, expensive, brittle.

This system:
1. **Records** a workflow once when a user completes it (DOM events + user instruction).
2. **Templates** it — auto-parameterizes typed PII (`63664` → `{{zip}}`, `6uq8v57xd43` → `{{mbi}}`) so the saved workflow is reusable, not a recording with test data baked in.
3. **Stores** it across three layers — Postgres (truth), Qdrant (semantic search), Neo4j (relationships).
4. **Shares** it — opt-in `marketplace_published=true` exposes the workflow to all tenants, who can **install (fork)** it into their own catalog independently.
5. **Versions** every edit + tracks **stats** (use_count, success_rate, last_used, unique_users) so admins know what's working.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│   API layer (FastAPI) — 15 endpoints                         │
│   ────────────────────                                        │
│   POST   /synthesize-from-run/{run_id}                       │
│   POST,GET,PATCH,DELETE  /                                    │
│   POST   /{id}/{enable,disable,approve,reject,invoke}         │
│   POST   /search                          (semantic, Qdrant)  │
│   GET    /{id}/versions                   (snapshots)         │
│   GET    /{id}/stats                      (use_count, etc.)   │
│   GET    /stats?scope=&visibility=        (org-wide stats)    │
│   POST   /{id}/rollback/{version}                             │
│   POST   /{id}/marketplace/{publish,unpublish}                │
│   GET    /marketplace/list?portal=&search=                    │
│   POST   /marketplace/install/{source_workflow_id}            │
└──────────────────────────────────────────────────────────────┘
                                │
       ┌────────────────────────┼─────────────────────────┐
       ▼                        ▼                          ▼
┌────────────┐         ┌────────────┐            ┌────────────────┐
│  Postgres  │         │   Qdrant   │            │     Neo4j      │
│ (truth)    │         │ (search)   │            │ (relationships)│
│            │         │            │            │                │
│ workflows  │         │ embeddings │            │ Workflow ──>   │
│ versions   │         │ per org    │            │   Organization │
│ stats view │         │            │            │   User         │
│ invocations│         │            │            │   Portal       │
└────────────┘         └────────────┘            │   Tag          │
                                                  └────────────────┘
```

## Data model

```
learned_workflows
├── id                            (uuid)
├── organization_id               (uuid, RLS)
├── created_by_user_id            (uuid)
├── name                          (snake_case, unique-per-org)
├── display_name                  (human-readable)
├── description                   (text)
├── skill_prompt                  (injected into autopilot system prompt
│                                  to tell the agent WHEN to call this)
├── parameters JSONB              ([{name, type, pattern, required, ...}, ...])
├── actions    JSONB              ([{action_type, target, value, reasoning}, ...])
├── tags       JSONB              (semantic search hints)
├── portal                        (e.g., "sunfire", "salesforce")
├── scope                         ("user" | "org")
├── visibility                    ("approved" | "pending_approval" | "rejected")
├── enabled                       (bool)
├── current_version               (int, bumped on edit)
├── use_count, last_used_at       (denormalized for fast UI)
├── qdrant_point_id               (deterministic UUID5 hash for upserts)
│
├── marketplace_published         (bool) ← cross-org publication
├── marketplace_install_count     (int)
├── marketplace_source_workflow_id (uuid) ← walk-back to original
│
├── source_run_id                 (which automation_test_run produced this)
├── created_at, updated_at, deleted_at (soft delete)

learned_workflow_versions          ← snapshot before every substantive edit
├── workflow_id, version_number, change_summary, ...

learned_workflow_invocations       ← every /invoke call
├── workflow_id, invoked_by_user_id, params, success, actions_executed, ...

learned_workflow_stats VIEW        ← aggregates invocations live
   workflow_id, use_count, success_count, failure_count, success_rate,
   last_used_at, unique_users
```

## The synthesizer (recording → template)

```python
from learned_workflows.synthesizer import synthesize_workflow_from_run

# Pull a recorded run + its events
run = {"id": "...", "intent_description": "customer lookup", "seed_data": {...}}
events = [
    {"event_type": "instruction", "comment": "fill ZIP: 90210 MBI: 1A2B3C4D5E6 DOB: 11/09/1962",
     "metadata": {"last_actions": [
        {"action_type": "type", "target": "#zip", "value": "90210"},
        {"action_type": "type", "target": "[data-testid=mbi]", "value": "1A2B3C4D5E6"},
        {"action_type": "click", "target": "button.lookup-primary"},
     ]}},
]

synth = await synthesize_workflow_from_run(run, events)
# Returns WorkflowSynthesis with:
#   name="medicare_customer_lookup"
#   display_name="Medicare Customer Lookup"
#   description=<auto-generated by Ollama Cloud>
#   skill_prompt=<auto-generated, 2-4 sentences>
#   parameters=[
#       {"name": "zip", "type": "string", "pattern": "^\\d{5}$"},
#       {"name": "mbi", "type": "string"},
#       {"name": "dob", "type": "string"},
#   ]
#   actions=[
#       {"action_type": "type", "target": "#zip", "value": "{{zip}}"},
#       {"action_type": "type", "target": "[data-testid=mbi]", "value": "{{mbi}}"},
#       {"action_type": "click", "target": "button.lookup-primary"},
#   ]
#   tags=["medicare", "customer_lookup", "sunfire"]
```

Aggressive parameterization:
- **Loose regex**: lowercase MBI (`6uq8v57xd43`) matches, digits-only DOB (`11091962`) matches
- **Instruction-text parsing**: `DOB: 11091962 MBI: 6uq8v57xd43 ZIP: 63664` → every value the user typed becomes a `{{param}}`
- **UI text allowlist**: "Submit", "Cancel", "yes" never get parameterized
- **Order matters**: phone (`555-555-1212`) is checked BEFORE MBI (mixed-alphanumeric-required) to prevent collision

## The marketplace (the part nobody else has)

```
Agency A: builds "medicare_customer_lookup" workflow                    Agency B: searches "look up customer"
        ↓                                                                       ↓
   visibility=approved                                              /marketplace/list?search=look up
        ↓                                                                       ↓
   admin clicks "Publish to marketplace"                            Agency B sees the listing
        ↓                                                                       ↓
   marketplace_published=true                                        clicks "Install"
        ↓                                                                       ↓
   appears in /marketplace/list cross-org                            POST /marketplace/install/{source_workflow_id}
                                                                              ↓
                                                                       Fork into Agency B's org:
                                                                       • new UUID, new organization_id
                                                                       • new created_by_user_id
                                                                       • scope=org, visibility=approved
                                                                       • marketplace_source_workflow_id → source
                                                                       • current_version=1 (fresh history)
                                                                       • indexed into Agency B's Qdrant collection
                                                                              ↓
                                                                       Source's marketplace_install_count++
```

Every install creates an **independent copy** — Agency B can edit, version, deprecate without affecting Agency A. Walk-back via `marketplace_source_workflow_id` for attribution / future "upstream changed" notifications.

## Versioning

Every edit to a workflow (name, actions, parameters, etc.) automatically **snapshots the previous state** into `learned_workflow_versions` before applying the patch. `current_version` bumps. Rollback is one POST:

```python
POST /v1/learned-workflows/{id}/rollback/3
# Snapshots current state as v_(N+1), restores v3, bumps current_version.
```

The `learned_workflow_stats` Postgres VIEW aggregates `use_count` / `success_rate` / `last_used_at` / `unique_users` live so the admin dashboard is always fresh without a denormalization job.

## Stack

| Layer | Tech |
|---|---|
| API | FastAPI |
| Truth | Postgres (Supabase, RLS by `organization_id`) |
| Semantic search | Qdrant (per-org collection, deterministic UUID5 point IDs) |
| Relationships | Neo4j (Workflow → Org → Portal → Tag) |
| Synthesis LLM | Ollama Cloud (deepseek-v4-pro) — pluggable |
| Migrations | Alembic (3 migrations included) |

## What you can do with this

- **Drop into any agent system** — the synthesizer is generic; pair it with [Browser-Use](https://github.com/browser-use/browser-use), [Skyvern](https://github.com/Skyvern-AI/skyvern), or the sibling [`agentic-browser-lab`](https://github.com/YOUR_HANDLE/agentic-browser-lab)
- **B2B SaaS multi-tenant** — every endpoint is RLS-gated by `organization_id` from JWT claims (never from request body)
- **Marketplace flywheel** — opt-in cross-tenant sharing without surrendering tenant isolation
- **Audit trail** — version history + invocation logs satisfy SOC2 change-tracking requirements
- **Search-first UX** — every workflow is embedded + searchable so the agent can suggest "you have a saved tool for this" before going to freeform planning

## Status

Production in May 2026. Three migrations applied to a live Supabase. 15 endpoints. Designed to plug into any tenant context provider (FastAPI dependency injection).

## License

MIT — see [LICENSE](./LICENSE)

## Maintainer

[@axumquant](https://github.com/axumquant) — extracted from Sales Coach (Medicare insurance B2B platform).
