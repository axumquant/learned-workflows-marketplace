# Architecture

## Storage triad: why three databases

Each store is chosen for what it's actually good at — none is forced
into a role it doesn't fit.

| Store | Role | Why this one |
|---|---|---|
| **Postgres** (Supabase) | Source of truth: workflow rows + versions + invocations | ACID writes, row-level security by `organization_id` for multi-tenant isolation, JSONB for actions/params, postgres views for live stats |
| **Qdrant** | Semantic search of workflows | Cosine similarity over embedded `name + description + skill_prompt + tags` so the agent can find "a tool to look up customers" without remembering the exact name |
| **Neo4j** | Relationship graph: workflow ↔ org ↔ portal ↔ user ↔ tag | Powers "what tools work on SunFire?" / "what comes after this one?" / "who else uses this?" queries that would be ugly in SQL |

## RLS pattern

Every Postgres write/read includes `organization_id` from the JWT
claim, never from the request body. The `learned_workflows` table has
RLS policies that enforce `organization_id = current_setting('app.org_id')`.
Even a privileged service-role key follows the policy because the API
sets the org context per-request.

```
JWT ──verify──▶ TenantContext { organization_id, user_id }
                       │
                       ▼
              FastAPI Depends(get_tenant_context)
                       │
                       ▼
              repo.list(organization_id=ctx.organization_id, ...)
                       │
                       ▼
              SELECT ... WHERE organization_id = $1
              (+ RLS policy enforces it at the DB level too)
```

## Marketplace: how the fork works

```
Source row (Agency A):                Marketplace                Forked row (Agency B):
{                                                                {
  id: A1,                                                          id: B1                ← new UUID
  organization_id: A,                                              organization_id: B    ← caller's org
  created_by_user_id: alice,                                       created_by_user_id: bob
  name: "customer_lookup",                                         name: "customer_lookup"
  display_name: "Customer Lookup",                                 display_name: "Customer Lookup"
  description: "...",                                              description: "..."     ← deep-copied
  skill_prompt: "...",                                             skill_prompt: "..."    ← deep-copied
  parameters: [{name:"zip",...}],     ──install→                   parameters: [...]      ← deep-copied
  actions: [{target:"#zip",...}],                                  actions: [...]         ← deep-copied
  tags: [...],                                                     tags: [...]            ← deep-copied
  portal: "sunfire",                                               portal: "sunfire"
  scope: "org",                                                    scope: "org"
  visibility: "approved",                                          visibility: "approved"
  enabled: true,                                                   enabled: true
  marketplace_published: true,                                     marketplace_published: false  ← fork not auto-published
  marketplace_install_count: 12,                                   marketplace_install_count: 0
  marketplace_source_workflow_id: null,                            marketplace_source_workflow_id: A1  ← walk-back
  current_version: 5,                                              current_version: 1     ← fresh version history
  use_count: 234,                                                  use_count: 0           ← fresh stats
  ...                                                            }
}                                                                  + indexed into Qdrant collection coach_<B-uuid>

After install: source.marketplace_install_count = 13               + edges added to Neo4j: (B1)-[:DERIVED_FROM]->(A1)
```

Critical: the fork is **fully independent**. Agency B can edit, version,
even delete their copy without affecting Agency A's row. The
`marketplace_source_workflow_id` is purely for attribution and possible
future "upstream changed" notifications — never for cascading writes.

## Versioning: snapshot-on-edit

Every `repo.update()` call checks if the patch touches any SUBSTANTIVE key:

```python
SUBSTANTIVE_KEYS = {
    "name", "display_name", "description", "skill_prompt",
    "parameters", "actions", "tags", "portal", "scope", "visibility",
}
```

If yes:
1. Snapshot the current row into `learned_workflow_versions` with the
   change_summary (caller supplies via `_change_summary` patch key).
2. Bump `current_version`.
3. Apply the patch.

Non-substantive updates (e.g., `enabled=true`, `use_count=124`) DO NOT
create a version — that would be noise.

Rollback is one POST:
```
POST /v1/learned-workflows/{id}/rollback/{version_number}
```
Which snapshots the current state (so the rollback is undoable too) then
patches the row back to the target version.

## Stats: live postgres view

```sql
CREATE OR REPLACE VIEW learned_workflow_stats AS
SELECT
    w.id, w.organization_id, w.name, w.display_name,
    w.scope, w.visibility, w.enabled, w.current_version,
    COUNT(i.id)                                              AS use_count,
    COUNT(i.id) FILTER (WHERE i.success = true)              AS success_count,
    COUNT(i.id) FILTER (WHERE i.success = false)             AS failure_count,
    CASE WHEN COUNT(i.id) > 0
         THEN ROUND( (COUNT(i.id) FILTER (WHERE i.success))::numeric
                   / COUNT(i.id)::numeric, 4)
         ELSE 0 END                                          AS success_rate,
    MAX(i.created_at)                                        AS last_used_at,
    COUNT(DISTINCT i.invoked_by_user_id)                     AS unique_users
FROM learned_workflows w
LEFT JOIN learned_workflow_invocations i ON i.workflow_id = w.id
GROUP BY w.id, w.organization_id, ...
```

No denormalization job, no scheduled refresh — the view is always fresh.
The admin dashboard hits `GET /v1/learned-workflows/stats` and Postgres
recomputes from invocations in real time. Add a materialized view if
your invocation table gets to billions of rows.

## Synthesizer: aggressive parameterization

Three layers of recognition, in priority order:

1. **Instruction-text parsing** — the user's typed comment
   (`"DOB:11091962 MBI:6uq8v57xd43 ZIP:63664"`) is parsed for
   `keyword:value` pairs using a regex built dynamically from the
   `_KEYWORD_TO_PARAM` dict. Only KNOWN keywords match (no greedy
   `data: DOB:` collision). Every value the user names becomes a
   `{{param}}` regardless of format.

2. **Seed data match** — if a typed value equals a value in the run's
   `seed_data` dict, use that key's name.

3. **PII regex** — case-insensitive MBI (11-char mixed alphanumeric),
   ZIP (5 digits), DOB (slashed or 8-digit), SSN, phone (multiple
   formats), email. A UI-text allowlist (`Submit`, `Cancel`, `yes`)
   protects common button/link text from false-positives.

The output is a clean **template**, not a recording: every literal
test value is replaced with a parameter placeholder.

## Why no ORM?

The repository talks to Supabase via REST (`PostgREST`) instead of
SQLAlchemy. Two reasons:

1. **Multi-tenant RLS works natively** in PostgREST — the JWT is
   passed as `Authorization: Bearer <jwt>`, RLS reads the org_id
   claim, no app-side filter needed.
2. **Migration-light** — schema lives in Alembic migrations, but the
   app layer is just Pydantic + httpx. No SQLAlchemy model drift
   between code + DB.

The trade-off: bulk inserts are slower. For this workload (low write
volume, high read volume) it's fine.
