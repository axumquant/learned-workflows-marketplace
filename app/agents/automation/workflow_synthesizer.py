"""Workflow synthesizer — turn a recorded Automation Lab run into a
reusable learned_workflow definition.

Pipeline:

  1. Read the run + events from the automation_test_runs / _events tables.
  2. Walk the events in created_at order. For each event whose
     ``metadata.kind == "live_instruction"``, look at ``decision`` and
     dom_summary to deduce action steps. Also walk decision/snapshot events
     for the picks the user approved.
  3. Build a candidate action sequence (ExecutorAction shape) — substitute
     typed values that look like seed_data fields or PII patterns with
     ``{{placeholder}}`` and create matching WorkflowParameter entries.
  4. Call Ollama Cloud (deepseek-v4-pro) with the candidate sequence + page
     context to generate:
       - name (snake_case)
       - display_name
       - description
       - skill_prompt (what to tell the autopilot agent)
       - tags
     The model is encouraged to keep our auto-detected parameters as-is.
  5. Return a ``WorkflowSynthesis`` Pydantic object the API hands back to
     the extension for review before save.

If Ollama Cloud is unreachable, we still return a synthesis built entirely
from heuristics — never block the user from saving a recording.
"""

from __future__ import annotations

import json
import re
from typing import Any

import structlog
from pydantic import BaseModel, Field
from pydantic_ai.settings import ModelSettings

from app.agents.ai_models import make_pai_agent
from app.agents.automation.pai_wiring import _ollama_cloud_settings

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Synthesis output (returned by both LLM and heuristic paths)
# ---------------------------------------------------------------------------


class SynthesizedParameter(BaseModel):
    name: str = Field(..., description="snake_case parameter name")
    type: str = Field(default="string", description="string | number | boolean")
    description: str = Field(default="", description="What the user should supply")
    pattern: str = Field(default="", description="Optional regex constraint")
    required: bool = True


class SynthesizedAction(BaseModel):
    action_type: str
    target: str = ""
    value: str = ""
    reasoning: str = ""
    frame_url: str = ""


class WorkflowSynthesis(BaseModel):
    """LLM-shaped output for a single recorded run."""

    name: str = Field(..., description="snake_case machine-callable name")
    display_name: str = Field(..., description="Human-readable label")
    description: str = Field(..., description="What the tool does")
    skill_prompt: str = Field(..., description="Instruction injected into the autopilot agent's system prompt")
    portal: str = Field(default="", description="Portal key (sunfire, enrollhere, etc.)")
    parameters: list[SynthesizedParameter] = Field(default_factory=list)
    actions: list[SynthesizedAction] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Heuristic action extraction
# ---------------------------------------------------------------------------

# Common PII / identifier patterns. Order matters — first match wins.
# Patterns intentionally loose so common formats (lowercase MBI,
# digits-only DOB, dashed phone) all parameterize. The goal is
# AGGRESSIVE templating — anything that looks like personal data
# becomes a `{{param}}` placeholder so the saved workflow is a
# clean template, not a recording with test PII baked in.
_PARAM_PATTERNS: list[tuple[str, str, str, int]] = [
    # (regex, suggested_name, description, re_flags)
    # ORDER MATTERS: more-specific / less-ambiguous patterns first so
    # a phone number ("555-555-1212") doesn't get mis-tagged as MBI.
    (r"^\d{5}(-\d{4})?$", "zip", "5-digit ZIP code", 0),
    (r"^[^@\s]+@[^@\s]+\.[^@\s]+$", "email", "Email address", 0),
    (r"^\d{3}-\d{2}-\d{4}$", "ssn", "Social Security number", 0),
    (r"^\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$", "phone", "Phone number", 0),
    # Slashed DOB (MM/DD/YYYY) and digits-only DOB (MMDDYYYY).
    (r"^\d{2}/\d{2}/\d{4}$", "dob", "Date of birth (MM/DD/YYYY)", 0),
    (r"^\d{8}$", "dob", "Date of birth (MMDDYYYY, digits only)", 0),
    # MBI: 11 characters, alphanumeric, MUST contain at least one
    # letter AND at least one digit. The mixed-class requirement
    # excludes pure-digit strings (which would otherwise collide
    # with 11-digit phone numbers like "5555551212"). Case-insensitive
    # so lowercase test data like "6uq8v57xd43" still matches.
    (
        r"^(?=.*[A-Z])(?=.*\d)[A-Z0-9]{11}$",
        "mbi",
        "Medicare Beneficiary ID (11 chars, mixed alphanumeric)",
        re.IGNORECASE,
    ),
    # MMDD or MM/DD for date-month-day inputs
    (r"^\d{1,2}/\d{1,2}$", "date_md", "Month/day", 0),
    # Single field date parts often typed into 3-input DOB widgets
    (r"^\d{4}$", "year", "4-digit year", 0),
]

# Common UI text — never parameterize these even if they pass a pattern.
# Adding to this list when a workflow surfaces false positives is fine.
_UI_TEXT_ALLOWLIST: frozenset[str] = frozenset({
    "submit", "next", "previous", "back", "continue", "cancel", "ok",
    "yes", "no", "save", "lookup", "search", "find plans", "find",
    "true", "false", "0", "1",
})


def _detect_param(value: str, seed_keys: set[str]) -> tuple[str, str] | None:
    """Return ``(param_name, description)`` if the value looks parameterizable.

    Order of precedence:
      1. Value matches a key name in seed_data (handled upstream)
      2. Value matches a known PII regex → use the regex's suggested name
      3. None
    """
    if not value:
        return None
    v = str(value).strip()
    if not v or v.lower() in _UI_TEXT_ALLOWLIST:
        return None
    for pattern, name, desc, flags in _PARAM_PATTERNS:
        if re.fullmatch(pattern, v, flags):
            return name, desc
    return None


def _value_matches_seed(value: str, seed_data: dict[str, Any]) -> str | None:
    """If a typed value equals a value in seed_data, return the seed_data key."""
    if not value or not seed_data:
        return None
    target = str(value).strip()
    for key, seed_val in seed_data.items():
        if seed_val is None:
            continue
        if str(seed_val).strip() == target:
            return key
    return None


# Match keyword:value pairs in the user's instruction text — e.g.
# "DOB:11091962 MBI:6uq8v57xd43 ZIP: 63664". Built dynamically from
# the known keyword set BELOW (_KEYWORD_TO_PARAM is filled first
# in module load order, then we compile the regex at the bottom of
# this file).
_INSTRUCTION_KV_RE: re.Pattern[str] | None = None  # compiled lazily — see _build_kv_re()

_KEYWORD_TO_PARAM: dict[str, tuple[str, str]] = {
    "zip": ("zip", "ZIP code"),
    "zip_code": ("zip", "ZIP code"),
    "zipcode": ("zip", "ZIP code"),
    "mbi": ("mbi", "Medicare Beneficiary ID"),
    "medicare_number": ("mbi", "Medicare Beneficiary ID"),
    "medicare": ("mbi", "Medicare Beneficiary ID"),
    "dob": ("dob", "Date of birth"),
    "date_of_birth": ("dob", "Date of birth"),
    "ssn": ("ssn", "Social Security number"),
    "phone": ("phone", "Phone number"),
    "email": ("email", "Email address"),
    "first_name": ("first_name", "First name"),
    "last_name": ("last_name", "Last name"),
    "full_name": ("full_name", "Full name"),
    "client_name": ("client_name", "Client / member full name"),
    "address": ("address", "Street address"),
    "city": ("city", "City"),
    "state": ("state", "State / region"),
}


def _build_kv_re() -> re.Pattern[str]:
    """Compile the keyword:value regex from the known _KEYWORD_TO_PARAM
    keys. By restricting the keyword side to a fixed alternation we
    avoid the greedy-match bug where "data:" gets parsed as a key and
    "DOB:" as its value, consuming the actual DOB keyword that follows.
    """
    # Sort longest first so e.g. "date_of_birth" matches before "date".
    keys = sorted(_KEYWORD_TO_PARAM.keys(), key=len, reverse=True)
    # Allow spaces or _ or - inside the keyword as user might type it.
    alt = "|".join(re.escape(k).replace(r"\_", "[ _-]") for k in keys)
    return re.compile(rf"(?:^|[\s.,;])({alt})\s*[:=]\s*([^\s,;]{{2,80}})", re.IGNORECASE)


def _values_from_instruction(comment: str) -> dict[str, tuple[str, str]]:
    """Parse "KEY: value KEY: value" patterns out of the instruction text.

    Returns ``{value: (param_name, description)}`` so every literal value
    the user named in the instruction becomes a parameter in the saved
    workflow — making the result a clean template instead of a
    PII-baked recording.

    The regex is compiled from the _KEYWORD_TO_PARAM keys directly,
    so only known param keywords match — unknown words can't act as
    keys and accidentally consume the next real keyword's value.
    """
    global _INSTRUCTION_KV_RE
    if _INSTRUCTION_KV_RE is None:
        _INSTRUCTION_KV_RE = _build_kv_re()
    out: dict[str, tuple[str, str]] = {}
    if not comment:
        return out
    for match in _INSTRUCTION_KV_RE.finditer(comment):
        raw_key = match.group(1).strip().lower().replace(" ", "_").replace("-", "_")
        raw_val = match.group(2).strip().rstrip(",.;:")
        if not raw_val or raw_key not in _KEYWORD_TO_PARAM:
            continue
        param_name, desc = _KEYWORD_TO_PARAM[raw_key]
        out[raw_val] = (param_name, desc)
    return out


def _extract_actions_from_events(
    events: list[dict[str, Any]],
    seed_data: dict[str, Any],
) -> tuple[list[SynthesizedAction], list[SynthesizedParameter]]:
    """Walk events and produce a candidate action sequence with placeholders.

    We pull from two event sources:
      - ``automation_lab_decision`` events recorded after a Capture/Approve
        click (the historical path)
      - ``instruction`` events whose backend response included planned
        actions (the agentic-loop path)

    For the agentic-loop path the planned actions were stored on the event
    in two places: the comment shows the user's instruction; the
    ``metadata`` may include the planner's reasoning. We don't replay the
    backend response (we don't have it) — instead we rebuild the action
    list from the dom_summary trail + any decision/snapshot events that
    captured the resulting page state.

    Heuristic: collect every action that was actually performed against
    the DOM. For now we treat each ``instruction`` event's comment as a
    high-level step and recover the concrete (type/click) actions from
    the planner's returned shape that was logged into the same event's
    ``metadata.last_actions``. Backend writes this when present.
    """
    actions: list[SynthesizedAction] = []
    params_by_name: dict[str, SynthesizedParameter] = {}

    # Build a "values mentioned in user instructions" lookup. Every
    # instruction event's comment ("DOB:11091962 MBI:6uq8v57xd43
    # ZIP: 63664") gets parsed for keyword:value pairs — those values
    # are guaranteed test data and MUST become parameters.
    instruction_value_map: dict[str, tuple[str, str]] = {}
    for ev in events:
        if ev.get("event_type") == "instruction":
            comment = str(ev.get("comment") or "")
            instruction_value_map.update(_values_from_instruction(comment))

    def _consider_param(value: str) -> str:
        """Decide whether ``value`` becomes a placeholder; return the value
        the planner should see (either the original literal or
        ``{{param_name}}``).

        Precedence:
          1. Value appears in user's instruction text as KEY:VALUE → use that KEY
          2. Value matches a value in seed_data → use seed key name
          3. Value matches a PII regex → use the regex's suggested name
          4. Pass through literal (UI text, etc.)
        """
        if not value:
            return value
        v = str(value).strip()
        if not v:
            return value
        # 1. User typed it explicitly as KEY:VALUE in their instruction
        if v in instruction_value_map:
            name, desc = instruction_value_map[v]
            params_by_name.setdefault(name, SynthesizedParameter(name=name, description=desc))
            return f"{{{{{name}}}}}"
        # 2. Matches seed_data value
        seed_key = _value_matches_seed(v, seed_data)
        if seed_key:
            params_by_name.setdefault(
                seed_key,
                SynthesizedParameter(name=seed_key, description=f"Value for the '{seed_key}' field"),
            )
            return f"{{{{{seed_key}}}}}"
        # 3. Matches a known PII pattern (loose: case-insensitive MBI,
        # digits-only DOB, etc.)
        detected = _detect_param(v, set(seed_data.keys()))
        if detected:
            name, desc = detected
            params_by_name.setdefault(name, SynthesizedParameter(name=name, description=desc))
            return f"{{{{{name}}}}}"
        return value

    for ev in events:
        meta = ev.get("metadata") or {}
        last_actions = meta.get("last_actions") if isinstance(meta, dict) else None
        if isinstance(last_actions, list):
            for a in last_actions:
                if not isinstance(a, dict):
                    continue
                value = _consider_param(str(a.get("value") or ""))
                actions.append(SynthesizedAction(
                    action_type=str(a.get("action_type") or "").lower() or "click",
                    target=str(a.get("target") or ""),
                    value=value,
                    reasoning=str(a.get("reasoning") or "")[:240],
                    frame_url=str(a.get("frame_url") or ""),
                ))
    return actions, list(params_by_name.values())


def _heuristic_synthesis(
    run: dict[str, Any],
    actions: list[SynthesizedAction],
    parameters: list[SynthesizedParameter],
) -> WorkflowSynthesis:
    portal = str(run.get("provider") or "")
    workflow = str(run.get("workflow") or "")
    intent = str(run.get("intent_description") or "").strip()
    base = intent or workflow or "recorded_workflow"
    name_seed = re.sub(r"[^A-Za-z0-9]+", "_", base.lower()).strip("_") or "recorded_workflow"
    name = f"{portal}_{name_seed}" if portal else name_seed
    name = name[:80]
    display = " ".join(part.capitalize() for part in name.split("_"))
    tags = [t for t in [portal, workflow, "auto_generated"] if t]
    return WorkflowSynthesis(
        name=name,
        display_name=display,
        description=intent or f"Recorded {workflow or 'workflow'} on {portal or 'a portal'} ({len(actions)} steps).",
        skill_prompt=(
            f"Call this when the user wants to {(intent or 'replay this workflow').rstrip('.')}. "
            f"Required inputs: {', '.join(p.name for p in parameters) or 'none'}."
        ),
        portal=portal,
        parameters=parameters,
        actions=actions,
        tags=tags,
        confidence=0.4,
        reasoning="Built from heuristics (no LLM call).",
    )


# ---------------------------------------------------------------------------
# LLM synthesis
# ---------------------------------------------------------------------------

_WORKFLOW_SYNTHESIZER_SYSTEM = """You are a Workflow Distiller for a browser
automation tool. You receive a sequence of recorded DOM actions (type, click,
press_key, navigate, wait) that an agent performed on a portal, plus the
portal name, the user's original intent, and any visible page-state notes.

Your job: turn that recording into a reusable MCP-style tool definition
that the autopilot agent can call later with parameters.

Output rules:
- ``name``: short snake_case identifier prefixed with the portal name when
  relevant (e.g. ``sunfire_customer_lookup``). Max 80 chars.
- ``display_name``: human-readable label.
- ``description``: 1-2 sentences about what the workflow accomplishes and
  what page it ends on.
- ``skill_prompt``: 2-4 sentences telling the autopilot agent WHEN to call
  this tool, WHAT inputs it needs, and WHAT the result will be.
- ``parameters``: keep the auto-detected parameters (zip, mbi, dob, etc.)
  exactly as they are. Add more if you notice typed values that clearly
  belong to a customer profile (first_name, last_name, address). Each
  parameter has a name (snake_case), type, description, and optional regex
  pattern.
- ``actions``: pass the recorded actions through UNCHANGED. The system has
  already substituted ``{{param}}`` placeholders for parameterizable
  values. Do not reorder, rewrite, or filter actions.
- ``tags``: 3-6 short lower-case tags. Include the portal name, a category
  (e.g. ``customer_lookup``, ``enrollment``, ``quote``), and the vertical
  (e.g. ``medicare`` if Medicare-related). No duplicates.
- ``portal``: the portal key (already given to you — echo it).
- ``confidence``: 0.0-1.0 for how well-formed the resulting tool is.
- ``reasoning``: one short sentence on what this workflow does, in agent
  voice (third-person).

If the recorded actions don't tell a coherent story (random clicks, no
typing, no navigation), set confidence < 0.4 and explain in reasoning.
"""


async def synthesize_workflow_from_run(
    run: dict[str, Any],
    events: list[dict[str, Any]],
) -> WorkflowSynthesis:
    """Turn an automation run + its events into a workflow synthesis.

    Always returns a synthesis — falls back to pure heuristics if Ollama
    Cloud is unreachable or returns malformed output, so the user can
    still review/save a recording even when the LLM is down.
    """
    seed_data = run.get("seed_data") or {}
    actions, parameters = _extract_actions_from_events(events, seed_data)

    if not actions:
        # No usable actions — return a heuristic skeleton the user can
        # fill in by hand. Low confidence so the UI can warn them.
        return _heuristic_synthesis(run, actions, parameters)

    try:
        agent = make_pai_agent(
            WorkflowSynthesis,
            _WORKFLOW_SYNTHESIZER_SYSTEM,
            settings=_ollama_cloud_settings(),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("workflow_synthesizer_agent_build_failed", error=str(exc))
        agent = None

    if agent is None:
        log.info("workflow_synthesizer_no_llm_fallback")
        return _heuristic_synthesis(run, actions, parameters)

    user_msg = json.dumps(
        {
            "portal": run.get("provider") or "",
            "workflow_template": run.get("workflow") or "",
            "user_intent": run.get("intent_description") or "",
            "target_url": run.get("target_url") or "",
            "auto_detected_parameters": [p.model_dump() for p in parameters],
            "recorded_actions": [a.model_dump() for a in actions],
            "event_count": len(events),
        },
        indent=2,
        default=str,
    )

    try:
        result = await agent.run(
            user_msg,
            model_settings=ModelSettings(max_tokens=2048, temperature=0.2),
        )
        synthesis: WorkflowSynthesis = result.output
        # Defensive: keep the auto-detected actions and parameters even if
        # the LLM tried to rewrite or drop them. This is the user's
        # recording — don't let the model second-guess it.
        if not synthesis.actions:
            synthesis.actions = actions
        if not synthesis.parameters:
            synthesis.parameters = parameters
        if not synthesis.portal:
            synthesis.portal = str(run.get("provider") or "")
        return synthesis
    except Exception as exc:  # noqa: BLE001
        log.warning("workflow_synthesizer_llm_failed", error=str(exc))
        return _heuristic_synthesis(run, actions, parameters)


__all__ = [
    "WorkflowSynthesis",
    "SynthesizedAction",
    "SynthesizedParameter",
    "synthesize_workflow_from_run",
]
