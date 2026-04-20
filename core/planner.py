"""
core/planner.py
───────────────
Task planner: turns a high-level goal into a structured sequence of steps.

Two tiers:
  1. Rule-based decomposition — instant, no LLM call, handles common patterns
  2. LLM-based decomposition — for novel goals, async, produces richer plans

Each step has:
  - title       short label
  - action_type "think" | "tool" | "ask" | "inform"
  - tool        optional tool name to invoke
  - args        dict of arguments
  - status      "pending" | "running" | "done" | "failed" | "skipped"
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

log = logging.getLogger("aida.planner")


@dataclass
class PlanStep:
    id:          str
    title:       str
    description: str
    action_type: str                   # "think" | "tool" | "ask" | "inform"
    tool:        Optional[str] = None
    args:        dict          = field(default_factory=dict)
    status:      str           = "pending"
    result:      Optional[str] = None
    created_at:  str           = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Plan:
    id:         str
    goal:       str
    steps:      list[PlanStep]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed:  bool = False

    def pending_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.status == "pending"]

    def next_step(self) -> Optional[PlanStep]:
        p = self.pending_steps()
        return p[0] if p else None

    def to_display(self) -> str:
        """Compact string for UI display."""
        icons = {"pending": "○", "running": "◉", "done": "✓",
                 "failed": "✗", "skipped": "⊘", "think": "💭",
                 "tool": "⚙", "ask": "❓", "inform": "ℹ"}
        lines = [f"📋 **{self.goal}**"]
        for i, step in enumerate(self.steps, 1):
            st = icons.get(step.status, "○")
            lines.append(f"  {st} {i}. {step.title}")
            if step.result and step.status == "done":
                lines.append(f"     └ {step.result[:80]}")
        return "\n".join(lines)


# ── Rule-based decomposition templates ───────────────────────────────────────

_TEMPLATES: list[tuple[str, list[dict]]] = [
    # Study sprint
    (r"\b(study|learn|review)\b",
     [{"title": "Identify topics",        "action_type": "think"},
      {"title": "Search background info", "action_type": "tool", "tool": "web_search"},
      {"title": "Create study checklist", "action_type": "inform"},
      {"title": "Schedule review blocks", "action_type": "inform"}]),

    # New project
    (r"\b(new project|create project|start project)\b",
     [{"title": "Define project scope",   "action_type": "think"},
      {"title": "Create project folder",  "action_type": "tool", "tool": "filesystem"},
      {"title": "Create README",          "action_type": "tool", "tool": "filesystem"},
      {"title": "Suggest next actions",   "action_type": "inform"}]),

    # Research task
    (r"\b(research|investigate|analyse|analyze|find out)\b",
     [{"title": "Search primary sources", "action_type": "tool", "tool": "web_search"},
      {"title": "Synthesize findings",    "action_type": "think"},
      {"title": "Summarize for user",     "action_type": "inform"}]),

    # Writing task
    (r"\b(write|draft|compose|create a (post|article|essay|email|report))\b",
     [{"title": "Outline structure",      "action_type": "think"},
      {"title": "Draft content",          "action_type": "think"},
      {"title": "Review and refine",      "action_type": "ask"},
      {"title": "Save final version",     "action_type": "tool", "tool": "filesystem"}]),

    # Setup/workflow
    (r"\b(set up|setup|configure|prepare|organize|organise)\b",
     [{"title": "Assess current state",   "action_type": "ask"},
      {"title": "Identify requirements",  "action_type": "think"},
      {"title": "Execute setup steps",    "action_type": "tool"},
      {"title": "Verify and confirm",     "action_type": "inform"}]),

    # Coding task
    (r"\b(build|implement|code|program|develop|fix|debug)\b",
     [{"title": "Clarify requirements",   "action_type": "ask"},
      {"title": "Design approach",        "action_type": "think"},
      {"title": "Implement solution",     "action_type": "tool", "tool": "filesystem"},
      {"title": "Test and validate",      "action_type": "think"}]),
]


def _make_step(raw: dict, index: int) -> PlanStep:
    return PlanStep(
        id          = str(uuid.uuid4())[:8],
        title       = raw["title"],
        description = raw.get("description", raw["title"]),
        action_type = raw.get("action_type", "think"),
        tool        = raw.get("tool"),
        args        = raw.get("args", {}),
    )


class Planner:
    def __init__(self):
        self._current_plan: Optional[Plan] = None

    @property
    def has_active_plan(self) -> bool:
        return (self._current_plan is not None and
                not self._current_plan.completed and
                bool(self._current_plan.pending_steps()))

    @property
    def current_plan(self) -> Optional[Plan]:
        return self._current_plan

    def decompose(self, goal: str) -> Plan:
        """
        Rule-based decomposition. Returns a Plan immediately.
        For novel goals not matching templates, returns a minimal 3-step plan.
        """
        lower = goal.lower()
        steps_raw: Optional[list[dict]] = None

        for pattern, template in _TEMPLATES:
            if re.search(pattern, lower):
                steps_raw = template
                log.info("Planner matched template for: %r", goal[:60])
                break

        if steps_raw is None:
            # Generic fallback
            steps_raw = [
                {"title": "Understand the goal",  "action_type": "think"},
                {"title": "Execute main action",  "action_type": "tool"},
                {"title": "Report result",        "action_type": "inform"},
            ]

        steps = [_make_step(s, i) for i, s in enumerate(steps_raw)]
        plan  = Plan(id=str(uuid.uuid4())[:8], goal=goal, steps=steps)
        self._current_plan = plan
        log.info("Plan created: %d steps for %r", len(steps), goal[:60])
        return plan

    async def decompose_with_llm(self, goal: str, llm_callable) -> Plan:
        """
        LLM-assisted decomposition for novel goals.
        llm_callable is an async function: (prompt: str) -> str
        Falls back to rule-based if LLM fails.
        """
        prompt = (
            f"Break this goal into 3-5 concrete steps: \"{goal}\"\n"
            "Respond ONLY as a numbered list, one step per line. "
            "Each step: short imperative phrase (max 8 words)."
        )
        try:
            raw = await llm_callable(prompt)
            step_titles = [
                re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
                for line in raw.strip().splitlines()
                if line.strip() and re.match(r"^\s*\d", line)
            ][:6]
            if step_titles:
                steps_raw = [{"title": t, "action_type": "think"} for t in step_titles]
                steps = [_make_step(s, i) for i, s in enumerate(steps_raw)]
                plan  = Plan(id=str(uuid.uuid4())[:8], goal=goal, steps=steps)
                self._current_plan = plan
                return plan
        except Exception as exc:
            log.warning("LLM decomposition failed: %s — falling back to rules", exc)
        return self.decompose(goal)

    def mark_step_done(self, step_id: str, result: str = "") -> None:
        if not self._current_plan:
            return
        for step in self._current_plan.steps:
            if step.id == step_id:
                step.status = "done"
                step.result = result
        if not self._current_plan.pending_steps():
            self._current_plan.completed = True

    def clear(self) -> None:
        self._current_plan = None

    def is_goal_request(self, text: str) -> bool:
        """Heuristic: does this look like a multi-step goal rather than a single query?"""
        triggers = [
            r"\b(help me|i want to|i need to|can you help|let's|let us)\b",
            r"\b(create a|build a|set up|make a|start a|plan a|prepare)\b",
            r"\b(step by step|walkthrough|guide me|workflow)\b",
        ]
        lower = text.lower()
        return any(re.search(p, lower) for p in triggers) and len(text) > 20
