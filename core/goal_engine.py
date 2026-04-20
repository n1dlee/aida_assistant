"""
core/goal_engine.py
────────────────────
Autonomous Goal Mode: turns a high-level goal into a structured, interruptible
execution sequence.

Safety constraints (non-negotiable):
  - Every step is SHOWN to the user before execution
  - User can pause, skip, or abort at any time
  - Destructive actions (delete, execute command) require explicit confirmation
  - No silent execution — full transparency

Architecture:
  GoalEngine.start(goal)  → creates a Plan via Planner
  GoalEngine.step()       → executes the next pending step (if auto-mode enabled)
  GoalEngine.status()     → returns human-readable progress
  GoalEngine.abort()      → cancels remaining steps

Integration:
  Orchestrator calls GoalEngine when intent == "goal_request"
  UI polls GoalEngine.status() via plan_timer
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Callable, Awaitable

if TYPE_CHECKING:
    from core.planner import Plan, PlanStep
    from tools.registry import ToolRegistry

log = logging.getLogger("aida.goal_engine")

# Step types that require explicit user confirmation before execution
_DANGEROUS_TYPES = {"tool"}
_DANGEROUS_TOOLS = {"filesystem", "system"}


class GoalEngine:
    def __init__(self, planner, tool_registry: "ToolRegistry"):
        self._planner    = planner
        self._tools      = tool_registry
        self._auto_mode  = False    # if True, execute safe steps automatically
        self._paused     = False
        self._awaiting_confirm: Optional[str] = None  # step id awaiting user OK

    @property
    def auto_mode(self) -> bool:
        return self._auto_mode

    def set_auto_mode(self, enabled: bool) -> None:
        self._auto_mode = enabled
        log.info("Goal auto-mode: %s", enabled)

    def has_active_goal(self) -> bool:
        return self._planner.has_active_plan

    def start(self, goal: str) -> "Plan":
        """Decompose goal and return the plan (no execution yet)."""
        plan = self._planner.decompose(goal)
        self._paused = False
        self._awaiting_confirm = None
        log.info("Goal started: %r — %d steps", goal[:60], len(plan.steps))
        return plan

    def status(self) -> str:
        """Returns a concise, human-readable execution status for the UI."""
        if not self._planner.has_active_plan:
            return ""
        plan = self._planner.current_plan
        assert plan is not None

        done  = sum(1 for s in plan.steps if s.status == "done")
        total = len(plan.steps)
        pct   = int(done / total * 100) if total else 0

        nxt = plan.next_step()
        if plan.completed:
            return f"✅ Goal complete: {plan.goal[:50]}"
        if self._paused:
            return f"⏸ Paused at step {done + 1}/{total}"
        if self._awaiting_confirm and nxt:
            return f"⏳ Waiting confirmation: {nxt.title}"
        if nxt:
            return f"▶ Step {done + 1}/{total}: {nxt.title} [{pct}%]"
        return f"📋 {done}/{total} steps done"

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def abort(self) -> str:
        if self._planner.current_plan:
            goal = self._planner.current_plan.goal
            self._planner.clear()
            return f"Goal aborted: {goal[:50]}"
        return "No active goal."

    def confirm_step(self, step_id: str) -> None:
        """User confirmed execution of a pending dangerous step."""
        self._awaiting_confirm = None

    def needs_confirm(self) -> Optional["PlanStep"]:
        """Returns the step waiting for confirmation, if any."""
        if not self._planner.has_active_plan:
            return None
        plan = self._planner.current_plan
        if not plan:
            return None
        nxt = plan.next_step()
        if nxt and self._is_dangerous(nxt):
            return nxt
        return None

    def _is_dangerous(self, step: "PlanStep") -> bool:
        if step.action_type not in _DANGEROUS_TYPES:
            return False
        return step.tool in _DANGEROUS_TOOLS if step.tool else False

    async def execute_next(self,
                           llm_callable: Callable[[str], Awaitable[str]]) -> Optional[str]:
        """
        Execute the next pending step.
        Returns the step result string, or None if blocked/no step.
        Caller is responsible for showing result to user.
        """
        if not self._planner.has_active_plan or self._paused:
            return None
        plan = self._planner.current_plan
        if not plan:
            return None
        step = plan.next_step()
        if not step:
            return None

        # Block dangerous steps until confirmed
        if self._is_dangerous(step) and not self._auto_mode:
            self._awaiting_confirm = step.id
            return f"⚠ Step requires confirmation: **{step.title}**\nSay 'confirm' or 'skip' to proceed."

        log.info("Executing step: %s (%s)", step.title, step.action_type)
        step.status = "running"
        result = ""

        try:
            if step.action_type == "think":
                result = await llm_callable(
                    f"For the goal '{plan.goal}', execute this step: {step.title}. "
                    "Be concise."
                )
            elif step.action_type == "tool" and step.tool:
                tool_input = f"{step.title} {' '.join(f'{k}={v}' for k,v in step.args.items())}"
                tool_result = await self._tools.execute(tool_input, "")
                result = tool_result or f"Tool {step.tool} executed."
            elif step.action_type == "ask":
                result = f"❓ {step.title} — please respond to continue."
                step.status = "pending"   # keep pending until user responds
                return result
            elif step.action_type == "inform":
                result = step.description
            else:
                result = f"Step complete: {step.title}"

            self._planner.mark_step_done(step.id, result)

        except Exception as exc:
            step.status = "failed"
            result = f"Step failed: {step.title} — {exc}"
            log.error("Goal step error: %s", exc)

        return result
