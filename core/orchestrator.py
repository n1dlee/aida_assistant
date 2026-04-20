"""
core/orchestrator.py
──────────────────────
Central coordinator for all AIDA subsystems.

Pipeline:
  user input
    → context snapshot
    → mode addendum injected
    → shadow event logged
    → intent classification
    → planner (for goal requests)
    → tool dispatch (for tool intents)
    → LLM streaming (for conversation/planning)
    → long-term memory updated
    → predictive suggestions generated
    → response returned
"""
import asyncio
import logging
import time
from typing import Optional, AsyncIterator

from brain.selector       import ModelSelector
from memory.buffer        import ConversationBuffer
from memory.vector_store  import VectorStore
from memory.episodic      import EpisodicMemory
from memory.long_term     import LongTermMemory
from core.personality     import Personality
from core.router          import IntentRouter
from core.mode_manager    import ModeManager
from core.planner         import Planner
from core.context_manager import ContextManager
from core.shadow_engine   import ShadowEngine
from core.predictive_engine import PredictiveEngine
from tools.registry       import ToolRegistry
from config.feature_flags import Flags
from core.goal_engine    import GoalEngine
from models.skill_profiles import SkillManager

log = logging.getLogger("aida.orchestrator")


class Orchestrator:
    def __init__(self):
        # ── Brain ─────────────────────────────────────────────────────────────
        self.selector    = ModelSelector()

        # ── Memory ────────────────────────────────────────────────────────────
        self.buffer      = ConversationBuffer(max_turns=20)
        self.vector_store = VectorStore()
        self.episodic    = EpisodicMemory()
        self.long_term   = LongTermMemory()

        # ── Core systems ──────────────────────────────────────────────────────
        self.personality = Personality()
        self.router      = IntentRouter()
        self.modes       = ModeManager()
        self.planner     = Planner()
        self.context     = ContextManager()
        self.tools       = ToolRegistry()

        # ── Observability ─────────────────────────────────────────────────────
        self.shadow      = ShadowEngine(enabled=Flags.SHADOW_MODE)
        self.predictive  = PredictiveEngine(
            shadow     = self.shadow,
            long_term  = self.long_term,
        )

        self.goal_engine = GoalEngine(self.planner, self.tools)
        self.skills      = SkillManager()
        self._running    = False
        log.info("Orchestrator fully initialized.")

    # ── System prompt ─────────────────────────────────────────────────────────

    def _build_system_prompt(self, memories: list[str],
                              ctx_fragment: str = "") -> str:
        base       = self.personality.build_system_prompt(memories=memories)
        mode_add   = self.modes.build_addendum()
        skill_add  = self.skills.build_addendum()
        prompt     = base + mode_add + skill_add
        if ctx_fragment:
            prompt += f"\n\n{ctx_fragment}"
        return prompt

    # ── Main process (non-streaming) ──────────────────────────────────────────

    async def process(self, user_input: str) -> str:
        full = ""
        async for tok in self.stream_process(user_input):
            full += tok
        return full

    # ── Streaming process ─────────────────────────────────────────────────────

    async def stream_process(self, user_input: str) -> AsyncIterator[str]:
        t0 = time.monotonic()
        log.debug("Processing: %s", user_input[:80])

        # ── Context snapshot ──────────────────────────────────────────────────
        snap = self.context.snapshot(include_clipboard=False)
        snap.active_mode = self.modes.current_name
        self.context.set_intent(user_input[:80])

        # ── Shadow logging ────────────────────────────────────────────────────
        intent = self.router.classify(user_input)
        log.info("Intent (stream): %s | mode: %s", intent, self.modes.current_name)

        if self.shadow.enabled:
            self.shadow.log_app_switch(snap.process_name)
            self.shadow.log_intent(intent, self.modes.current_name)

        # ── Resolve "this" references ─────────────────────────────────────────
        this_ref = self.context.resolve_this(user_input)
        if this_ref:
            user_input += f"\n\n[Referenced content: {this_ref[:300]}]"

        # ── System query: AIDA self-status ───────────────────────────────────────
        if intent == "system_query":
            response = self._handle_system_query(user_input)
            self._update_memory(user_input, response)
            yield response
            return

        # ── Tool dispatch (fast path) ─────────────────────────────────────────
        if intent == "tool_call":
            t_start = time.monotonic()
            result = await self.tools.execute(user_input, llm_response="")
            if result:
                ms = int((time.monotonic() - t_start) * 1000)
                self._update_memory(user_input, result)
                if self.shadow.enabled:
                    self.shadow.log_tool_use("registry", success=True)
                if Flags.ACTION_LOGGING:
                    self.long_term.log_action(
                        tool="registry", input_text=user_input,
                        outcome=result, success=True, duration_ms=ms
                    )
                yield result
                return

        # ── Plan check: is this a multi-step goal? ────────────────────────────
        if Flags.PLANNER_ENABLED and self.planner.is_goal_request(user_input):
            plan = self.planner.decompose(user_input)
            plan_display = plan.to_display()
            # Inject plan into context for LLM
            user_input_with_plan = (
                f"{user_input}\n\n[Execution plan created:]\n{plan_display}"
            )
        else:
            user_input_with_plan = user_input

        # ── Memory retrieval ──────────────────────────────────────────────────
        memories  = self.vector_store.search(user_input, top_k=3, role="user")
        history   = self.buffer.get_history()
        ctx_frag  = snap.to_prompt_fragment()
        system    = self._build_system_prompt(memories, ctx_frag)

        # ── Dual brain mode ───────────────────────────────────────────────────
        if Flags.DUAL_BRAIN and self._is_dual_request(user_input):
            async for tok in self._dual_brain_stream(system, history, user_input):
                yield tok
            full_response = ""   # captured in dual_brain_stream memory
            return

        # ── LLM streaming ─────────────────────────────────────────────────────
        full_response = ""
        async for tok in self.selector.stream_complete(
            system, history, user_input_with_plan, intent
        ):
            full_response += tok
            yield tok

        log.info("Stream complete, %d chars", len(full_response))

        # ── Memory update ─────────────────────────────────────────────────────
        self._update_memory(user_input, full_response)

        # ── Self-improvement logging ──────────────────────────────────────────
        if Flags.ACTION_LOGGING:
            ms = int((time.monotonic() - t0) * 1000)
            self.long_term.log_action(
                tool="llm", input_text=user_input[:200],
                outcome=full_response[:200], success=True, duration_ms=ms
            )

    # ── Dual brain ────────────────────────────────────────────────────────────


    def _handle_system_query(self, text: str) -> str:
        """Handle AIDA meta-queries without an LLM call."""
        import re
        lower = text.lower()

        if re.search(r"\b(mode|режим)\b", lower):
            m = self.modes.current
            return (f"Current mode: **{m.emoji} {m.label}**\n"
                    f"Available: {', '.join(self.modes.mode_names())}")

        if re.search(r"\b(plan|план)\b", lower):
            if self.planner.has_active_plan:
                return self.planner.current_plan.to_display()
            return "No active plan. Say 'help me [goal]' to start one."

        if re.search(r"\b(memory|память|know about me)\b", lower):
            facts = self.long_term.recent_facts(5)
            prefs = self.long_term.all_preferences()
            parts = []
            if prefs:
                parts.append("**Preferences:** " + ", ".join(f"{k}={v}" for k,v in list(prefs.items())[:5]))
            if facts:
                parts.append("**Recent facts:**\n" + "\n".join(f"- {f['content'][:60]}" for f in facts))
            return "\n".join(parts) if parts else "No long-term memories yet."

        if re.search(r"\b(shadow|подсказки|suggestions)\b", lower):
            if self.shadow.enabled:
                insights = self.shadow.generate_insights()
                return ("Shadow Mode: ON\n" +
                        ("\n".join(f"- {i}" for i in insights) if insights
                         else "Not enough data yet (needs ~3+ sessions)."))
            return "Shadow Mode: OFF. Enable it in the Brain panel."

        if re.search(r"\b(status|статус)\b", lower):
            groq_ok  = self.selector.groq.is_available()
            local_ok = self.selector.local.is_available()
            mem      = self.vector_store.count()
            mode     = self.modes.current.label
            return (f"**AIDA Status**\n"
                    f"Mode: {mode} | "
                    f"LLM: {'Groq ✓' if groq_ok else 'Local' if local_ok else '⚠ none'} | "
                    f"Memory: {mem} vectors")

        # Generic: return current context
        snap = self.context.snapshot()
        return (f"Mode: {self.modes.current.emoji} {self.modes.current.label} | "
                f"App: {snap.process_name or 'unknown'} | "
                f"Plan: {'active' if self.planner.has_active_plan else 'none'}")

    def _is_dual_request(self, text: str) -> bool:
        import re
        patterns = [
            r"\b(pros and cons|both sides|tradeoff|trade-off|two perspectives)\b",
            r"\b(analytical.*creative|creative.*analytical|brainstorm)\b",
            r"\b(compare|weigh|evaluate|should i)\b",
        ]
        lower = text.lower()
        return any(re.search(p, lower) for p in patterns)

    async def _dual_brain_stream(self, system: str, history: list,
                                  user_input: str):
        """Yields analytical response then creative response, clearly labelled."""
        analytical_prompt = (
            system +
            "\n\n[ANALYTICAL BRAIN MODE: Be precise, structured, evidence-based. "
            "Use data and logic. Identify risks and tradeoffs clearly.]"
        )
        creative_prompt = (
            system +
            "\n\n[CREATIVE BRAIN MODE: Be imaginative, lateral, generative. "
            "Explore unconventional angles. Challenge assumptions.]"
        )
        yield "**🔬 Analytical perspective:**\n"
        analytical = ""
        async for tok in self.selector.stream_complete(
            analytical_prompt, history, user_input, "conversation"
        ):
            analytical += tok
            yield tok

        yield "\n\n**🎨 Creative perspective:**\n"
        creative = ""
        async for tok in self.selector.stream_complete(
            creative_prompt, history, user_input, "conversation"
        ):
            creative += tok
            yield tok

        combined = f"[Analytical]\n{analytical}\n\n[Creative]\n{creative}"
        self._update_memory(user_input, combined)

    # ── Memory helpers ────────────────────────────────────────────────────────

    def _update_memory(self, user_input: str, response: str):
        self.buffer.add(role="user",      content=user_input)
        self.buffer.add(role="assistant", content=response)
        self.vector_store.add(user_input, metadata={"role": "user"})
        self.vector_store.add(response,   metadata={"role": "assistant"})
        if Flags.EPISODIC_MEMORY:
            self.episodic.log_event(
                event_type="exchange",
                content=user_input,
                metadata={"response": response[:200],
                          "mode": self.modes.current_name},
            )

    # ── Convenience ───────────────────────────────────────────────────────────

    def get_suggestions(self, recent_transcript: str = "") -> list:
        if not Flags.PREDICTIVE_SUGGESTIONS:
            return []
        snap = self.context.last_snapshot
        return self.predictive.generate(
            snapshot           = snap,
            recent_transcript  = recent_transcript,
            current_mode       = self.modes.current_name,
        )

    def get_goal_status(self) -> str:
        """For UI polling."""
        return self.goal_engine.status()

    def stop(self):
        self._running = False
