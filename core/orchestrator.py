"""
core/orchestrator.py
Central coordinator: receives input, builds context, routes to brain, executes actions.

Fixes applied:
- EpisodicMemory integrated: logs every user+assistant exchange
- Tool dispatch now runs BEFORE the LLM for pure tool intents, avoiding
  wasted LLM inference that would be immediately overwritten
- VectorStore.search now filters by role="user" so assistant responses
  don't pollute memory retrieval
"""
import asyncio
import logging
from typing import Optional

from brain.selector import ModelSelector
from memory.buffer import ConversationBuffer
from memory.vector_store import VectorStore
from memory.episodic import EpisodicMemory
from core.personality import Personality
from core.router import IntentRouter
from tools.registry import ToolRegistry

log = logging.getLogger("aida.orchestrator")


class Orchestrator:
    def __init__(self):
        self.selector    = ModelSelector()
        self.buffer      = ConversationBuffer(max_turns=20)
        self.vector_store = VectorStore()
        self.episodic    = EpisodicMemory()
        self.personality = Personality()
        self.router      = IntentRouter()
        self.tools       = ToolRegistry()
        self._running    = False

    async def start(self):
        """Main loop — listens for input and processes it."""
        self._running = True
        log.info("AIDA online. Type your message (or 'quit' to exit).")

        loop = asyncio.get_running_loop()
        while self._running:
            try:
                user_input = await loop.run_in_executor(None, input, "\nYou: ")
            except (EOFError, KeyboardInterrupt):
                break

            if user_input.strip().lower() in ("quit", "exit", "q"):
                log.info("Shutting down.")
                break

            if not user_input.strip():
                continue

            response = await self.process(user_input)
            print(f"\nAIDA: {response}")

    async def process(self, user_input: str) -> str:
        """Full pipeline: input → context → (tool shortcut?) → model → response."""
        log.debug("Processing: %s", user_input)

        # 1. Detect intent
        intent = self.router.classify(user_input)
        log.info("Intent: %s", intent)

        # 2. Try tool dispatch FIRST for tool intents — skip LLM if tool answers
        if intent == "tool_call":
            tool_result = await self.tools.execute(user_input, llm_response="")
            if tool_result:
                self._update_memory(user_input, tool_result)
                return tool_result

        # 3. Retrieve relevant user-side memories (not polluted by assistant text)
        memories = self.vector_store.search(user_input, top_k=3, role="user")

        # 4. Build context
        history       = self.buffer.get_history()
        system_prompt = self.personality.build_system_prompt(memories=memories)

        # 5. Select model and get response
        response, model_used = await self.selector.complete(
            system=system_prompt,
            history=history,
            user_input=user_input,
            intent=intent,
        )
        log.info("Model used: %s", model_used)

        # 6. Update memory
        self._update_memory(user_input, response)
        return response

    def _update_memory(self, user_input: str, response: str):
        """Persist exchange to short-term buffer, semantic store, and episodic log."""
        self.buffer.add(role="user",      content=user_input)
        self.buffer.add(role="assistant", content=response)

        # Store with explicit role so VectorStore.search(role=...) works
        self.vector_store.add(user_input, metadata={"role": "user"})
        self.vector_store.add(response,   metadata={"role": "assistant"})

        # Episodic log — timestamped record of every exchange
        self.episodic.log_event(
            event_type="exchange",
            content=user_input,
            metadata={"response": response[:200]},
        )


    async def stream_process(self, user_input: str):
        """Async generator: yields LLM tokens for streaming UI display."""
        intent = self.router.classify(user_input)
        log.info("Intent (stream): %s", intent)

        # Tools answer instantly — no streaming needed
        if intent == "tool_call":
            result = await self.tools.execute(user_input, llm_response="")
            if result:
                self._update_memory(user_input, result)
                yield result
                return

        memories     = self.vector_store.search(user_input, top_k=3, role="user")
        history      = self.buffer.get_history()
        system_prompt = self.personality.build_system_prompt(memories=memories)

        full_response = ""
        model_used    = "unknown"
        async for tok in self.selector.stream_complete(
            system_prompt, history, user_input, intent
        ):
            full_response += tok
            yield tok

        log.info("Stream complete, %d chars", len(full_response))
        self._update_memory(user_input, full_response)
    def stop(self):
        self._running = False
