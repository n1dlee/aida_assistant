"""
core/orchestrator.py
Central coordinator: receives input, builds context, routes to brain, executes actions.
"""
import asyncio
import logging
from typing import Optional

from brain.selector import ModelSelector
from memory.buffer import ConversationBuffer
from memory.vector_store import VectorStore
from core.personality import Personality
from core.router import IntentRouter
from tools.registry import ToolRegistry

log = logging.getLogger("aida.orchestrator")


class Orchestrator:
    def __init__(self):
        self.selector = ModelSelector()
        self.buffer = ConversationBuffer(max_turns=20)
        self.vector_store = VectorStore()
        self.personality = Personality()
        self.router = IntentRouter()
        self.tools = ToolRegistry()
        self._running = False

    async def start(self):
        """Main loop — listens for input and processes it."""
        self._running = True
        log.info("AIDA online. Type your message (or 'quit' to exit).")

        # In a full build, voice/listener.py feeds here via asyncio queue.
        # For now: text input loop so you can test immediately.
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
        """Full pipeline: input → context → model → (tools?) → response."""
        log.debug("Processing: %s", user_input)

        # 1. Detect intent
        intent = self.router.classify(user_input)
        log.info("Intent: %s", intent)

        # 2. Retrieve relevant memories
        memories = self.vector_store.search(user_input, top_k=3)

        # 3. Build context
        history = self.buffer.get_history()
        system_prompt = self.personality.build_system_prompt(memories=memories)

        # 4. Select model and get response
        response, model_used = await self.selector.complete(
            system=system_prompt,
            history=history,
            user_input=user_input,
            intent=intent,
        )
        log.info("Model used: %s", model_used)

        # 5. Execute tool calls if needed
        if intent == "tool_call":
            tool_result = await self.tools.execute(user_input, response)
            if tool_result:
                response = tool_result

        # 6. Update memory
        self.buffer.add(role="user", content=user_input)
        self.buffer.add(role="assistant", content=response)
        self.vector_store.add(user_input, metadata={"role": "user"})
        self.vector_store.add(response, metadata={"role": "assistant"})

        return response

    def stop(self):
        self._running = False
