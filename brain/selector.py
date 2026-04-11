"""
brain/selector.py
Decides whether to use a local Ollama model or a cloud API (Gemini / OpenAI).
Routing logic based on task complexity, context length, and config thresholds.
"""
import logging
import os
from typing import Tuple, List, Dict

from brain.local_llm import LocalLLM
from brain.cloud_llm import CloudLLM

log = logging.getLogger("aida.brain.selector")

# Complexity keywords that push toward cloud
CLOUD_KEYWORDS = [
    "analyze", "аналiz", "compare", "explain in detail", "write code",
    "debug", "translate", "summarize this document", "plan", "strategy",
    "объясни подробно", "напиши код", "проанализируй", "сравни",
]

# Token threshold — if conversation history is long, cloud handles better
CLOUD_CONTEXT_THRESHOLD = 2000


class ModelSelector:
    def __init__(self):
        self.local = LocalLLM()
        self.cloud = CloudLLM()

    def _complexity_score(self, user_input: str, history: List[Dict]) -> float:
        """Returns 0.0 (simple) → 1.0 (complex)."""
        score = 0.0
        lower = user_input.lower()

        # Keyword complexity
        for kw in CLOUD_KEYWORDS:
            if kw in lower:
                score += 0.3

        # Length of input
        if len(user_input) > 200:
            score += 0.2

        # History length
        total_chars = sum(len(m.get("content", "")) for m in history)
        if total_chars > CLOUD_CONTEXT_THRESHOLD:
            score += 0.2

        return min(score, 1.0)

    def _should_use_cloud(self, user_input: str, history: List[Dict], intent: str) -> bool:
        if intent == "tool_call":
            return False  # Tools are handled locally; LLM just parses intent
        score = self._complexity_score(user_input, history)
        threshold = float(os.getenv("AIDA_CLOUD_THRESHOLD", "0.4"))
        log.debug("Complexity score: %.2f (threshold: %.2f)", score, threshold)
        return score >= threshold

    async def complete(
        self,
        system: str,
        history: List[Dict],
        user_input: str,
        intent: str = "conversation",
    ) -> Tuple[str, str]:
        """
        Returns (response_text, model_name_used).
        Falls back to cloud if local fails.
        """
        use_cloud = self._should_use_cloud(user_input, history, intent)

        if use_cloud and self.cloud.is_available():
            try:
                response = await self.cloud.complete(system, history, user_input)
                return response, self.cloud.model_name
            except Exception as e:
                log.warning("Cloud LLM failed (%s), falling back to local.", e)

        # Local
        if self.local.is_available():
            try:
                response = await self.local.complete(system, history, user_input)
                return response, self.local.model_name
            except Exception as e:
                log.error("Local LLM also failed: %s", e)

        return "Sorry, no AI model is currently available. Please check your config.", "none"
