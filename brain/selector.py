"""
brain/selector.py
Decides whether to use a local Ollama model or a cloud API (Gemini / OpenAI).
Routing logic based on task complexity, context length, and config thresholds.
"""
import logging
import os
import time
from typing import Tuple, List, Dict

from brain.local_llm import LocalLLM
from brain.cloud_llm import CloudLLM

log = logging.getLogger("aida.brain.selector")

# Complexity keywords that push toward cloud
CLOUD_KEYWORDS = [
    "analyze", "анализ", "compare", "explain in detail", "write code",
    "debug", "translate", "summarize this document", "plan", "strategy",
    "объясни подробно", "напиши код", "проанализируй", "сравни",
]

# Token threshold — if conversation history is long, cloud handles better
CLOUD_CONTEXT_THRESHOLD = 2000

# Max score contribution from keyword matching (prevents runaway accumulation)
_KEYWORD_SCORE_CAP = 0.4


class ModelSelector:
    def __init__(self):
        self.local = LocalLLM()
        self.cloud = CloudLLM()
        self._avail_cache: dict[str, tuple[bool, float]] = {}
        self._avail_ttl = 10.0   # seconds between availability checks

    def _is_available_cached(self, provider, name: str) -> bool:
        now = time.monotonic()
        cached = self._avail_cache.get(name)
        if cached and now - cached[1] < self._avail_ttl:
            return cached[0]
        result = provider.is_available()
        self._avail_cache[name] = (result, now)
        return result

    def _complexity_score(self, user_input: str, history: List[Dict]) -> float:
        """Returns 0.0 (simple) → 1.0 (complex)."""
        score = 0.0
        lower = user_input.lower()

        # Keyword complexity — capped so many keywords don't bypass min()
        keyword_hits = sum(1 for kw in CLOUD_KEYWORDS if kw in lower)
        score += min(_KEYWORD_SCORE_CAP, keyword_hits * 0.2)

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
        Tries preferred model first, then automatically falls back
        to the other configured provider.
        """
        use_cloud = self._should_use_cloud(user_input, history, intent)

        providers = []
        if use_cloud:
            providers = [("cloud", self.cloud), ("local", self.local)]
        else:
            providers = [("local", self.local), ("cloud", self.cloud)]

        errors = []
        for name, provider in providers:
            if not self._is_available_cached(provider, name):
                log.debug("%s provider is not available.", name)
                continue
            try:
                response = await provider.complete(system, history, user_input)
                return response, provider.model_name
            except Exception as e:
                errors.append((name, str(e)))
                log.warning("%s LLM failed: %s", name.capitalize(), e)

        if errors:
            log.error("All available providers failed: %s", errors)
            first_provider, first_error = errors[0]
            return (
                f"{first_provider.capitalize()} model failed: {first_error}. "
                "Try a smaller local model (e.g. 3b/7b), or enable a cloud API key.",
                "none",
            )

        return "Sorry, no AI model is currently available. Please check your config.", "none"

    async def stream_complete(
        self,
        system: str,
        history: List[Dict],
        user_input: str,
        intent: str = "conversation",
    ):
        """Async generator: yields tokens, preferring local unless complex."""
        use_cloud = self._should_use_cloud(user_input, history, intent)
        pairs = [("cloud", self.cloud), ("local", self.local)] if use_cloud                 else [("local", self.local), ("cloud", self.cloud)]

        for name, provider in pairs:
            if not self._is_available_cached(provider, name):
                continue
            if not hasattr(provider, "stream_complete"):
                # Non-streaming fallback
                try:
                    resp = await provider.complete(system, history, user_input)
                    yield resp
                    return
                except Exception as exc:
                    log.warning("%s non-stream failed: %s", name, exc)
                    continue
            try:
                async for tok in provider.stream_complete(system, history, user_input):
                    yield tok
                return
            except Exception as exc:
                log.warning("%s stream failed: %s", name, exc)

        yield "Sorry, no AI model is available."

