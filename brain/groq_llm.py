"""
brain/groq_llm.py
──────────────────
Groq cloud inference — free tier, ~600 tokens/sec.
Uses the openai package (already installed) with Groq's OpenAI-compatible API.

Setup:
  1. Get free key at https://console.groq.com  (takes 30 seconds)
  2. Set environment variable:  GROQ_API_KEY=gsk_...
  3. Restart AIDA — Groq becomes the primary LLM automatically.

Free tier limits (more than enough for personal use):
  - 14,400 requests / day
  - 6,000 tokens / minute
  - 131,072 context window (llama-3.3-70b)

Models (set AIDA_GROQ_MODEL env var to override):
  llama-3.3-70b-versatile  — best quality, default
  llama3-8b-8192           — fastest, lowest latency
  gemma2-9b-it             — Google Gemma 2 9B
"""

import logging
import os
from typing import List, Dict

log = logging.getLogger("aida.brain.groq")

_DEFAULT_MODEL  = "llama-3.3-70b-versatile"
_GROQ_BASE_URL  = "https://api.groq.com/openai/v1"


class GroqLLM:
    def __init__(self):
        self.api_key    = os.getenv("GROQ_API_KEY", "")
        self.model_name = os.getenv("AIDA_GROQ_MODEL", _DEFAULT_MODEL)
        self._client    = None

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key  = self.api_key,
                base_url = _GROQ_BASE_URL,
            )
        return self._client

    async def complete(self, system: str, history: List[Dict], user_input: str) -> str:
        full = ""
        async for tok in self.stream_complete(system, history, user_input):
            full += tok
        return full

    async def stream_complete(self, system: str, history: List[Dict], user_input: str):
        """Async generator — yields tokens as they arrive from Groq."""
        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        client = self._get_client()
        log.debug("Groq request: model=%s msgs=%d", self.model_name, len(messages))

        stream = await client.chat.completions.create(
            model       = self.model_name,
            messages    = messages,
            stream      = True,
            temperature = 0.7,
            max_tokens  = 1024,
        )
        async for chunk in stream:
            tok = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            if tok:
                yield tok
