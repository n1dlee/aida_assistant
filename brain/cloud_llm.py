"""
brain/cloud_llm.py
Cloud model wrapper. Tries Gemini first, falls back to OpenAI.
Set API keys in .env or environment variables:
  GEMINI_API_KEY=...
  OPENAI_API_KEY=...
"""
import logging
import os
from typing import List, Dict

log = logging.getLogger("aida.brain.cloud")


class CloudLLM:
    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY", "")
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.model_name = "gemini-2.0-flash" if self.gemini_key else "gpt-4o-mini"

    def is_available(self) -> bool:
        return bool(self.gemini_key or self.openai_key)

    async def complete(
        self, system: str, history: List[Dict], user_input: str
    ) -> str:
        if self.gemini_key:
            return await self._gemini(system, history, user_input)
        if self.openai_key:
            return await self._openai(system, history, user_input)
        raise RuntimeError("No cloud API key configured.")

    async def _gemini(self, system: str, history: List[Dict], user_input: str) -> str:
        import asyncio
        import google.generativeai as genai

        genai.configure(api_key=self.gemini_key)
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system,
        )

        # Build Gemini-format history
        gem_history = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            gem_history.append({"role": role, "parts": [msg["content"]]})

        def _sync():
            chat = model.start_chat(history=gem_history)
            return chat.send_message(user_input).text

        return await asyncio.get_running_loop().run_in_executor(None, _sync)

    async def _openai(self, system: str, history: List[Dict], user_input: str) -> str:
        import asyncio
        from openai import OpenAI

        client = OpenAI(api_key=self.openai_key)
        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        def _sync():
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
            )
            return resp.choices[0].message.content

        return await asyncio.get_running_loop().run_in_executor(None, _sync)
