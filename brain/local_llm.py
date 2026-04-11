"""
brain/local_llm.py
Wrapper for local Ollama inference.
Requires Ollama running: https://ollama.ai
Default model: llama3.1:8b (change in config/settings.yaml)
"""
import logging
import os
from typing import List, Dict

log = logging.getLogger("aida.brain.local")


class LocalLLM:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = os.getenv("AIDA_LOCAL_MODEL", "gemma4:31b")
        self._client = None

    def is_available(self) -> bool:
        try:
            import ollama
            client = ollama.Client(host=self.base_url)
            models_resp = client.list()

            # Some ollama versions return dict-like payloads,
            # others return pydantic models with attributes.
            models = []
            if isinstance(models_resp, dict):
                models = models_resp.get("models", []) or []
            else:
                models = getattr(models_resp, "models", []) or []

            names = []
            for model in models:
                if isinstance(model, dict):
                    names.append(model.get("name") or model.get("model"))
                else:
                    names.append(getattr(model, "name", None) or getattr(model, "model", None))

            names = [n for n in names if n]
            if not names:
                log.warning("Ollama reachable, but no models reported at %s", self.base_url)
                return False

            if self.model_name not in names:
                log.warning(
                    "Configured model '%s' is not available in Ollama. Available: %s",
                    self.model_name,
                    names,
                )
                return False

            return True
        except Exception as e:
            log.warning("Local Ollama is unavailable at %s: %s", self.base_url, e)
            return False

    def _get_client(self):
        if self._client is None:
            import ollama
            self._client = ollama.Client(host=self.base_url)
        return self._client

    async def complete(
        self, system: str, history: List[Dict], user_input: str
    ) -> str:
        import asyncio

        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        client = self._get_client()

        def _sync_call():
            response = client.chat(
                model=self.model_name,
                messages=messages,
                options={"temperature": 0.7, "num_ctx": 4096},
            )
            # ChatResponse is a SubscriptableBaseModel (Pydantic), NOT a plain dict.
            # Use attribute access — works in all ollama versions and satisfies Pylance.
            return response.message.content

        return await asyncio.get_running_loop().run_in_executor(None, _sync_call)
