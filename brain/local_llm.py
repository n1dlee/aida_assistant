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
                # Some ollama-python builds return model objects in a shape that is not
                # fully stable across versions. If Ollama is reachable, treat local as
                # available and let chat() surface the exact runtime error instead of
                # incorrectly reporting "no model available".
                log.warning(
                    "Ollama reachable at %s, but model list could not be parsed.",
                    self.base_url,
                )
                return True

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

    async def stream_complete(
        self, system: str, history: list, user_input: str
    ):
        """Async generator — yields string tokens as they stream from Ollama."""
        import asyncio, queue as _q

        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        client  = self._get_client()
        q: "_q.Queue[object]" = _q.Queue()
        _DONE   = object()

        def _stream_thread():
            try:
                for chunk in client.chat(
                    model=self.model_name,
                    messages=messages,
                    stream=True,
                    options={"temperature": 0.7, "num_ctx": 4096},
                ):
                    tok = chunk.message.content or ""
                    if tok:
                        q.put(tok)
            except Exception as exc:
                q.put(exc)
            finally:
                q.put(_DONE)

        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, _stream_thread)

        while True:
            try:
                item = q.get_nowait()
            except _q.Empty:
                await asyncio.sleep(0.008)
                continue
            if item is _DONE:
                break
            if isinstance(item, Exception):
                raise item
            yield str(item)

