"""
core/personality.py
Builds the system prompt that defines AIDA's personality and behaviour.
Reads base personality from config/prompts.yaml, injects live memories.
"""
import os
import yaml
from typing import List, Optional

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "prompts.yaml")

DEFAULT_PERSONALITY = """You are AIDA (AI Desktop Assistant) — a highly capable, proactive personal assistant.
You are concise, warm, and precise. You speak in the language the user uses.
You remember past conversations and use that context to be more helpful.
You have access to tools: you can run commands, search the web, manage files.
When you need more information before acting, you ask one focused question.
Never pretend you did something if you didn't. Be honest about your limitations."""


class Personality:
    def __init__(self):
        self._base = self._load_config()

    def _load_config(self) -> str:
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data.get("system_prompt", DEFAULT_PERSONALITY)
        except FileNotFoundError:
            return DEFAULT_PERSONALITY

    def build_system_prompt(self, memories: Optional[List[str]] = None) -> str:
        prompt = self._base
        if memories:
            mem_block = "\n".join(f"- {m}" for m in memories)
            prompt += f"\n\n[Relevant memories from past conversations:]\n{mem_block}"
        return prompt
