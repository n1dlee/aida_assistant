"""
core/router.py
───────────────
Classifies user intent to guide orchestrator dispatch.

Intents (priority order):
  system_query   — AIDA meta-queries (status, mode, memory, plan)
  memory_query   — asking about past conversations or facts
  goal_request   — multi-step goal that needs planning
  dual_brain     — explicit request for two perspectives
  tool_call      — single tool invocation
  proactive_setup — scheduling / automation setup
  conversation   — default fallback
"""

import re
from typing import Optional

INTENT_PATTERNS: dict[str, list[str]] = {

    # ── AIDA self-queries ─────────────────────────────────────────────────────
    "system_query": [
        r"\b(what mode|current mode|switch mode|change mode|режим)\b",
        r"\b(what do you know|your memory|your status|твоя память)\b",
        r"\b(show plan|current plan|next step|покажи план)\b",
        r"\b(shadow mode|predictive|suggestions|подсказки)\b",
        r"\b(clear (memory|history|plan))\b",
    ],

    # ── Memory retrieval ──────────────────────────────────────────────────────
    "memory_query": [
        r"\b(remember|recall|last time|previously|did i|have i)\b",
        r"\b(what did (i|we)|when did|you said|told you)\b",
        r"\b(помни|помнишь|в прошлый раз|прошлый раз|ты говорил)\b",
    ],

    # ── Goal / planning ───────────────────────────────────────────────────────
    "goal_request": [
        r"\b(help me (build|create|set up|make|plan|start|organize|study))\b",
        r"\b(i want to (build|create|make|start|learn|set up))\b",
        r"\b(create (a |an )?(project|workflow|system|plan|study|sprint))\b",
        r"\b(step by step|walk me through|помоги мне|хочу создать|помоги настроить)\b",
    ],

    # ── Dual brain / two perspectives ────────────────────────────────────────
    "dual_brain": [
        r"\b(pros and cons|two (sides|perspectives)|both sides|трейдофф)\b",
        r"\b(analytical.*creative|creative.*analytical|две точки зрения)\b",
        r"\b(brainstorm|compare options|weigh the|should i .{5,40}\?)\b",
    ],

    # ── Tool dispatch ─────────────────────────────────────────────────────────
    "tool_call": [
        r"\b(open|run|execute|launch|search|find|create|delete|move|copy|rename|list|read|write)\b",
        r"\b(folder|directory|file|папка|файл|директория)\b",
        r"\b(set (alarm|timer|reminder)|weather|time|date)\b",
        r"\b(play|pause|stop|volume|calendar|event|schedule)\b",
        r"\b(найди|создай|удали|открой|запусти|поиск)\b",
    ],

    # ── Proactive / scheduling ────────────────────────────────────────────────
    "proactive_setup": [
        r"\b(every day|every hour|remind me|notify me|when .+ happens)\b",
        r"\b(schedule|automate|watch for|напомни|каждый день)\b",
    ],

    # conversation is the default — no patterns needed
    "conversation": [],
}

# Intent priority order (first match wins)
_PRIORITY: list[str] = [
    "system_query",
    "memory_query",
    "goal_request",
    "dual_brain",
    "tool_call",
    "proactive_setup",
    "conversation",
]


class IntentRouter:
    def classify(self, text: str) -> str:
        lower = text.lower()
        for intent in _PRIORITY:
            patterns = INTENT_PATTERNS.get(intent, [])
            for pattern in patterns:
                if re.search(pattern, lower):
                    return intent
        return "conversation"

    def is_goal(self, text: str) -> bool:
        return self.classify(text) == "goal_request"

    def is_dual(self, text: str) -> bool:
        return self.classify(text) == "dual_brain"

    def is_system(self, text: str) -> bool:
        return self.classify(text) == "system_query"
