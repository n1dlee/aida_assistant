"""
core/router.py
Classifies user intent to help orchestrator decide next action.
"""
import re

INTENT_PATTERNS = {
    "memory_query": [
        r"\b(remember|recall|last time|previously|did i|have i)\b",
        r"\b(what did (i|we)|when did)\b",
    ],
    "proactive_setup": [
        r"\b(every day|every hour|remind me|notify me|when .+ happens)\b",
        r"\b(schedule|automate|watch for)\b",
    ],
    "tool_call": [
        r"\b(open|run|execute|launch|search|find|create|delete|move|copy)\b",
        r"\b(set (alarm|timer|reminder))\b",
        r"\b(weather|time|date)\b",
        r"\b(play|pause|stop|volume)\b",
    ],
    "conversation": [],  # default fallback
}


class IntentRouter:
    def classify(self, text: str) -> str:
        lower = text.lower()
        for intent, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, lower):
                    return intent
        return "conversation"
