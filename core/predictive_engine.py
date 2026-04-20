"""
core/predictive_engine.py
──────────────────────────
Generates context-aware next-action suggestions.

Sources:
  - shadow engine patterns (app transitions, tool frequency)
  - long-term memory (preferences, recurring tasks)
  - current context (active app, time of day, mode)
  - recent conversation (what was just discussed)

Suggestions are SHOWN to the user — never silently executed.
User explicitly accepts or dismisses each suggestion.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.shadow_engine import ShadowEngine
    from core.context_manager import ContextSnapshot
    from memory.long_term import LongTermMemory

log = logging.getLogger("aida.predictive")


class Suggestion:
    def __init__(self, text: str, action_type: str,
                 confidence: float, source: str):
        self.text        = text
        self.action_type = action_type   # "open_app" | "tool" | "remind" | "plan"
        self.confidence  = confidence    # 0.0–1.0
        self.source      = source        # where it came from
        self.timestamp   = datetime.now().isoformat()

    def __repr__(self) -> str:
        return f"Suggestion({self.text!r}, conf={self.confidence:.2f})"


class PredictiveEngine:
    def __init__(self, shadow: "ShadowEngine",
                 long_term: "LongTermMemory",
                 min_confidence: float = 0.4):
        self._shadow       = shadow
        self._long_term    = long_term
        self._min_conf     = min_confidence
        self._last_suggestions: list[Suggestion] = []

    def generate(self,
                 snapshot: Optional["ContextSnapshot"],
                 recent_transcript: str = "",
                 current_mode: str = "default") -> list[Suggestion]:
        """
        Generate suggestions based on current context.
        Returns at most 3 suggestions above the confidence threshold.
        """
        candidates: list[Suggestion] = []

        # ── Pattern-based: app transitions ───────────────────────────────────
        if self._shadow.enabled and snapshot:
            current_app = snapshot.process_name
            for pair, count in self._shadow.top_app_transitions(10):
                if "→" not in pair:
                    continue
                from_app, to_app = pair.split("→")
                if from_app == current_app and count >= 3:
                    conf = min(0.9, 0.4 + count * 0.05)
                    candidates.append(Suggestion(
                        text        = f"You usually open {to_app} after {from_app}. Open it?",
                        action_type = "open_app",
                        confidence  = conf,
                        source      = "shadow_pattern",
                    ))

        # ── Time-based patterns ───────────────────────────────────────────────
        hour = datetime.now().hour
        if 8 <= hour <= 10:
            top_tools = self._shadow.top_tools(3)
            for tool, count in top_tools:
                if count >= 5:
                    candidates.append(Suggestion(
                        text        = f"Morning workflow: shall I run {tool} for you?",
                        action_type = "tool",
                        confidence  = 0.5,
                        source      = "time_pattern",
                    ))

        # ── Conversation-based ────────────────────────────────────────────────
        if recent_transcript:
            low = recent_transcript.lower()
            if any(w in low for w in ["file", "folder", "directory", "папка", "файл"]):
                candidates.append(Suggestion(
                    text        = "Want me to list the files in your current folder?",
                    action_type = "tool",
                    confidence  = 0.55,
                    source      = "conversation",
                ))
            if any(w in low for w in ["search", "find", "look up", "найди"]):
                candidates.append(Suggestion(
                    text        = "Shall I search the web for more on this?",
                    action_type = "tool",
                    confidence  = 0.6,
                    source      = "conversation",
                ))

        # ── Mode-based ────────────────────────────────────────────────────────
        if current_mode == "study":
            candidates.append(Suggestion(
                text        = "Want me to create a quick quiz on what we just covered?",
                action_type = "plan",
                confidence  = 0.5,
                source      = "mode",
            ))
        elif current_mode == "coding":
            candidates.append(Suggestion(
                text        = "Shall I search for documentation or examples?",
                action_type = "tool",
                confidence  = 0.5,
                source      = "mode",
            ))

        # Filter and sort
        filtered = [s for s in candidates if s.confidence >= self._min_conf]
        filtered.sort(key=lambda s: -s.confidence)
        result = filtered[:3]
        self._last_suggestions = result
        return result

    @property
    def last_suggestions(self) -> list[Suggestion]:
        return self._last_suggestions
