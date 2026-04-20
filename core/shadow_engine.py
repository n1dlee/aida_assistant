"""
core/shadow_engine.py
──────────────────────
Shadow Mode: opt-in passive workflow observer.

Logs structured events (NOT raw screen/keystrokes) and detects
recurring patterns that AIDA can later surface as suggestions.

Privacy constraints:
  - Disabled by default (feature_flags.SHADOW_MODE = False)
  - Logs only: app names, intent types, time-of-day, tool usage
  - No screen capture, no keystroke logging, no file content
  - All data is local, never transmitted
  - User can clear at any time

Pattern detection (simple):
  - "App A often followed by App B"
  - "Tool X used at similar times"
  - "Intent type Y frequent on day Z"
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("aida.shadow")

_LOG_PATH = Path(__file__).parent.parent / "data" / "shadow_log.jsonl"


class ShadowEvent:
    def __init__(self, event_type: str, data: dict):
        self.event_type = event_type
        self.data       = data
        self.timestamp  = datetime.now().isoformat()
        self.hour       = datetime.now().hour
        self.weekday    = datetime.now().weekday()

    def to_dict(self) -> dict:
        return {
            "type":      self.event_type,
            "data":      self.data,
            "timestamp": self.timestamp,
            "hour":      self.hour,
            "weekday":   self.weekday,
        }


class ShadowEngine:
    def __init__(self, enabled: bool = False):
        self._enabled      = enabled
        self._recent:     deque[ShadowEvent] = deque(maxlen=200)
        self._app_seq:    deque[str]         = deque(maxlen=20)
        self._patterns:   dict               = defaultdict(int)

        if enabled:
            _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            log.info("Shadow Mode ENABLED — logging to %s", _LOG_PATH)
        else:
            log.debug("Shadow Mode disabled.")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log.info("Shadow Mode enabled by user.")

    def disable(self) -> None:
        self._enabled = False
        log.info("Shadow Mode disabled by user.")

    def clear(self) -> None:
        self._recent.clear()
        self._patterns.clear()
        if _LOG_PATH.exists():
            _LOG_PATH.unlink()
        log.info("Shadow log cleared.")

    # ── Event logging ─────────────────────────────────────────────────────────

    def log_app_switch(self, app_name: str) -> None:
        if not self._enabled or not app_name:
            return
        # Track transition patterns
        if self._app_seq:
            pair = f"{self._app_seq[-1]}→{app_name}"
            self._patterns[pair] += 1
        self._app_seq.append(app_name)
        self._log(ShadowEvent("app_switch", {"app": app_name}))

    def log_intent(self, intent: str, mode: str) -> None:
        if not self._enabled:
            return
        self._log(ShadowEvent("intent", {"intent": intent, "mode": mode}))

    def log_tool_use(self, tool: str, success: bool) -> None:
        if not self._enabled:
            return
        key = f"tool:{tool}"
        self._patterns[key] += 1
        self._log(ShadowEvent("tool_use", {"tool": tool, "success": success}))

    def log_voice_session(self, duration_s: float) -> None:
        if not self._enabled:
            return
        self._log(ShadowEvent("voice_session", {"duration_s": duration_s}))

    def _log(self, event: ShadowEvent) -> None:
        self._recent.append(event)
        try:
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        except Exception as exc:
            log.debug("Shadow log write error: %s", exc)

    # ── Pattern detection ─────────────────────────────────────────────────────

    def top_app_transitions(self, n: int = 5) -> list[tuple[str, int]]:
        """Returns most common app→app transitions."""
        transitions = {
            k: v for k, v in self._patterns.items()
            if "→" in k
        }
        return sorted(transitions.items(), key=lambda x: -x[1])[:n]

    def top_tools(self, n: int = 5) -> list[tuple[str, int]]:
        tools = {
            k.replace("tool:", ""): v for k, v in self._patterns.items()
            if k.startswith("tool:")
        }
        return sorted(tools.items(), key=lambda x: -x[1])[:n]

    def generate_insights(self) -> list[str]:
        """
        Produces human-readable pattern observations.
        Returns empty list if not enough data.
        """
        insights = []
        transitions = self.top_app_transitions(3)
        for pair, count in transitions:
            if count >= 3:
                a, b = pair.split("→")
                insights.append(f"You often switch from {a} to {b} ({count}×)")
        top_t = self.top_tools(3)
        for tool, count in top_t:
            if count >= 5:
                insights.append(f"You frequently use {tool} ({count} times)")
        return insights
