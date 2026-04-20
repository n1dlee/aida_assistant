"""
core/mode_manager.py
─────────────────────
Manages AIDA's behavioural mode (personality profile).

Modes affect:
  - system prompt tone and style
  - verbosity
  - tool preferences
  - planning style
  - suggestion aggressiveness

Built-in modes:
  default     — balanced, warm, concise
  analytical  — precise, structured, evidence-based
  creative    — open-ended, lateral, generative
  coding      — technical, terse, code-first
  study       — pedagogical, patient, quiz-ready
  founder     — startup-minded, action-biased, growth-focused
  focus       — minimal, task-only, no smalltalk
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("aida.mode_manager")


@dataclass
class ModeProfile:
    name: str
    label: str
    emoji: str
    system_addendum: str           # appended to base system prompt
    verbosity: str                 # "terse" | "normal" | "verbose"
    temperature_hint: float        # 0.0–1.0 (hint passed to model)
    tool_priority: list[str]       # preferred tools first
    planning_style: str            # "sequential" | "agile" | "creative"
    tags: list[str] = field(default_factory=list)


_PROFILES: dict[str, ModeProfile] = {
    "default": ModeProfile(
        name="default", label="Default", emoji="🤖",
        system_addendum=(
            "Be concise and warm. Speak the user's language. "
            "Balance helpfulness with brevity."
        ),
        verbosity="normal", temperature_hint=0.7,
        tool_priority=[], planning_style="sequential",
        tags=["general"],
    ),
    "analytical": ModeProfile(
        name="analytical", label="Analytical", emoji="🔬",
        system_addendum=(
            "Be precise, structured, and evidence-based. "
            "Use bullet points and numbered steps. Avoid speculation. "
            "When uncertain, state your confidence level explicitly."
        ),
        verbosity="verbose", temperature_hint=0.3,
        tool_priority=["web_search", "filesystem"],
        planning_style="sequential",
        tags=["research", "analysis"],
    ),
    "creative": ModeProfile(
        name="creative", label="Creative", emoji="🎨",
        system_addendum=(
            "Be imaginative and open-ended. Offer lateral perspectives. "
            "Embrace ambiguity. Generate multiple options rather than one answer. "
            "Think out loud and explore unconventional paths."
        ),
        verbosity="verbose", temperature_hint=0.9,
        tool_priority=[], planning_style="creative",
        tags=["brainstorm", "writing", "ideas"],
    ),
    "coding": ModeProfile(
        name="coding", label="Coding", emoji="💻",
        system_addendum=(
            "Be technical and terse. Prioritise working code over explanation. "
            "Use language-appropriate idioms. "
            "Assume the user is a developer. Skip pleasantries."
        ),
        verbosity="terse", temperature_hint=0.2,
        tool_priority=["filesystem", "system"],
        planning_style="sequential",
        tags=["code", "dev", "technical"],
    ),
    "study": ModeProfile(
        name="study", label="Study", emoji="📚",
        system_addendum=(
            "Be pedagogical and patient. Break concepts into digestible steps. "
            "Use analogies and examples. Offer to quiz the user. "
            "Reinforce key ideas at the end of each answer."
        ),
        verbosity="verbose", temperature_hint=0.6,
        tool_priority=["web_search"],
        planning_style="sequential",
        tags=["learning", "education"],
    ),
    "founder": ModeProfile(
        name="founder", label="Founder", emoji="🚀",
        system_addendum=(
            "Be action-biased, startup-minded, and growth-focused. "
            "Think in terms of leverage, speed, and user value. "
            "Be direct. Prioritise execution over perfection. "
            "Use frameworks like first-principles, 80/20, and PMF."
        ),
        verbosity="terse", temperature_hint=0.7,
        tool_priority=["web_search", "calendar"],
        planning_style="agile",
        tags=["startup", "business", "productivity"],
    ),
    "focus": ModeProfile(
        name="focus", label="Focus", emoji="🎯",
        system_addendum=(
            "Minimal mode. Respond only to the task at hand. "
            "No filler, no suggestions, no smalltalk. "
            "Pure signal."
        ),
        verbosity="terse", temperature_hint=0.4,
        tool_priority=[], planning_style="sequential",
        tags=["minimal", "deep-work"],
    ),
}


class ModeManager:
    def __init__(self, initial_mode: str = "default"):
        self._current: str = initial_mode if initial_mode in _PROFILES else "default"
        log.info("ModeManager initialized: mode=%s", self._current)

    @property
    def current(self) -> ModeProfile:
        return _PROFILES[self._current]

    @property
    def current_name(self) -> str:
        return self._current

    def set_mode(self, mode_name: str) -> ModeProfile:
        if mode_name not in _PROFILES:
            log.warning("Unknown mode %r — keeping %r", mode_name, self._current)
            return self.current
        self._current = mode_name
        log.info("Mode switched → %s", mode_name)
        return self.current

    def list_modes(self) -> list[ModeProfile]:
        return list(_PROFILES.values())

    def mode_names(self) -> list[str]:
        return list(_PROFILES.keys())

    def mode_labels(self) -> list[str]:
        """Returns 'emoji label' strings for UI display."""
        return [f"{p.emoji} {p.label}" for p in _PROFILES.values()]

    def build_addendum(self) -> str:
        """Returns the system prompt addendum for the current mode."""
        p = self.current
        verbosity_note = {
            "terse":   "Keep responses short — 1-3 sentences unless depth is essential.",
            "normal":  "Keep responses concise but complete.",
            "verbose": "Provide thorough answers with structure and examples.",
        }.get(p.verbosity, "")
        return f"\n\n[Mode: {p.label}]\n{p.system_addendum}\n{verbosity_note}"

    @staticmethod
    def from_label(label: str) -> Optional[str]:
        """Map 'emoji label' string back to mode key."""
        for name, profile in _PROFILES.items():
            if label in (name, profile.label, f"{profile.emoji} {profile.label}"):
                return name
        return None
