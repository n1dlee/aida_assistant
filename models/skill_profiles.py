"""
models/skill_profiles.py
──────────────────────────
Skill injection system: loads specialised reasoning presets into the LLM context.

Skills differ from modes:
  - Modes are personality/tone profiles (how AIDA speaks and plans)
  - Skills are expertise domains (what AIDA knows and prioritises)

A skill can be active alongside any mode.
Multiple skills can be stacked (e.g. coding + security).

Each skill injects a domain-specific addendum into the system prompt,
biases tool selection, and provides relevant vocabulary to the router.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import logging

log = logging.getLogger("aida.skills")


@dataclass
class SkillProfile:
    name:        str
    label:       str
    emoji:       str
    domain:      str
    addendum:    str           # injected into system prompt
    keywords:    list[str]    # boosts routing for relevant queries
    tool_hints:  list[str]    # preferred tools for this domain
    references:  list[str] = field(default_factory=list)  # useful context URLs


_SKILLS: dict[str, SkillProfile] = {

    "none": SkillProfile(
        name="none", label="No skill", emoji="—",
        domain="general",
        addendum="",
        keywords=[], tool_hints=[],
    ),

    "coding": SkillProfile(
        name="coding", label="Software Engineering", emoji="💻",
        domain="software",
        addendum=(
            "You are operating in software engineering mode. "
            "Prefer idiomatic code. Think in terms of: algorithms, data structures, "
            "complexity, design patterns, testing, and maintainability. "
            "Reference Python, TypeScript, Rust, or whatever language is relevant. "
            "When reviewing code: prioritise correctness, then clarity, then performance."
        ),
        keywords=["function", "class", "bug", "error", "algorithm", "refactor",
                  "test", "api", "deploy", "git", "lint", "type"],
        tool_hints=["filesystem", "system"],
    ),

    "research": SkillProfile(
        name="research", label="Research Analyst", emoji="🔬",
        domain="research",
        addendum=(
            "You are operating in research analyst mode. "
            "Prioritise primary sources and verifiable data. "
            "Distinguish between facts, inferences, and speculation clearly. "
            "Summarise findings in structured formats: background, evidence, gaps, conclusions. "
            "Always cite source quality when relevant."
        ),
        keywords=["study", "paper", "research", "data", "evidence", "source",
                  "analysis", "statistics", "findings"],
        tool_hints=["web_search"],
    ),

    "writing": SkillProfile(
        name="writing", label="Writing Assistant", emoji="✍️",
        domain="writing",
        addendum=(
            "You are operating in writing assistant mode. "
            "Focus on clarity, flow, and voice. "
            "When editing: preserve the author's intent while improving expression. "
            "Offer specific rewrites rather than vague feedback. "
            "Know the difference between copyediting, structural editing, and style editing."
        ),
        keywords=["write", "draft", "edit", "revise", "proofread", "article",
                  "email", "essay", "blog", "story", "tone"],
        tool_hints=["filesystem"],
    ),

    "trading": SkillProfile(
        name="trading", label="Trading Analysis", emoji="📈",
        domain="finance",
        addendum=(
            "You are operating in trading analysis mode. "
            "Think in terms of: market structure, trend, momentum, risk/reward. "
            "Reference technical and fundamental concepts where relevant. "
            "Always note that this is analysis, not financial advice. "
            "Be precise about timeframes, entry/exit logic, and position sizing concepts."
        ),
        keywords=["price", "stock", "crypto", "market", "trade", "chart",
                  "support", "resistance", "breakout", "position", "risk"],
        tool_hints=["web_search"],
    ),

    "productivity": SkillProfile(
        name="productivity", label="Productivity Coach", emoji="⚡",
        domain="productivity",
        addendum=(
            "You are operating in productivity coaching mode. "
            "Think in terms of: prioritisation, focus blocks, systems, habits, leverage. "
            "Frameworks: GTD, Eisenhower matrix, time-blocking, 80/20. "
            "Be direct and action-biased. Help the user reduce friction. "
            "Identify the ONE most impactful next action."
        ),
        keywords=["task", "prioritise", "focus", "workflow", "habit",
                  "system", "todo", "productive", "plan", "schedule"],
        tool_hints=["calendar", "filesystem"],
    ),

    "security": SkillProfile(
        name="security", label="Security Analyst", emoji="🔐",
        domain="cybersecurity",
        addendum=(
            "You are operating in security analysis mode. "
            "Think in terms of: threat models, attack surfaces, mitigations, and defence-in-depth. "
            "Reference OWASP, CVE patterns, and security best practices. "
            "Never provide actual exploit code. Focus on understanding and defence."
        ),
        keywords=["security", "vulnerability", "exploit", "cve", "auth",
                  "permission", "injection", "xss", "csrf", "encryption"],
        tool_hints=["web_search"],
    ),
}


class SkillManager:
    def __init__(self):
        self._active: list[str] = []   # active skill names (stacked)

    def activate(self, skill_name: str) -> Optional[SkillProfile]:
        if skill_name not in _SKILLS:
            log.warning("Unknown skill: %r", skill_name)
            return None
        if skill_name == "none":
            self._active.clear()
            log.info("All skills cleared.")
            return _SKILLS["none"]
        if skill_name not in self._active:
            self._active.append(skill_name)
            log.info("Skill activated: %s", skill_name)
        return _SKILLS[skill_name]

    def deactivate(self, skill_name: str) -> None:
        if skill_name in self._active:
            self._active.remove(skill_name)
            log.info("Skill deactivated: %s", skill_name)

    def clear(self) -> None:
        self._active.clear()

    @property
    def active_skills(self) -> list[SkillProfile]:
        return [_SKILLS[n] for n in self._active if n in _SKILLS]

    def build_addendum(self) -> str:
        """Returns stacked skill addendums for system prompt injection."""
        parts = [s.addendum for s in self.active_skills if s.addendum]
        return ("\n\n[Active skills: " +
                ", ".join(s.label for s in self.active_skills) + "]\n" +
                "\n\n".join(parts)) if parts else ""

    def skill_keywords(self) -> list[str]:
        keywords = []
        for s in self.active_skills:
            keywords.extend(s.keywords)
        return keywords

    def preferred_tools(self) -> list[str]:
        tools = []
        for s in self.active_skills:
            for t in s.tool_hints:
                if t not in tools:
                    tools.append(t)
        return tools

    def list_skills(self) -> list[SkillProfile]:
        return list(_SKILLS.values())

    def skill_labels(self) -> list[str]:
        return [f"{s.emoji} {s.label}" for s in _SKILLS.values()]

    @staticmethod
    def from_label(label: str) -> Optional[str]:
        for name, profile in _SKILLS.items():
            if label in (name, profile.label, f"{profile.emoji} {profile.label}"):
                return name
        return None
