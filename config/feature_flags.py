"""
config/feature_flags.py
────────────────────────
Single source of truth for all toggleable AIDA features.
Reads from environment variables and config/settings.yaml.
All flags default to safe/conservative values.
"""

import os
import yaml
from pathlib import Path

_SETTINGS = Path(__file__).parent / "settings.yaml"


def _load_yaml() -> dict:
    try:
        return yaml.safe_load(_SETTINGS.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _flag(env_var: str, yaml_path: list, default: bool) -> bool:
    """Resolve a boolean flag: env var > yaml > default."""
    env = os.getenv(env_var)
    if env is not None:
        return env.lower() in ("1", "true", "yes")
    cfg = _load_yaml()
    val = cfg
    for key in yaml_path:
        if not isinstance(val, dict):
            return default
        val = val.get(key)
        if val is None:
            return default
    return bool(val)


class Flags:
    """All feature flags. Access as Flags.SHADOW_MODE, etc."""

    # ── Voice ─────────────────────────────────────────────────────────────────
    VOICE_ENABLED           = _flag("AIDA_VOICE",          ["voice", "enabled"],             True)
    TTS_ENABLED             = _flag("AIDA_TTS",            ["voice", "tts_enabled"],          True)

    # ── Memory ────────────────────────────────────────────────────────────────
    LONG_TERM_MEMORY        = _flag("AIDA_LONG_TERM_MEM",  ["memory", "long_term_enabled"],   True)
    EPISODIC_MEMORY         = _flag("AIDA_EPISODIC",       ["memory", "episodic_enabled"],    True)

    # ── Behaviour modes ───────────────────────────────────────────────────────
    MODE_SYSTEM             = _flag("AIDA_MODES",          ["modes", "enabled"],              True)

    # ── Planning ──────────────────────────────────────────────────────────────
    PLANNER_ENABLED         = _flag("AIDA_PLANNER",        ["planner", "enabled"],            True)
    GOAL_MODE               = _flag("AIDA_GOAL_MODE",      ["planner", "goal_mode"],          True)

    # ── Shadow & predictive (OFF by default — privacy-sensitive) ─────────────
    SHADOW_MODE             = _flag("AIDA_SHADOW",         ["shadow", "enabled"],             False)
    PREDICTIVE_SUGGESTIONS  = _flag("AIDA_PREDICTIVE",     ["predictive", "enabled"],         False)

    # ── Dual brain ────────────────────────────────────────────────────────────
    DUAL_BRAIN              = _flag("AIDA_DUAL_BRAIN",     ["dual_brain", "enabled"],         True)

    # ── Self-improvement logging ──────────────────────────────────────────────
    ACTION_LOGGING          = _flag("AIDA_ACTION_LOG",     ["logging", "action_log"],         True)
    SUGGESTION_LOGGING      = _flag("AIDA_SUGGEST_LOG",    ["logging", "suggestion_log"],     True)

    # ── UI ────────────────────────────────────────────────────────────────────
    SHOW_PLAN_PANEL         = _flag("AIDA_PLAN_PANEL",     ["ui", "plan_panel"],              True)
    SHOW_CONTEXT_PANEL      = _flag("AIDA_CTX_PANEL",      ["ui", "context_panel"],           True)
