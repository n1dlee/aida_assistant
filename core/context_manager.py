"""
core/context_manager.py
────────────────────────
Collects ambient desktop context so AIDA can answer contextually.

Gathers (non-invasively):
  - active window title + process name
  - clipboard text (only when explicitly requested or user says "this")
  - recent app sequence (for pattern detection)
  - current working intent (set by orchestrator)

Privacy notes:
  - screen content is NOT captured
  - clipboard is read only when relevant
  - all data stays local
"""

from __future__ import annotations

import logging
import os
import platform
from collections import deque
from datetime import datetime
from typing import Optional

log = logging.getLogger("aida.context")


def _get_active_window() -> tuple[str, str]:
    """Returns (window_title, process_name). Best-effort, never raises."""
    title, process = "", ""
    try:
        if platform.system() == "Windows":
            import ctypes
            import ctypes.wintypes

            hwnd = ctypes.windll.user32.GetForegroundWindow()
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value

            import struct
            pid = ctypes.wintypes.DWORD()
            ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            PROCESS_QUERY_LIMITED = 0x1000
            h = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED, False, pid.value)
            if h:
                buf2 = ctypes.create_unicode_buffer(260)
                ctypes.windll.psapi.GetModuleFileNameExW(h, None, buf2, 260)
                ctypes.windll.kernel32.CloseHandle(h)
                process = os.path.basename(buf2.value)
        elif platform.system() == "Darwin":
            import subprocess
            r = subprocess.run(
                ["osascript", "-e",
                 'tell application "System Events" to get name of first process '
                 'whose frontmost is true'],
                capture_output=True, text=True, timeout=2
            )
            process = r.stdout.strip()
        # Linux: would use wnck or xdotool — scaffold only
    except Exception as exc:
        log.debug("active_window error: %s", exc)
    return title, process


def _get_clipboard() -> str:
    """Read clipboard text. Returns '' on any error."""
    try:
        if platform.system() == "Windows":
            import ctypes
            CF_UNICODETEXT = 13
            if not ctypes.windll.user32.OpenClipboard(0):
                return ""
            try:
                h = ctypes.windll.user32.GetClipboardData(CF_UNICODETEXT)
                if not h:
                    return ""
                ptr = ctypes.windll.kernel32.GlobalLock(h)
                if not ptr:
                    return ""
                text = ctypes.wstring_at(ptr)
                ctypes.windll.kernel32.GlobalUnlock(h)
                return text[:2000]
            finally:
                ctypes.windll.user32.CloseClipboard()
        else:
            import subprocess
            r = subprocess.run(
                ["pbpaste"] if platform.system() == "Darwin"
                else ["xclip", "-selection", "clipboard", "-o"],
                capture_output=True, text=True, timeout=2
            )
            return r.stdout[:2000]
    except Exception as exc:
        log.debug("clipboard error: %s", exc)
        return ""


class ContextSnapshot:
    def __init__(self,
                 window_title:  str,
                 process_name:  str,
                 clipboard:     str,
                 active_mode:   str,
                 recent_apps:   list[str],
                 user_intent:   str,
                 timestamp:     str):
        self.window_title  = window_title
        self.process_name  = process_name
        self.clipboard     = clipboard
        self.active_mode   = active_mode
        self.recent_apps   = recent_apps
        self.user_intent   = user_intent
        self.timestamp     = timestamp

    def to_prompt_fragment(self) -> str:
        """Compact summary injected into system prompt when relevant."""
        parts = []
        if self.process_name:
            parts.append(f"Active app: {self.process_name}")
        if self.window_title:
            parts.append(f"Window: {self.window_title[:80]}")
        if self.user_intent:
            parts.append(f"Current task: {self.user_intent}")
        if not parts:
            return ""
        return "[Desktop context: " + " | ".join(parts) + "]"

    def __repr__(self) -> str:
        return (f"ContextSnapshot(app={self.process_name!r}, "
                f"title={self.window_title[:40]!r})")


class ContextManager:
    def __init__(self, history_size: int = 20):
        self._app_history: deque[tuple[str, str]] = deque(maxlen=history_size)
        self._current_intent: str = ""
        self._last_snapshot: Optional[ContextSnapshot] = None

    def snapshot(self, include_clipboard: bool = False) -> ContextSnapshot:
        title, process = _get_active_window()
        clipboard = _get_clipboard() if include_clipboard else ""

        # Track app history
        if process and (not self._app_history or self._app_history[-1][0] != process):
            self._app_history.append((process, datetime.now().isoformat()))

        snap = ContextSnapshot(
            window_title  = title,
            process_name  = process,
            clipboard     = clipboard,
            active_mode   = "",         # filled by orchestrator
            recent_apps   = [a[0] for a in list(self._app_history)[-5:]],
            user_intent   = self._current_intent,
            timestamp     = datetime.now().isoformat(),
        )
        self._last_snapshot = snap
        return snap

    def set_intent(self, intent: str) -> None:
        self._current_intent = intent

    def recent_apps(self, n: int = 5) -> list[str]:
        return [a[0] for a in list(self._app_history)[-n:]]

    @property
    def last_snapshot(self) -> Optional[ContextSnapshot]:
        return self._last_snapshot

    def clipboard_text(self) -> str:
        return _get_clipboard()

    def resolve_this(self, text: str) -> str:
        """
        When user says 'explain this' or 'what is this',
        resolve 'this' to clipboard content or window title.
        """
        import re
        if re.search(r"\bthis\b", text, re.IGNORECASE):
            clip = _get_clipboard()
            if clip.strip():
                return clip[:500]
            snap = self.snapshot()
            if snap.window_title:
                return snap.window_title
        return ""
