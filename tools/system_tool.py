"""
tools/system_tool.py
Runs safe OS-level commands (open apps, list files, etc.)
"""
import subprocess
import sys
import logging
from tools.base_tool import BaseTool

log = logging.getLogger("aida.tools.system")

SAFE_COMMANDS = {
    "notepad": ["notepad.exe"],
    "calculator": ["calc.exe"],
    "browser": ["start", "chrome"] if sys.platform == "win32" else ["xdg-open", "https://google.com"],
    "explorer": ["explorer.exe"],
    "terminal": ["cmd.exe"] if sys.platform == "win32" else ["xterm"],
}


class SystemTool(BaseTool):
    @property
    def name(self) -> str:
        return "system"

    @property
    def description(self) -> str:
        return "Open or run applications on the OS."

    async def run(self, query: str, **kwargs) -> str:
        lower = query.lower()
        for keyword, cmd in SAFE_COMMANDS.items():
            if keyword in lower:
                try:
                    subprocess.Popen(cmd, shell=(sys.platform == "win32"))
                    return f"Opened {keyword}."
                except Exception as e:
                    log.error("Failed to open %s: %s", keyword, e)
                    return f"Could not open {keyword}: {e}"
        return "I don't have a safe command mapped for that request."
