"""
tools/registry.py
Auto-discovers all BaseTool subclasses and routes execution.
"""
import logging
from typing import Optional

from tools.base_tool import BaseTool
from tools.system_tool import SystemTool
from tools.web_tool import WebTool
from tools.calendar_tool import CalendarTool
from tools.time_tool import TimeTool

log = logging.getLogger("aida.tools.registry")


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._register_defaults()

    def _register_defaults(self):
        for tool in [SystemTool(), WebTool(), CalendarTool(), TimeTool()]:
            self.register(tool)

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool
        log.debug("Registered tool: %s", tool.name)

    async def execute(self, user_input: str, llm_response: str) -> Optional[str]:
        """Simple keyword-based dispatcher. In a full build, use LLM function-calling."""
        lower = user_input.lower()

        if any(w in lower for w in ["time", "clock", "час", "время"]):
            return await self._tools["get_time"].run(user_input)

        if any(w in lower for w in ["weather", "погода"]):
            return await self._tools["web_search"].run(f"weather {user_input}")

        if any(w in lower for w in ["search", "find online", "найди в интернете", "поиск"]):
            return await self._tools["web_search"].run(user_input)

        if any(w in lower for w in ["open", "run", "execute", "открой", "запусти"]):
            return await self._tools["system"].run(user_input)

        return None  # No tool matched; return LLM response as-is

    def list_tools(self) -> list[str]:
        return [f"{t.name}: {t.description}" for t in self._tools.values()]
