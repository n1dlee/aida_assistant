"""
tools/time_tool.py
Returns current date and time.
"""
from datetime import datetime
from tools.base_tool import BaseTool


class TimeTool(BaseTool):
    @property
    def name(self) -> str:
        return "get_time"

    @property
    def description(self) -> str:
        return "Returns the current local date and time."

    async def run(self, query: str, **kwargs) -> str:
        now = datetime.now()
        return now.strftime("Current time: %H:%M:%S, %A %d %B %Y")
