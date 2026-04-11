"""
tools/base_tool.py
Abstract base class for all AIDA tools.
To create a new tool: subclass BaseTool, implement name/description/run().
"""
from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'web_search'"""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description shown to the LLM."""

    @abstractmethod
    async def run(self, query: str, **kwargs) -> str:
        """Execute the tool and return a string result."""

    def __repr__(self):
        return f"<Tool:{self.name}>"
